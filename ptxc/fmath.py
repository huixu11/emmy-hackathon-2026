"""Elementwise f32 operations: one emitter per IR op name.

Rounding is explicit (`add.rn`, `mul.rn`, ...) on purpose: without a rounding modifier ptxas may
contract a multiply and an add into one fma, which rounds once instead of twice and no longer
matches numpy's float32 result bit for bit (the pointwise problems check with atol = rtol = 0).
`div.rn`, `rcp.rn` and `sqrt.rn` are the IEEE-exact forms for the same reason.

PTX has no exp instruction; `ex2.approx` computes 2^x to about 2 ulp. `exp` below rebuilds
x * log2(e) in extra precision so the result stays near that accuracy for any |x|.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable

from .errors import CompileError
from .ptx import PtxBuilder, f32_imm

# ---- one instruction ----------------------------------------------------------------------

_DIRECT: dict[str, tuple[str, int]] = {
    "add": ("add.rn.f32", 2),
    "subtract": ("sub.rn.f32", 2),
    "multiply": ("mul.rn.f32", 2),
    "divide": ("div.rn.f32", 2),
    "maximum": ("max.f32", 2),
    "minimum": ("min.f32", 2),
    "negative": ("neg.f32", 1),
    "abs": ("abs.f32", 1),
    "sqrt": ("sqrt.rn.f32", 1),
    "reciprocal": ("rcp.rn.f32", 1),
    "copy": ("mov.f32", 1),
    "pad": ("mov.f32", 1),  # a zero-width pad: the identity
}


# ---- several instructions -----------------------------------------------------------------

def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


_LOG2E = _f32(math.log2(math.e))
_LOG2E_LO = _f32(math.log2(math.e) - _LOG2E)  # what rounding log2(e) to f32 dropped
_LN2 = _f32(math.log(2.0))
ZERO = f32_imm(0.0)
ONE = f32_imm(1.0)
NAN = "0f7FFFFFFF"
NEG_INF = "0fFF800000"
POS_INF = "0f7F800000"


def _regs(b: PtxBuilder, n: int) -> list[str]:
    return [b.reg("f32") for _ in range(n)]


def _poly(b: PtxBuilder, u: str, coeffs: list[float]) -> str:
    """c0 + c1*u + ... + cn*u^n by Horner's rule, one fma per coefficient; returns the register."""
    acc = b.reg("f32")
    b.emit("fma.rn.f32", acc, u, f32_imm(coeffs[-1]), f32_imm(coeffs[-2]))
    for c in reversed(coeffs[:-2]):
        nxt = b.reg("f32")
        b.emit("fma.rn.f32", nxt, acc, u, f32_imm(c))
        acc = nxt
    return acc


def _relu(b: PtxBuilder, dst: str, x: str) -> None:
    b.emit("max.f32", dst, x, ZERO)


def _square(b: PtxBuilder, dst: str, x: str) -> None:
    b.emit("mul.rn.f32", dst, x, x)


def _rsqrt(b: PtxBuilder, dst: str, x: str) -> None:
    # 1 / sqrt(x) with both steps correctly rounded, exactly what numpy computes.
    root = b.reg("f32")
    b.emit("sqrt.rn.f32", root, x)
    b.emit("rcp.rn.f32", dst, root)


def _exp(b: PtxBuilder, dst: str, x: str) -> None:
    # exp(x) = 2^t with t = x*log2(e). Split t = hi + lo exactly enough: hi = round(x*L), and
    # lo collects the product's rounding error (fma) plus x times the error of L itself.
    # Then exp(x) = 2^hi * 2^lo ~= 2^hi * (1 + lo*ln2), with 2^hi from ex2.approx.
    # x is first clamped to [-104, 89] (NaN passes): past those the result is 0 or inf anyway,
    # and the split stays finite, so the correction needs no special cases. Results below
    # 2^-126 flush to zero (.ftz), which spares the subnormal handling around ex2.
    xc, hi, neg_hi, lo, p2, corr = _regs(b, 6)
    b.emit("max.NaN.f32", xc, x, f32_imm(-104.0))
    b.emit("min.NaN.f32", xc, xc, f32_imm(89.0))
    b.emit("mul.rn.f32", hi, xc, f32_imm(_LOG2E))
    b.emit("neg.f32", neg_hi, hi)
    b.emit("fma.rn.f32", lo, xc, f32_imm(_LOG2E), neg_hi)
    b.emit("fma.rn.f32", lo, xc, f32_imm(_LOG2E_LO), lo)
    b.emit("ex2.approx.ftz.f32", p2, hi)
    b.emit("mul.rn.f32", corr, lo, f32_imm(_LN2))
    b.emit("fma.rn.f32", dst, p2, corr, p2)


def _exp_fast(b: PtxBuilder, dst: str, x: str) -> None:
    # CUDA's __expf: one multiply and the hardware 2^x.
    t = b.reg("f32")
    b.emit("mul.f32", t, x, f32_imm(_LOG2E))
    b.emit("ex2.approx.ftz.f32", dst, t)


def _sigmoid(b: PtxBuilder, dst: str, x: str) -> None:
    # 1 / (1 + exp(-x)); exp(-x) = inf for very negative x gives exactly 0.
    neg_x, e, denom = (b.reg("f32") for _ in range(3))
    b.emit("neg.f32", neg_x, x)
    _exp(b, e, neg_x)
    b.emit("add.rn.f32", denom, e, ONE)
    b.emit("rcp.rn.f32", dst, denom)


def _silu(b: PtxBuilder, dst: str, x: str) -> None:
    # x / (1 + exp(-x)), the same division numpy performs.
    neg_x, e, denom = _regs(b, 3)
    b.emit("neg.f32", neg_x, x)
    _exp(b, e, neg_x)
    b.emit("add.rn.f32", denom, e, ONE)
    b.emit("div.rn.f32", dst, x, denom)


# tanh(x) = x + x^3 * P(x^2) for |x| < 0.55: P is a relative-error fit of (tanh(x) - x) / x^3,
# 0.7 ulp in f32 arithmetic. Beyond 0.55, tanh >= 0.5 and 1 - 2/(exp(2|x|) + 1) loses nothing.
_TANH_SPLIT = 0.55
_TANH_P = [-0.33333316445350647, 0.13332585990428925, -0.05385231226682663,
           0.02107168920338154, -0.006274253595620394]


def _tanh(b: PtxBuilder, dst: str, x: str) -> None:
    ax, two_ax, e, denom, q, big, signed, u, pu, small = _regs(b, 10)
    use_big = b.reg("pred")
    b.emit("abs.f32", ax, x)
    b.emit("add.rn.f32", two_ax, ax, ax)
    _exp(b, e, two_ax)
    b.emit("add.rn.f32", denom, e, ONE)
    b.emit("rcp.rn.f32", q, denom)
    b.emit("fma.rn.f32", big, q, f32_imm(-2.0), ONE)  # 1 - 2q, rounded once
    b.emit("copysign.f32", signed, x, big)  # copysign d, a, b: |b| with the sign of a
    b.emit("mul.rn.f32", u, x, x)
    b.emit("mul.rn.f32", pu, _poly(b, u, _TANH_P), u)
    b.emit("fma.rn.f32", small, pu, x, x)
    b.emit("setp.ge.f32", use_big, ax, f32_imm(_TANH_SPLIT))
    b.emit("selp.f32", dst, signed, small, use_big)


# erf(x) = x * P(x^2) for |x| < 1 (2.3 ulp), and erf(x) = 1 - exp(-x^2) * R(|x|) beyond, where
# R fits erfc(x) * exp(x^2) on [1, 4] (1.5 ulp). Past 4 (in fact past 3.92) erf rounds to +-1.
_ERF_P = [1.1283791065216064, -0.3761232793331146, 0.11280184239149094,
          -0.02671140432357788, 0.004917643498629332, -0.0005631754756905138]
_ERF_R = [0.9924924969673157, -1.0832406282424927, 0.876099705696106, -0.5447275042533875,
          0.2598664462566376, -0.09362982213497162, 0.02486923523247242, -0.004697802010923624,
          0.0005950689082965255, -4.5203360059531406e-05, 1.5538399793513236e-06]


def _erf(b: PtxBuilder, dst: str, x: str) -> None:
    ax, u, small, axc, uc, neg_uc, e, er, big, signed = _regs(b, 10)
    use_small = b.reg("pred")
    b.emit("abs.f32", ax, x)
    b.emit("mul.rn.f32", u, x, x)
    b.emit("mul.rn.f32", small, _poly(b, u, _ERF_P), x)
    b.emit("min.NaN.f32", axc, ax, f32_imm(4.0))  # keep R inside its fitted range
    b.emit("mul.rn.f32", uc, axc, axc)
    b.emit("neg.f32", neg_uc, uc)
    _exp(b, e, neg_uc)
    b.emit("mul.rn.f32", er, e, _poly(b, axc, _ERF_R))
    b.emit("sub.rn.f32", big, ONE, er)
    b.emit("copysign.f32", signed, x, big)
    b.emit("setp.lt.f32", use_small, ax, ONE)
    b.emit("selp.f32", dst, small, signed, use_small)


def _gelu(b: PtxBuilder, dst: str, x: str) -> None:
    # 0.5 * x * (1 + erf(x / sqrt(2)))
    t, e, one_e, half_x = _regs(b, 4)
    b.emit("mul.rn.f32", t, x, f32_imm(math.sqrt(0.5)))
    _erf(b, e, t)
    b.emit("add.rn.f32", one_e, e, ONE)
    b.emit("mul.rn.f32", half_x, x, f32_imm(0.5))
    b.emit("mul.rn.f32", dst, half_x, one_e)


def _gelu_tanh(b: PtxBuilder, dst: str, x: str) -> None:
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x2, x3, inner, t, th, one_t, half_x = _regs(b, 7)
    b.emit("mul.rn.f32", x2, x, x)
    b.emit("mul.rn.f32", x3, x2, x)
    b.emit("fma.rn.f32", inner, x3, f32_imm(0.044715), x)
    b.emit("mul.rn.f32", t, inner, f32_imm(math.sqrt(2.0 / math.pi)))
    _tanh(b, th, t)
    b.emit("add.rn.f32", one_t, th, ONE)
    b.emit("mul.rn.f32", half_x, x, f32_imm(0.5))
    b.emit("mul.rn.f32", dst, half_x, one_t)


# Natural log by the classic fdlibm reduction (as in e_logf.c): x = 2^k * m with m in
# [sqrt(2)/2, sqrt(2)), f = m - 1, s = f / (2 + f), and
#   log(x) = k*ln2 + f - f^2/2 + s * (f^2/2 + R(s^2)),
# with fdlibm's coefficients for R and ln2 split into an exact head and a tail. Under 1 ulp.
_LG1, _LG2, _LG3, _LG4 = (float.fromhex(h) for h in
                          ("0xaaaaaa.0p-24", "0xccce13.0p-25", "0x91e9ee.0p-25", "0xf89e26.0p-26"))
_LN2_HI, _LN2_LO = _from_bits(0x3F317180), _from_bits(0x3717F7D1)
_SQRT_HALF_BITS = 0x3F3504F3


def _log(b: PtxBuilder, dst: str, x: str) -> None:
    scaled, xs, m, f, two_f, s, z, w, t1, t2, r, half_f, hfsq, dk, acc, fallback = _regs(b, 16)
    bits, ix, k_raw, k_sub, k, mant = (b.reg("b32") for _ in range(6))
    tiny, positive, finite, ok = (b.reg("pred") for _ in range(4))
    # Subnormal x: scale into the normal range first, and take the 25 back out of k.
    b.emit("setp.lt.f32", tiny, x, f32_imm(2.0 ** -126))
    b.emit("mul.rn.f32", scaled, x, f32_imm(2.0 ** 25))
    b.emit("selp.f32", xs, scaled, x, tiny)
    b.emit("mov.b32", bits, xs)
    b.emit("sub.s32", ix, bits, hex(_SQRT_HALF_BITS))
    b.emit("shr.s32", k_raw, ix, 23)
    b.emit("sub.s32", k_sub, k_raw, 25)
    b.emit("selp.s32", k, k_sub, k_raw, tiny)
    b.emit("and.b32", mant, ix, "0x007FFFFF")
    b.emit("add.s32", mant, mant, hex(_SQRT_HALF_BITS))
    b.emit("mov.b32", m, mant)
    b.emit("sub.rn.f32", f, m, ONE)
    b.emit("add.rn.f32", two_f, f, f32_imm(2.0))
    b.emit("div.rn.f32", s, f, two_f)
    b.emit("mul.rn.f32", z, s, s)
    b.emit("mul.rn.f32", w, z, z)
    b.emit("fma.rn.f32", t1, w, f32_imm(_LG4), f32_imm(_LG2))
    b.emit("mul.rn.f32", t1, t1, w)
    b.emit("fma.rn.f32", t2, w, f32_imm(_LG3), f32_imm(_LG1))
    b.emit("mul.rn.f32", t2, t2, z)
    b.emit("add.rn.f32", r, t2, t1)
    b.emit("mul.rn.f32", half_f, f, f32_imm(0.5))
    b.emit("mul.rn.f32", hfsq, half_f, f)
    b.emit("cvt.rn.f32.s32", dk, k)
    b.emit("add.rn.f32", acc, hfsq, r)
    b.emit("mul.rn.f32", acc, s, acc)
    b.emit("fma.rn.f32", acc, dk, f32_imm(_LN2_LO), acc)
    b.emit("sub.rn.f32", acc, acc, hfsq)
    b.emit("add.rn.f32", acc, acc, f)
    b.emit("fma.rn.f32", acc, dk, f32_imm(_LN2_HI), acc)  # k * ln2_hi is exact
    # x <= 0, inf and NaN: lg2.approx already gives the IEEE answer (NaN, -inf, inf, NaN).
    b.emit("setp.gt.f32", positive, x, ZERO)
    b.emit("testp.finite.f32", finite, x)
    b.emit("and.pred", ok, positive, finite)
    b.emit("lg2.approx.f32", fallback, x)
    b.emit("selp.f32", dst, acc, fallback, ok)


# sin and cos as in cephes' sinf.c: j = |x| / (pi/4) truncated, then rounded up to even, and
# r = |x| - j*pi/4 in [-pi/4, pi/4], with pi/4 split into three f32 parts (one fma each, so the
# reduction stays accurate far past |x| = 1e4). Then cephes' minimax polynomial for sin or cos
# on r; j mod 8 picks the polynomial and the sign. Relative error < 2.4e-7 up to |x| = 1e4.
_PI4_HI = _f32(math.pi / 4)
_PI4_MID = _f32(math.pi / 4 - _PI4_HI)
_PI4_LO = _f32(math.pi / 4 - _PI4_HI - _PI4_MID)
_SIN_P = [-1.6666654611e-1, 8.3321608736e-3, -1.9515295891e-4]
_COS_P = [4.166664568298827e-2, -1.388731625493765e-3, 2.443315711809948e-5]


def _sincos(cos: bool) -> Callable[..., None]:
    def emit(b: PtxBuilder, dst: str, x: str) -> None:
        ax, t, y, r, z, ps, s, zz, h, c, v = _regs(b, 11)
        j, odd, quad, sign, bits = (b.reg("b32") for _ in range(5))
        use_cos = b.reg("pred")
        b.emit("abs.f32", ax, x)
        b.emit("mul.rn.f32", t, ax, f32_imm(4 / math.pi))
        b.emit("cvt.rzi.s32.f32", j, t)
        b.emit("and.b32", odd, j, 1)
        b.emit("add.s32", j, j, odd)
        b.emit("cvt.rn.f32.s32", y, j)
        b.emit("fma.rn.f32", r, y, f32_imm(-_PI4_HI), ax)
        b.emit("fma.rn.f32", r, y, f32_imm(-_PI4_MID), r)
        b.emit("fma.rn.f32", r, y, f32_imm(-_PI4_LO), r)
        if cos:
            b.emit("add.s32", j, j, 2)  # cos|x| = sin(|x| + pi/2)
        b.emit("mul.rn.f32", z, r, r)
        b.emit("mul.rn.f32", ps, _poly(b, z, _SIN_P), z)
        b.emit("fma.rn.f32", s, ps, r, r)  # r + r^3 P(r^2)
        b.emit("mul.rn.f32", zz, z, z)
        b.emit("fma.rn.f32", h, z, f32_imm(-0.5), ONE)
        b.emit("fma.rn.f32", c, _poly(b, z, _COS_P), zz, h)  # 1 - r^2/2 + r^4 Q(r^2)
        b.emit("and.b32", quad, j, 2)
        b.emit("setp.ne.s32", use_cos, quad, 0)
        b.emit("selp.f32", v, c, s, use_cos)
        b.emit("and.b32", sign, j, 4)
        b.emit("shl.b32", sign, sign, 29)  # j & 4 flips the sign bit
        if not cos:  # sin is odd: x's sign as well
            xs = b.reg("b32")
            b.emit("mov.b32", xs, x)
            b.emit("and.b32", xs, xs, "0x80000000")
            b.emit("xor.b32", sign, sign, xs)
        b.emit("mov.b32", bits, v)
        b.emit("xor.b32", bits, bits, sign)
        b.emit("mov.b32", dst, bits)
    return emit


_POW_INT_LIMIT = 64  # integral exponents up to this go through repeated squaring


def _pow(b: PtxBuilder, dst: str, x: str, y: str) -> None:
    # |x|^y = exp(y * log|x|) in general. An integral exponent up to 64 in magnitude (x**3,
    # x**-1, ...) multiplies by repeated squaring instead: a handful of correctly rounded
    # products, far closer than log/exp. Negative x takes the sign of an odd integral y and is
    # NaN for a fractional one; y == 0 and x == 1 give 1 for every x and y, as in C's powf.
    known = _imm_value(y)
    if known is not None and math.isfinite(known):
        _pow_known(b, dst, x, known)
        return
    ax, lg, t, general, ay, y_trunc, power, base, inv, mag, half, half_trunc, res = _regs(b, 13)
    n, bit = b.reg("b32"), b.reg("b32")
    integral, in_range, use_int, take, negative_y, odd, x_neg, flip, nan = (b.reg("pred") for _ in range(9))
    b.emit("abs.f32", ax, x)
    _log(b, lg, ax)
    b.emit("mul.rn.f32", t, y, lg)
    _exp(b, general, t)

    b.emit("cvt.rzi.f32.f32", y_trunc, y)
    b.emit("setp.eq.f32", integral, y_trunc, y)
    b.emit("abs.f32", ay, y)
    b.emit("setp.le.f32", in_range, ay, f32_imm(_POW_INT_LIMIT))
    b.emit("and.pred", use_int, integral, in_range)
    b.emit("cvt.rzi.u32.f32", n, ay)
    b.emit("mov.f32", power, ONE)
    b.emit("mov.f32", base, ax)
    for i in range(_POW_INT_LIMIT.bit_length()):
        b.emit("and.b32", bit, n, 1 << i)
        b.emit("setp.ne.u32", take, bit, 0)
        b.emit("mul.rn.f32", power, power, base, guard=take)
        b.emit("mul.rn.f32", base, base, base)
    b.emit("setp.lt.f32", negative_y, y, ZERO)
    b.emit("rcp.rn.f32", inv, power)
    b.emit("selp.f32", power, inv, power, negative_y)
    b.emit("selp.f32", mag, power, general, use_int)

    b.emit("mul.rn.f32", half, y, f32_imm(0.5))
    b.emit("cvt.rzi.f32.f32", half_trunc, half)
    b.emit("setp.ne.f32", odd, half_trunc, half)
    b.emit("setp.lt.f32", x_neg, x, ZERO)
    b.emit("and.pred", flip, x_neg, odd)
    b.emit("and.pred", flip, flip, integral)
    b.emit("mov.f32", res, mag)
    b.emit("neg.f32", res, mag, guard=flip)
    b.emit("and.pred", nan, x_neg, "!" + integral)
    b.emit("setp.ne.and.f32", nan, x, NEG_INF, nan)  # (-inf)^y is +inf or +0, not NaN
    b.emit("mov.f32", res, NAN, guard=nan)
    unit, y_inf = b.reg("pred"), b.reg("pred")
    b.emit("setp.eq.f32", y_inf, ay, POS_INF)
    b.emit("setp.eq.and.f32", unit, ax, ONE, y_inf)  # (-1)^+-inf is 1
    b.emit("setp.eq.or.f32", unit, x, ONE, unit)  # 1^y is 1, for y = NaN too
    b.emit("setp.eq.or.f32", unit, y, ZERO, unit)
    b.emit("mov.f32", res, ONE, guard=unit)
    b.emit("mov.f32", dst, res)


def _pow_known(b: PtxBuilder, dst: str, x: str, y: float) -> None:
    """pow(x, y) for an exponent the kernel fixes (a constant of the graph): only the path that
    y takes, about a tenth of the instructions. Special values follow C's powf as above."""
    if y == 0:
        b.emit("mov.f32", dst, ONE)  # for every x, NaN included
        return
    if y == int(y) and abs(y) <= _POW_INT_LIMIT:
        # x itself through the squarings, so an odd power keeps its sign; the products come in
        # the order of the general path, so results are the same bits.
        b.emit("rcp.rn.f32" if y < 0 else "mov.f32", dst, _int_power(b, x, int(abs(y))))
        return
    ax = b.reg("f32")
    b.emit("abs.f32", ax, x)
    if y * 2 == int(y * 2) and abs(y) <= _POW_INT_LIMIT:
        # y = +-(k + 1/2): |x|^k * sqrt|x|, all correctly rounded steps (x**1.5, x**-0.5)
        mag = b.reg("f32")
        b.emit("sqrt.rn.f32", mag, ax)
        if abs(y) > 1:
            prod = b.reg("f32")
            b.emit("mul.rn.f32", prod, _int_power(b, ax, int(abs(y))), mag)
            mag = prod
        if y < 0:
            inv = b.reg("f32")
            b.emit("rcp.rn.f32", inv, mag)
            mag = inv
    else:
        lg, t, mag = _regs(b, 3)
        _log(b, lg, ax)
        b.emit("mul.rn.f32", t, f32_imm(y), lg)
        _exp(b, mag, t)
    if y == int(y):  # an integer past the squaring limit
        if int(y) % 2:  # odd: x's sign
            xs, bits = b.reg("b32"), b.reg("b32")
            b.emit("mov.b32", xs, x)
            b.emit("and.b32", xs, xs, "0x80000000")
            b.emit("mov.b32", bits, mag)
            b.emit("or.b32", bits, bits, xs)
            b.emit("mov.b32", dst, bits)
        else:
            b.emit("mov.f32", dst, mag)
        return
    neg, nan = b.reg("pred"), b.reg("pred")
    b.emit("setp.lt.f32", neg, x, ZERO)
    b.emit("setp.ne.and.f32", nan, x, NEG_INF, neg)  # negative x is NaN, but (-inf)^y is not
    b.emit("selp.f32", dst, NAN, mag, nan)


def _int_power(b: PtxBuilder, x: str, n: int) -> str:
    """x^n for n >= 1 by repeated squaring: the operand holding it."""
    power, base = None, x
    while True:
        if n & 1:
            if power is None:
                power = base
            else:
                prod = b.reg("f32")
                b.emit("mul.rn.f32", prod, power, base)
                power = prod
        n >>= 1
        if not n:
            return power
        square = b.reg("f32")
        b.emit("mul.rn.f32", square, base, base)
        base = square


def _imm_value(operand: str) -> float | None:
    """The value of an f32 immediate operand (0f3FC00000 -> 1.5); None for a register."""
    if operand.startswith("0f") and len(operand) == 10:
        return _from_bits(int(operand[2:], 16))
    return None


# Comparisons and logic on f32 masks (nonzero is true), producing 1.0 or 0.0 like C's (float)(a < b).
_COMPARE = {"equal": "eq", "not_equal": "neu", "greater": "gt", "less": "lt",
            "greater_equal": "ge", "less_equal": "le"}


def _comparison(cmp: str) -> Callable[..., None]:
    def emit(b: PtxBuilder, dst: str, a: str, c: str) -> None:
        b.emit(f"set.{cmp}.f32.f32", dst, a, c)
    return emit


def _logical(op: str) -> Callable[..., None]:
    def emit(b: PtxBuilder, dst: str, a: str, c: str) -> None:
        p, q, r = (b.reg("pred") for _ in range(3))
        b.emit("setp.neu.f32", p, a, ZERO)
        b.emit("setp.neu.f32", q, c, ZERO)
        b.emit(f"{op}.pred", r, p, q)
        b.emit("selp.f32", dst, ONE, ZERO, r)
    return emit


def _logical_not(b: PtxBuilder, dst: str, a: str) -> None:
    b.emit("set.eq.f32.f32", dst, a, ZERO)


def _where(b: PtxBuilder, dst: str, cond: str, a: str, c: str) -> None:
    p = b.reg("pred")
    b.emit("setp.neu.f32", p, cond, ZERO)
    b.emit("selp.f32", dst, a, c, p)


_COMPOSITE: dict[str, tuple[Callable[..., None], int]] = {
    "relu": (_relu, 1),
    "square": (_square, 1),
    "rsqrt": (_rsqrt, 1),
    "exp": (_exp, 1),
    "exp_fast": (_exp_fast, 1),
    "sigmoid": (_sigmoid, 1),
    "silu": (_silu, 1),
    "tanh": (_tanh, 1),
    "erf": (_erf, 1),
    "gelu": (_gelu, 1),
    "gelu_tanh": (_gelu_tanh, 1),
    "log": (_log, 1),
    "sin": (_sincos(cos=False), 1),
    "cos": (_sincos(cos=True), 1),
    "pow": (_pow, 2),
    "where": (_where, 3),
    "bitwise_and": (_logical("and"), 2),
    "bitwise_or": (_logical("or"), 2),
    "bitwise_not": (_logical_not, 1),
    **{name: (_comparison(cmp), 2) for name, cmp in _COMPARE.items()},
}


def emit_elementwise(b: PtxBuilder, op: str, dst: str, args: list[str]) -> None:
    """Emit `dst = op(*args)` for f32 operands (registers or immediates)."""
    if op == "multiply" and ONE in args:  # x * 1.0 is exactly x (softmax's `one` Let)
        _check_arity(op, 2, args)
        b.emit("mov.f32", dst, args[1] if args[0] == ONE else args[0])
    elif op in _DIRECT:
        opcode, arity = _DIRECT[op]
        _check_arity(op, arity, args)
        b.emit(opcode, dst, *args)
    elif op in _COMPOSITE:
        fn, arity = _COMPOSITE[op]
        _check_arity(op, arity, args)
        fn(b, dst, *args)
    else:
        raise CompileError(f"unsupported elementwise op {op!r}")


def _check_arity(op: str, arity: int, args: list[str]) -> None:
    if len(args) != arity:
        raise CompileError(f"{op} takes {arity} argument(s), got {len(args)}")
