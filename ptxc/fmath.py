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
}


# ---- several instructions -----------------------------------------------------------------

def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


_LOG2E = _f32(math.log2(math.e))
_LOG2E_LO = _f32(math.log2(math.e) - _LOG2E)  # what rounding log2(e) to f32 dropped
_LN2 = _f32(math.log(2.0))
ZERO = f32_imm(0.0)
ONE = f32_imm(1.0)


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
    hi, neg_hi, lo, p2, corr = (b.reg("f32") for _ in range(5))
    normal = b.reg("pred")
    b.emit("mul.rn.f32", hi, x, f32_imm(_LOG2E))
    b.emit("neg.f32", neg_hi, hi)
    b.emit("fma.rn.f32", lo, x, f32_imm(_LOG2E), neg_hi)
    b.emit("fma.rn.f32", lo, x, f32_imm(_LOG2E_LO), lo)
    b.emit("ex2.approx.f32", p2, hi)
    b.emit("mul.rn.f32", corr, lo, f32_imm(_LN2))
    # 0, subnormal, inf and NaN results need (and, for inf/NaN inputs, survive) no correction.
    b.emit("testp.normal.f32", normal, p2)
    b.emit("fma.rn.f32", dst, p2, corr, p2, guard=normal)
    b.emit("mov.f32", dst, p2, guard="!" + normal)


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


_COMPOSITE: dict[str, tuple[Callable[..., None], int]] = {
    "relu": (_relu, 1),
    "square": (_square, 1),
    "rsqrt": (_rsqrt, 1),
    "exp": (_exp, 1),
    "exp_fast": (_exp_fast, 1),
    "sigmoid": (_sigmoid, 1),
}


def emit_elementwise(b: PtxBuilder, op: str, dst: str, args: list[str]) -> None:
    """Emit `dst = op(*args)` for f32 operands (registers or immediates)."""
    if op in _DIRECT:
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
