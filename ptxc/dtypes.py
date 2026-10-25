"""Data types: how a value of each IR dtype is held in registers, read, written and converted.

| dtype | in memory | held in | arithmetic |
|---|---|---|---|
| f32 | 4 bytes | an f32 register | fmath.py |
| f16, bf16 | 2 bytes | an f32 register, at 16-bit precision | fmath.py in f32, then one rounding to the dtype |
| bool | 1 byte | an f32 register, 1.0 or 0.0 | the comparison and logic ops of fmath.py |
| f64 | 8 bytes | an f64 register | add, mul, div, sqrt, ... in f64; other functions through f32 |
| i32, i64 | 4, 8 bytes | a b32 / b64 register | integer instructions, floor division and modulo as in torch |

A 16-bit float always holds a value its dtype can represent, so it converts to f32 for free and
an operation on it is an f32 operation plus one rounding. For add, subtract and multiply that
is exactly the correctly rounded 16-bit result (the f32 sum or product of two 16-bit values is
exact, or too far from a rounding boundary to matter), which is what torch computes.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .errors import CompileError
from .fmath import emit_elementwise
from .ptx import PtxBuilder, f32_imm, f64_imm, is_register


@dataclass(frozen=True)
class DType:
    name: str
    nbytes: int
    reg: str        # register class a value lives in
    mem: str        # PTX type of one element's load or store
    mem_reg: str    # register class a load writes, before conversion

    @property
    def is_int(self) -> bool:
        return self.reg in ("b32", "b64")

    @property
    def bits(self) -> int:
        return 64 if self.reg in ("b64", "f64") else 32

    @property
    def is_half(self) -> bool:
        return self.name in ("f16", "bf16")


F32 = DType("f32", 4, "f32", "f32", "f32")
F16 = DType("f16", 2, "f32", "b16", "b16")
BF16 = DType("bf16", 2, "f32", "b16", "b16")
F64 = DType("f64", 8, "f64", "f64", "f64")
I32 = DType("i32", 4, "b32", "s32", "b32")
I64 = DType("i64", 8, "b64", "s64", "b64")
BOOL = DType("bool", 1, "f32", "u8", "b32")

_BY_NAME = {d.name: d for d in (F32, F16, BF16, F64, I32, I64, BOOL)}
_ALIASES = {"float": F32, "half": F16, "__half": F16, "__nv_bfloat16": BF16, "double": F64,
            "int": I32, "int32": I32, "int64": I64, "long": I64}


def dtype(d) -> DType:
    """The DType of an IR dtype (a DataType record or its name); None means f32."""
    if d is None:
        return F32
    name = getattr(d, "name", d)
    found = _BY_NAME.get(name) or _ALIASES.get(name)
    if found is None:
        raise CompileError(f"dtype {name} is not supported")
    return found


def common(dtypes: list[DType]) -> DType:
    """The dtype operands of a comparison meet in, as torch promotes them: any float makes it a
    float (f64 over the f32-held ones), else the widest integer; bools alone compare as f32."""
    if F64 in dtypes:
        return F64
    if any(d.reg == "f32" and d != BOOL for d in dtypes):
        return F32
    for d in (I64, I32):
        if d in dtypes:
            return d
    return F32


# ---- values ----------------------------------------------------------------------------------

def immediate(value: float, d: DType) -> str:
    """`value` as an operand of dtype d, rounded to it."""
    if d.reg == "f32":
        return f32_imm(_round_host(float(value), d))
    if d.reg == "f64":
        return f64_imm(float(value))
    top = 2 ** (d.bits - 1)
    if math.isinf(value):
        return str(top - 1 if value > 0 else -top)
    return str(max(-top, min(top - 1, int(value))))  # -1e30, a max's identity, is the least int


def _round_host(x: float, d: DType) -> float:
    if d.name == "f16":
        try:
            return struct.unpack("<e", struct.pack("<e", x))[0]
        except OverflowError:
            return math.copysign(math.inf, x)
    if d.name == "bf16":
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        if math.isnan(x):
            return x
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000  # round to nearest even
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    if d.name == "bool":
        return 1.0 if x else 0.0
    return x


def round_to(b: PtxBuilder, value: str, d: DType) -> str:
    """An f32 value rounded to the 16-bit dtype d (identity for other dtypes)."""
    if not d.is_half:
        return value
    if not is_register(value):
        return f32_imm(_round_host(_imm(value), d))
    out = b.reg("f32")
    if d.name == "f16":
        h = b.reg("b16")
        b.emit("cvt.rn.f16.f32", h, value)
        b.emit("cvt.f32.f16", out, h)
    else:  # bf16 sits in the high half of an f32: round the low half away
        pair = b.reg("b32")
        b.emit("cvt.rn.bf16x2.f32", pair, value, value)
        b.emit("and.b32", pair, pair, "0xFFFF0000")
        b.emit("mov.b32", out, pair)
    return out


def _imm(operand: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(operand[2:], 16)))[0]


def convert(b: PtxBuilder, value: str, src: DType, dst: DType) -> str:
    """`value`, held as dtype src, as dtype dst (torch's conversions: floats to ints truncate)."""
    if src == dst:
        return value
    if dst == BOOL:  # nonzero is true, NaN included
        out = b.reg("f32")
        if src.is_int:
            b.emit(f"set.ne.f32.{_ptx(src)}", out, value, "0")
        else:
            zero = f32_imm(0.0) if src.reg == "f32" else f64_imm(0.0)
            b.emit(f"set.neu.f32.{_ptx(src)}", out, value, zero)
        return out
    if src.reg == dst.reg:  # f32-held dtypes, or i32 / i64 to itself
        return round_to(b, value, dst)
    out = b.reg(dst.reg)
    if dst.reg == "f32":  # from f64 or an integer
        b.emit(f"cvt.rn.f32.{_ptx(src)}", out, value)
        return round_to(b, out, dst)
    if dst.reg == "f64":
        b.emit(f"cvt.{'rn.' if src.is_int else ''}f64.{_ptx(src)}", out, value)
        return out
    if src.is_int:  # i32 <-> i64
        b.emit(f"cvt.{_ptx(dst)}.{_ptx(src)}", out, value)
    else:  # float -> int: truncate toward zero
        b.emit(f"cvt.rzi.{_ptx(dst)}.{_ptx(src)}", out, value)
    return out


def _ptx(d: DType) -> str:
    """The PTX type a value of d is held as in registers."""
    return {"f32": "f32", "f64": "f64", "b32": "s32", "b64": "s64"}[d.reg]


def _in_register(b: PtxBuilder, value: str, d: DType) -> str:
    if is_register(value):
        return value
    out = b.reg(d.reg)
    b.emit(f"mov.{_ptx(d) if not d.is_int else 'b' + str(d.bits)}", out, value)
    return out


def move(b: PtxBuilder, dst: str, value: str, d: DType) -> None:
    b.emit(f"mov.{'b' + str(d.bits) if d.is_int else _ptx(d)}", dst, value)


# ---- memory ----------------------------------------------------------------------------------

def from_memory(b: PtxBuilder, regs: list[str], d: DType) -> list[str]:
    """Values of dtype d from the registers a load of d wrote."""
    if d.mem_reg == d.reg:
        return regs
    out = []
    for r in regs:
        v = b.reg(d.reg)
        if d.name == "f16":
            b.emit("cvt.f32.f16", v, r)
        elif d.name == "bf16":  # the high half of an f32
            zero = b.reg("b16")
            b.emit("mov.b16", zero, "0")
            b.emit("mov.b32", v, "{" + f"{zero}, {r}" + "}")
        else:  # bool: one byte, 0 or 1
            b.emit("cvt.rn.f32.u32", v, r)
        out.append(v)
    return out


def to_memory(b: PtxBuilder, value: str, d: DType) -> str:
    """The register a store of dtype d writes for `value`, held as d."""
    if d.mem_reg == d.reg:
        return _in_register(b, value, d)
    out = b.reg(d.mem_reg)
    if d.name == "f16":
        b.emit("cvt.rn.f16.f32", out, value)
    elif d.name == "bf16":
        b.emit("cvt.rn.bf16.f32", out, value)
    else:  # bool
        b.emit("cvt.rzi.u32.f32", out, value)
    return out


def store_type(d: DType) -> DType:
    """The dtype a value goes to to_memory in for a buffer of d: a 16-bit float stays f32 and
    rounds once, in the store's own conversion."""
    return F32 if d.is_half else d


def atomic_add(d: DType) -> str:
    """The red opcode suffix of an atomic add of dtype d."""
    suffix = {"f32": "add.f32", "f64": "add.f64", "i32": "add.s32", "i64": "add.u64", "f16": "add.noftz.f16"}
    if d.name not in suffix:
        raise CompileError(f"atomic add of {d.name} is not supported on sm_89")
    return suffix[d.name]


# ---- arithmetic ------------------------------------------------------------------------------

_COMPARE_FLOAT = {"equal": "eq", "not_equal": "neu", "greater": "gt", "less": "lt",
                  "greater_equal": "ge", "less_equal": "le"}
_COMPARE_INT = {"equal": "eq", "not_equal": "ne", "greater": "gt", "less": "lt",
                "greater_equal": "ge", "less_equal": "le"}
COMPARISONS = set(_COMPARE_FLOAT)

_F64_DIRECT = {"add": "add.rn.f64", "subtract": "sub.rn.f64", "multiply": "mul.rn.f64",
               "divide": "div.rn.f64", "maximum": "max.f64", "minimum": "min.f64",
               "negative": "neg.f64", "abs": "abs.f64", "sqrt": "sqrt.rn.f64",
               "reciprocal": "rcp.rn.f64", "copy": "mov.f64", "pad": "mov.f64"}

_INT_DIRECT = {"add": "add.s", "subtract": "sub.s", "multiply": "mul.lo.s", "maximum": "max.s",
               "minimum": "min.s", "negative": "neg.s", "abs": "abs.s", "copy": "mov.b",
               "pad": "mov.b", "bitwise_and": "and.b", "bitwise_or": "or.b",
               "bitwise_xor": "xor.b", "bitwise_not": "not.b"}


def compare(b: PtxBuilder, op: str, dst: str, a: str, c: str, d: DType) -> None:
    """dst (a bool, 1.0 or 0.0) = a <op> c, both of dtype d."""
    if d.reg == "f32":
        emit_elementwise(b, op, dst, [a, c])
    elif d.reg == "f64":
        b.emit(f"set.{_COMPARE_FLOAT[op]}.f32.f64", dst, a, c)
    else:
        b.emit(f"set.{_COMPARE_INT[op]}.f32.{_ptx(d)}", dst, a, c)


def select(b: PtxBuilder, dst: str, cond: str, a: str, c: str, d: DType) -> None:
    """dst = cond ? a : c, where cond is a bool and a, c have dtype d."""
    p = b.reg("pred")
    b.emit("setp.neu.f32", p, cond, f32_imm(0.0))
    b.emit(f"selp.{'b' + str(d.bits) if d.is_int else _ptx(d)}", dst, a, c, p)


def emit(b: PtxBuilder, op: str, dst: str, args: list[str], d: DType) -> None:
    """dst = op(*args) in dtype d, every argument already of dtype d."""
    if d.reg == "f32":
        emit_elementwise(b, op, dst, args)
        return
    if d.reg == "f64":
        _emit_f64(b, op, dst, args)
    else:
        _emit_int(b, op, dst, args, d)


def _emit_f64(b: PtxBuilder, op: str, dst: str, args: list[str]) -> None:
    if op in _F64_DIRECT:
        b.emit(_F64_DIRECT[op], dst, *args)
    elif op == "square":
        b.emit("mul.rn.f64", dst, args[0], args[0])
    elif op == "relu":
        b.emit("max.f64", dst, args[0], f64_imm(0.0))
    elif op == "rsqrt":
        root = b.reg("f64")
        b.emit("sqrt.rn.f64", root, args[0])
        b.emit("rcp.rn.f64", dst, root)
    else:  # a function fmath.py has in f32 only: evaluated in f32, as Emmy's CUDA does
        narrow = [convert(b, a, F64, F32) for a in args]
        out = b.reg("f32")
        emit_elementwise(b, op, out, narrow)
        b.emit("cvt.f64.f32", dst, out)


def _emit_int(b: PtxBuilder, op: str, dst: str, args: list[str], d: DType) -> None:
    w = str(d.bits)
    if op in _INT_DIRECT:
        b.emit(_INT_DIRECT[op] + w, dst, *args)
    elif op == "square":
        b.emit(f"mul.lo.s{w}", dst, args[0], args[0])
    elif op == "relu":
        b.emit(f"max.s{w}", dst, args[0], "0")
    elif op in ("floor_divide", "remainder") and _power_of_two(args[1]):
        # By 2^k, rounding down is an arithmetic shift and the remainder its low k bits.
        k = int(args[1])
        if op == "floor_divide":
            b.emit(f"shr.s{w}", dst, args[0], k.bit_length() - 1)
        else:
            b.emit(f"and.b{w}", dst, args[0], k - 1)
    elif op in ("floor_divide", "remainder"):
        # torch: the quotient rounds down and the remainder takes the divisor's sign; PTX's
        # div and rem truncate toward zero, so fix both when the remainder and divisor differ in sign.
        a, c = args
        rem, sign = b.reg(d.reg), b.reg(d.reg)
        nonzero, differ, fix = b.reg("pred"), b.reg("pred"), b.reg("pred")
        b.emit(f"rem.s{w}", rem, a, c)
        b.emit(f"setp.ne.s{w}", nonzero, rem, "0")
        b.emit(f"xor.b{w}", sign, rem, c)
        b.emit(f"setp.lt.s{w}", differ, sign, "0")
        b.emit("and.pred", fix, nonzero, differ)
        if op == "floor_divide":
            b.emit(f"div.s{w}", dst, a, c)
            b.emit(f"sub.s{w}", dst, dst, "1", guard=fix)
        else:
            b.emit(f"mov.b{w}", dst, rem)
            b.emit(f"add.s{w}", dst, dst, c, guard=fix)
    else:
        raise CompileError(f"{op} on {d.name} values is not supported")


def _power_of_two(operand: str) -> bool:
    """Whether an integer operand is an immediate 2^k, k >= 0."""
    return operand.isdigit() and int(operand) > 0 and int(operand) & (int(operand) - 1) == 0
