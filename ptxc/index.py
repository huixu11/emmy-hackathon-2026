"""Integer index arithmetic, predicates and memory addresses.

An index expression (Var / Literal / BinaryExpr / TernaryExpr / an integer CastExpr) compiles
to `Lin(reg, const)`: a register plus a compile-time constant. Keeping the constant apart lets
it fold through additions and constant multiplies and land in the address immediate, so the
unrolled loads `x[(a0 * 4) + 1]` and `x[(a0 * 4) + 2]` share one address register and differ
only in the offset: `[%rd5+4]`, `[%rd5+8]`.

Integer arithmetic is 32-bit signed, the `int` of the reference CUDA (buffers hold at most
2^26 elements, so byte offsets fit). Every instruction emitted here is pure and goes through
`derive`, which reuses the register of an identical earlier computation in scope and records
what is known about its result (facts.py): a range and a divisor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Expr, Literal, TernaryExpr, Var

from .errors import CompileError
from .facts import UNKNOWN, Fact, infer
from .ptx import PtxBuilder, address, is_register, register_class
from .scope import Scopes


@dataclass(frozen=True)
class Lin:
    """The integer value `reg + const`; `reg` is None for a compile-time constant."""

    reg: str | None
    const: int = 0


ZERO = Lin(None, 0)

_ARITH = {"+", "-", "*", "/", "//", "%"}  # emmy writes floor division both ways; operands are >= 0
_BITWISE = {"&": "and.b32", "|": "or.b32", "^": "xor.b32", "<<": "shl.b32", ">>": "shr.s32"}
_FOLD_BITWISE = {"&": int.__and__, "|": int.__or__, "^": int.__xor__,
                 "<<": int.__lshift__, ">>": int.__rshift__}
_COMPARE = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_MIRROR = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
_LOGIC = {"&&": "and.pred", "||": "or.pred"}
_INT_CASTS = {"int", "int32", "i32", "long", "int64", "i64"}
_TO_INT = {"b32": "mov.b32", "b64": "cvt.u32.u64", "f32": "cvt.rzi.s32.f32", "f64": "cvt.rzi.s32.f64"}


def _decide(cmp: str, a: Fact, b: Fact) -> bool | None:
    """The comparison's value when the ranges of its sides settle it, else None."""
    if None in (a.lo, a.hi, b.lo, b.hi):
        return None
    lo, hi = a.lo - b.hi, a.hi - b.lo  # the range of a - b
    if cmp == "lt":
        return True if hi < 0 else False if lo >= 0 else None
    if cmp == "le":
        return True if hi <= 0 else False if lo > 0 else None
    if cmp == "gt":
        return True if lo > 0 else False if hi <= 0 else None
    if cmp == "ge":
        return True if lo >= 0 else False if hi < 0 else None
    same = lo == hi == 0
    apart = hi < 0 or lo > 0
    if cmp == "eq":
        return True if same else False if apart else None
    return False if same else True if apart else None  # ne


def _c_div(a: int, b: int) -> int:
    """C integer division: truncates toward zero (Python's // floors)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


class IndexCompiler:
    def __init__(self, builder: PtxBuilder, scopes: Scopes) -> None:
        self.b = builder
        self.scopes = scopes
        self.facts: dict[str, Fact] = {}  # operand (register, %tid.x, shared symbol) -> fact
        self.depth: dict[str, int] = {}   # derived register -> depth of the scope caching it

    def derive(self, cls: str, opcode: str, *srcs) -> str:
        """The result register of a pure instruction, computed once: reused from an enclosing
        scope's cache, or emitted in the outermost scope its operands allow (scope.py)."""
        key = (opcode, *srcs)
        cached = self.scopes.cached(key)
        if cached is not None:
            return cached
        home, at = self.scopes.placement(max((self._depth_of(s) for s in srcs), default=0))
        dst = self.b.reg(cls)
        self.b.emit(opcode, dst, *srcs, at=at)
        self.facts[dst] = infer(opcode, srcs, self._fact_of)
        self.depth[dst] = home.depth
        home.values[key] = dst
        return dst

    def _depth_of(self, src) -> int:
        """The scope depth an operand is defined at: 0 for constants, special registers and
        symbols; for a register not made by derive (a loop counter), the current depth."""
        if isinstance(src, int) or not src.startswith("%") or "." in src:
            return 0
        return self.depth.get(src, self.scopes.current.depth)

    # ---- facts -----------------------------------------------------------------------------

    def assume(self, operand: str, fact: Fact) -> None:
        """Record something the launch or the IR guarantees about an operand."""
        self.facts[operand] = self.facts.get(operand, UNKNOWN).meet(fact)

    def fact(self, v: Lin) -> Fact:
        if v.reg is None:
            return Fact.const(v.const)
        return self.facts.get(v.reg, UNKNOWN).shifted(v.const)

    def _fact_of(self, src) -> Fact:
        return Fact.const(src) if isinstance(src, int) else self.facts.get(src, UNKNOWN)

    # ---- integer values -------------------------------------------------------------------

    def value(self, e: Expr) -> Lin:
        if isinstance(e, Literal):
            if not float(e.value).is_integer():
                raise CompileError(f"non-integer literal {e.value!r} in an index expression")
            return Lin(None, int(e.value))
        if isinstance(e, Var):
            v = self.scopes.lookup(e.name)
            if not isinstance(v, Lin):
                raise CompileError(f"{e.name!r} is used as an index but is not an integer")
            return v
        if isinstance(e, BinaryExpr) and e.op in _ARITH:
            return self.arith(e.op, self.value(e.left), self.value(e.right))
        if isinstance(e, BinaryExpr) and e.op in _BITWISE:
            return self.bitwise(e.op, self.value(e.left), self.value(e.right))
        if isinstance(e, TernaryExpr):
            p = self.pred(e.cond)
            if isinstance(p, bool):
                return self.value(e.if_true if p else e.if_false)
            t, f = self.value(e.if_true), self.value(e.if_false)
            return Lin(self.derive("b32", "selp.b32", self.operand(t), self.operand(f), p))
        if isinstance(e, CastExpr) and e.dtype in _INT_CASTS:
            return self._as_index(e.expr)
        raise CompileError(f"unsupported index expression: {e!r}")

    def _as_index(self, e: Expr) -> Lin:
        """`(int)e`: an index is an int already; a data value (the i64 or the bool-as-1.0 an
        embedding or gather computes its row from) is converted, as C does, into a register of
        its own, never cached, since the value's register can be assigned again."""
        v = self.scopes.lookup(e.name) if isinstance(e, Var) else None
        if not isinstance(v, str):
            return self.value(e)
        if not is_register(v) or register_class(v) not in _TO_INT:
            raise CompileError(f"cannot use {e.name!r} as an index")
        dst = self.b.reg("b32")
        self.b.emit(_TO_INT[register_class(v)], dst, v)
        return Lin(dst)

    def arith(self, op: str, a: Lin, b: Lin) -> Lin:
        if op == "+":
            return self.add(a, b)
        if op == "-":
            return self.add(a, self.neg(b))
        if op == "*":
            return self.mul(a, b)
        if a.reg is None and b.reg is None:
            q = _c_div(a.const, b.const)
            return Lin(None, q if op == "/" else a.const - q * b.const)
        if b == Lin(None, 1):
            return ZERO if op == "%" else a
        if b.reg is None and b.const > 0 and self.fact(a).nonneg:
            return self._by_constant(op, a, b.const)
        opcode = "rem.s32" if op == "%" else "div.s32"
        return Lin(self.derive("b32", opcode, self.reg(a), self.operand(b)))

    def _by_constant(self, op: str, a: Lin, k: int) -> Lin:
        """a / k or a % k for a constant k > 0 and a >= 0, without a divide: ptxas turns
        div/rem even by a power of two into a reciprocal-based division sequence. A power of
        two is a shift or a mask; any other k a multiply-high by a magic reciprocal
        (Granlund & Montgomery): with l = ceil(log2 k) and m = ceil(2^(31+l) / k),
        a / k = mulhi(a, m) >> (l - 1) exactly for every 0 <= a < 2^31."""
        x = self.reg(a)
        fact = self.fact(a)
        if k & (k - 1) == 0:
            if op == "%":
                r = self.derive("b32", "and.b32", x, k - 1)
            else:
                r = self.derive("b32", "shr.u32", x, k.bit_length() - 1)
        else:
            l = (k - 1).bit_length()
            q = self.derive("b32", "mul.hi.u32", x, -(-(1 << (31 + l)) // k))
            if l > 1:
                q = self.derive("b32", "shr.u32", q, l - 1)
            if op != "%":
                self.assume(q, Fact(fact.lo // k, None if fact.hi is None else fact.hi // k))
                return Lin(q)
            r = self.derive("b32", "mad.lo.s32", q, -k, x)  # a - (a / k) * k
        if op == "%":
            self.assume(r, Fact(0, k - 1 if fact.hi is None else min(fact.hi, k - 1), math.gcd(fact.div, k)))
        return Lin(r)

    def bitwise(self, op: str, a: Lin, b: Lin) -> Lin:
        if a.reg is None and b.reg is None:
            return Lin(None, _FOLD_BITWISE[op](a.const, b.const))
        if a.reg is None and op in "&|^":  # commutative: put the register first
            a, b = b, a
        return Lin(self.derive("b32", _BITWISE[op], self.reg(a), self.operand(b)))

    def add(self, a: Lin, b: Lin) -> Lin:
        if a.reg is None or b.reg is None:
            return Lin(a.reg or b.reg, a.const + b.const)
        lo, hi = sorted((a.reg, b.reg))
        return Lin(self.derive("b32", "add.s32", lo, hi), a.const + b.const)

    def neg(self, a: Lin) -> Lin:
        return Lin(self.derive("b32", "neg.s32", a.reg) if a.reg else None, -a.const)

    def mul(self, a: Lin, b: Lin) -> Lin:
        if a.reg is not None and b.reg is not None:
            lo, hi = sorted((self.reg(a), self.reg(b)))
            return Lin(self.derive("b32", "mul.lo.s32", lo, hi))
        if a.reg is None:
            a, b = b, a
        k = b.const  # (reg + c) * k = reg * k + c * k
        if a.reg is None or k == 0:
            return Lin(None, a.const * k)
        reg = a.reg if k == 1 else self.derive("b32", "mul.lo.s32", a.reg, k)
        return Lin(reg, a.const * k)

    def reg(self, v: Lin) -> str:
        """`v` materialized in a register."""
        if v.reg is None:
            return self.derive("b32", "mov.s32", v.const)
        if v.const == 0:
            return v.reg
        return self.derive("b32", "add.s32", v.reg, v.const)

    def operand(self, v: Lin) -> str:
        """`v` as an instruction source: an immediate when it is a constant."""
        return str(v.const) if v.reg is None else self.reg(v)

    # ---- predicates ------------------------------------------------------------------------

    def pred(self, e: Expr) -> bool | str:
        """A predicate register, or a Python bool when the condition is known statically."""
        if isinstance(e, BinaryExpr) and e.op in _COMPARE:
            return self.compare(_COMPARE[e.op], self.value(e.left), self.value(e.right))
        if isinstance(e, BinaryExpr) and e.op in _LOGIC:
            a, b = self.pred(e.left), self.pred(e.right)
            absorbing = e.op == "||"  # True absorbs ||, False absorbs &&
            if absorbing in (a, b):
                return absorbing
            if isinstance(a, bool):
                return b
            if isinstance(b, bool):
                return a
            return self.derive("pred", _LOGIC[e.op], a, b)
        return self.compare("ne", self.value(e), ZERO)  # C truthiness of an integer

    def compare(self, cmp: str, a: Lin, b: Lin) -> bool | str:
        known = _decide(cmp, self.fact(a), self.fact(b))
        if known is not None:
            return known
        if a.reg is None:
            a, b, cmp = b, a, _MIRROR[cmp]
        if b.reg is None:  # reg + c  <cmp>  k   <=>   reg  <cmp>  k - c
            return self.derive("pred", f"setp.{cmp}.s32", a.reg, b.const - a.const)
        return self.derive("pred", f"setp.{cmp}.s32", self.reg(a), self.reg(b))

    # ---- addresses -------------------------------------------------------------------------

    def flat(self, index: tuple[Expr, ...], shape: tuple[int, ...]) -> Lin:
        """The row-major element offset of a coordinate (a 1-tuple is already flat)."""
        if len(index) <= 1:
            return self.value(index[0]) if index else ZERO
        if len(index) != len(shape):
            raise CompileError(f"{len(index)}-d index into a buffer of shape {shape}")
        offset = ZERO
        for e, extent in zip(index, shape):
            offset = self.add(self.mul(offset, Lin(None, extent)), self.value(e))
        return offset

    def global_address(self, base: str, flat: Lin, esize: int) -> str:
        if flat.reg is None:
            return address(base, flat.const * esize)
        offset = self.derive("b64", "mul.wide.s32", flat.reg, esize)
        ptr = self.derive("b64", "add.s64", base, offset)
        return address(ptr, flat.const * esize)

    def shared_address(self, symbol: str, flat: Lin, esize: int) -> str:
        if flat.reg is None:
            return address(symbol, flat.const * esize)
        base = self.derive("b32", "mov.u32", symbol)
        ptr = self.derive("b32", "mad.lo.s32", flat.reg, esize, base)
        return address(ptr, flat.const * esize)
