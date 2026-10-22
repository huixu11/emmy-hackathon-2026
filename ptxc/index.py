"""Integer index arithmetic, predicates and memory addresses.

An index expression (Var / Literal / BinaryExpr / TernaryExpr) compiles to `Lin(reg, const)`:
a register plus a compile-time constant. Keeping the constant apart lets it fold through
additions and constant multiplies and land in the address immediate, so the unrolled loads
`x[(a0 * 4) + 1]` and `x[(a0 * 4) + 2]` share one address register and differ only in the
offset: `[%rd5+4]`, `[%rd5+8]`.

Integer arithmetic is 32-bit signed, the `int` of the reference CUDA (buffers hold at most
2^26 elements, so byte offsets fit). Every instruction emitted here is pure and goes through
`derive`, which reuses the register of an identical earlier computation in scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var

from .errors import CompileError
from .ptx import PtxBuilder, address
from .scope import Scopes


@dataclass(frozen=True)
class Lin:
    """The integer value `reg + const`; `reg` is None for a compile-time constant."""

    reg: str | None
    const: int = 0


ZERO = Lin(None, 0)

_ARITH = {"+", "-", "*", "/", "%"}
_COMPARE = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_MIRROR = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
_LOGIC = {"&&": "and.pred", "||": "or.pred"}


def _c_div(a: int, b: int) -> int:
    """C integer division: truncates toward zero (Python's // floors)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


class IndexCompiler:
    def __init__(self, builder: PtxBuilder, scopes: Scopes) -> None:
        self.b = builder
        self.scopes = scopes

    def derive(self, cls: str, opcode: str, *srcs) -> str:
        """The result register of a pure instruction, emitted once per scope."""
        def emit() -> str:
            dst = self.b.reg(cls)
            self.b.emit(opcode, dst, *srcs)
            return dst
        return self.scopes.memo((opcode, *srcs), emit)

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
        if isinstance(e, TernaryExpr):
            p = self.pred(e.cond)
            if isinstance(p, bool):
                return self.value(e.if_true if p else e.if_false)
            t, f = self.value(e.if_true), self.value(e.if_false)
            return Lin(self.derive("b32", "selp.b32", self.operand(t), self.operand(f), p))
        raise CompileError(f"unsupported index expression: {e!r}")

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
            return a if op == "/" else ZERO
        opcode = "div.s32" if op == "/" else "rem.s32"
        return Lin(self.derive("b32", opcode, self.reg(a), self.operand(b)))

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
        if a.reg is None and b.reg is None:
            x, y = a.const, b.const
            return {"lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y, "eq": x == y, "ne": x != y}[cmp]
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
