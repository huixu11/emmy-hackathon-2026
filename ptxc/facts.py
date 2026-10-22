"""What is known about an integer register: its range and a divisor.

Every index register is defined once, by a pure instruction (see IndexCompiler.derive), so a
fact derived when it is defined holds everywhere it is used. Facts start from the launch
(%tid.x < block size, a Tile coordinate < its extent) and flow through the arithmetic:

    a0_co in [0, 127]   ->   a0_co * 4 in [0, 508], a multiple of 4

The lowering uses them to count a loop's trips (to unroll it, or to drop the loop when it
runs at most once), to fold comparisons, and to prove an address aligned for a vector load.
Unknown stays unknown: every rule below is exact or gives up.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    lo: int | None = None  # inclusive bounds; None when unknown
    hi: int | None = None
    div: int = 1           # the value is a multiple of div (0: the value is 0)

    @staticmethod
    def const(c: int) -> Fact:
        return Fact(c, c, abs(c))

    @property
    def nonneg(self) -> bool:
        return self.lo is not None and self.lo >= 0

    def shifted(self, c: int) -> Fact:
        """The fact for value + c."""
        return Fact(_add(self.lo, c), _add(self.hi, c), math.gcd(self.div, c))

    def meet(self, other: Fact) -> Fact:
        """Both facts hold: the tighter bounds, and a divisor of both kinds."""
        lo = other.lo if self.lo is None else self.lo if other.lo is None else max(self.lo, other.lo)
        hi = other.hi if self.hi is None else self.hi if other.hi is None else min(self.hi, other.hi)
        return Fact(lo, hi, math.lcm(self.div, other.div))


UNKNOWN = Fact()


def _add(a: int | None, b: int | None) -> int | None:
    return None if a is None or b is None else a + b


def _sum(a: Fact, b: Fact) -> Fact:
    return Fact(_add(a.lo, b.lo), _add(a.hi, b.hi), math.gcd(a.div, b.div))


def _product(a: Fact, b: Fact) -> Fact:
    div = a.div * b.div
    if a.nonneg and b.nonneg and a.hi is not None and b.hi is not None:
        return Fact(a.lo * b.lo, a.hi * b.hi, div)
    if b.lo is not None and b.lo == b.hi:  # times a constant of either sign
        k = b.lo
        ends = [None if v is None else v * k for v in (a.lo, a.hi)]
        if k < 0:
            ends.reverse()
        return Fact(ends[0], ends[1], div)
    return Fact(None, None, div)


def _quotient(a: Fact, k: int) -> Fact:
    if not a.nonneg or k <= 0:
        return UNKNOWN
    return Fact(a.lo // k, None if a.hi is None else a.hi // k, a.div // k if a.div % k == 0 else 1)


def _remainder(a: Fact, k: int) -> Fact:
    if not a.nonneg or k <= 0:
        return UNKNOWN
    if a.hi is not None and a.hi < k:
        return a
    return Fact(0, k - 1, math.gcd(a.div, k))


def infer(opcode: str, srcs: tuple, fact_of: Callable[[object], Fact]) -> Fact:
    """The fact for the result of `opcode srcs` (the integer instructions the compiler emits)."""
    f = [fact_of(s) for s in srcs]
    k = srcs[-1] if srcs and isinstance(srcs[-1], int) else None  # a constant last operand
    op = opcode.split(".")[0]
    if op == "mov":  # a constant, a special register (%tid.x) or a shared symbol's address
        return Fact.const(srcs[0]) if isinstance(srcs[0], int) else f[0]
    if op == "add":
        return _sum(f[0], f[1])
    if op == "neg":
        return Fact(None if f[0].hi is None else -f[0].hi, None if f[0].lo is None else -f[0].lo, f[0].div)
    if op == "mul":  # mul.lo.s32 and mul.wide.s32
        return _product(f[0], f[1])
    if op == "mad":
        return _sum(_product(f[0], f[1]), f[2])
    if op == "shl" and k is not None:
        return _product(f[0], Fact.const(1 << k))
    if op == "div" and k is not None:
        return _quotient(f[0], k)
    if op in ("shr",) and k is not None:
        return _quotient(f[0], 1 << k)
    if op == "rem" and k is not None:
        return _remainder(f[0], k)
    if op == "and" and k is not None and k >= 0 and f[0].nonneg:
        return Fact(0, k if f[0].hi is None else min(f[0].hi, k), 1)
    if op == "min" and k is not None:
        lo = None if f[0].lo is None else min(f[0].lo, k)
        return Fact(lo, k if f[0].hi is None else min(f[0].hi, k), 1)
    if op == "selp":
        a, b = f[0], f[1]
        lo = None if a.lo is None or b.lo is None else min(a.lo, b.lo)
        hi = None if a.hi is None or b.hi is None else max(a.hi, b.hi)
        return Fact(lo, hi, math.gcd(a.div, b.div))
    return UNKNOWN
