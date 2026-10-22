"""Merge neighbouring scalar loads into vector loads: a pass over the finished instruction stream.

Unrolled loops leave runs of loads through one base register at consecutive offsets, e.g.
m01's inner product reads A one column per trip:

    ld.shared.f32   %f5, [%r62];        (trip 0)
    ...
    ld.shared.f32   %f9, [%r62+4];      (trip 1)  ... [%r62+8], [%r62+12]

One `ld.shared.v4.f32 {%f5, %f9, ...}, [%r62]` moves the same 16 bytes in one instruction. That
needs the address aligned to the vector's size, which the base register's known divisor
(facts.py) and the offset prove. The merged load takes the place of the earliest one, so a
later load moves up. That is safe when nothing in between writes that memory (a store, an
async copy, a barrier, a branch or label all end the region considered) or touches its
destination register.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .ptx import Instr, address, vector

_SCALAR_LOADS = ("ld.shared.f32", "ld.global.nc.f32")
_ADDR = re.compile(r"\[([%\w$]+)(?:\+(-?\d+))?\]")
_REGION_ENDS = ("st.", "red.", "atom.", "cp.async", "bar.", "membar", "fence", "bra", "ret")
_F32_BYTES = 4


def vectorize_loads(lines: list, alignment: Callable[[str], int]) -> list:
    """`lines` with runs of scalar f32 loads merged into .v2/.v4 loads where provably legal.
    `alignment(base)` is the largest power of two the base operand is known to be a multiple of."""
    lines = list(lines)
    pending: dict[tuple[str, str], list[tuple[int, int, str]]] = {}  # (opcode, base) -> loads

    def close_region() -> None:
        for (opcode, base), loads in pending.items():
            _merge(lines, opcode, base, loads, alignment(base))
        pending.clear()

    for pos, item in enumerate(lines):
        if not isinstance(item, Instr):
            if not item.lstrip().startswith("//"):  # a label
                close_region()
            continue
        match = _ADDR.fullmatch(item.operands[1]) if item.opcode in _SCALAR_LOADS and not item.guard else None
        if match:
            pending.setdefault((item.opcode, match[1]), []).append((pos, int(match[2] or 0), item.operands[0]))
        elif item.opcode.startswith(_REGION_ENDS) or (item.operands and any(base == item.operands[0] for _, base in pending)):
            close_region()
    close_region()
    return [item for item in lines if item is not None]


def _merge(lines: list, opcode: str, base: str, loads: list[tuple[int, int, str]], align: int) -> None:
    by_offset: dict[int, tuple[int, str]] = {}
    for pos, offset, dst in loads:
        by_offset.setdefault(offset, (pos, dst))  # a repeated address keeps its first load
    offsets = sorted(by_offset)
    i = 0
    while i < len(offsets):
        for width in (4, 2):
            run = [offsets[i] + _F32_BYTES * k for k in range(width)]
            nbytes = _F32_BYTES * width
            if (all(o in by_offset for o in run) and run[0] % nbytes == 0 and align % nbytes == 0
                    and _movable(lines, [by_offset[o] for o in run])):
                members = [by_offset[o] for o in run]
                first = min(pos for pos, _ in members)
                lines[first] = Instr(opcode.replace(".f32", f".v{width}.f32"),
                                     [vector([dst for _, dst in members]), address(base, run[0])])
                for pos, _ in members:
                    if pos != first:
                        lines[pos] = None
                i += width
                break
        else:
            i += 1


def _movable(lines: list, members: list[tuple[int, str]]) -> bool:
    """Whether every load of the run can move up to the earliest one: no instruction between
    reads or writes its destination register."""
    first = min(pos for pos, _ in members)
    for pos, dst in members:
        mention = re.compile(re.escape(dst) + r"(?!\d)")
        for between in lines[first + 1:pos]:
            if isinstance(between, Instr) and (
                    any(mention.search(op) for op in between.operands) or (between.guard and mention.search(between.guard))):
                return False
    return True
