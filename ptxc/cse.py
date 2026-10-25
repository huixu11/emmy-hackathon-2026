"""Drop recomputed divisions, reciprocals and square roots, and f64 round trips of f32 values:
a pass over the finished instruction stream.

Emmy repeats pure scalar work per output fragment, e.g. attention's epilogue takes the
reciprocal of each row's sum once per 16x8 tile it stores:

    rcp.rn.f32  %f900, %f17;    ... stores of tile 0 ...
    rcp.rn.f32  %f940, %f17;    ... stores of tile 1 ...

ptxas merges repeated arithmetic itself, but not rcp.rn, div.rn and sqrt.rn, whose slow path
it calls as a subroutine; so here the second becomes uses of %f900. That is safe when both
instructions are in one basic block (no label or branch between them), nothing between them
writes a source of the first or reads the second's result, and each result register is
written by no other instruction (so %f900 still holds the value wherever %f940 was read).
Merging the cheap ops here as well gives the same SASS, and moved one small attention
kernel's schedule from 4.8 to 6.8 us, so they are left to ptxas.

Under the same conditions, narrowing an f64 that widened an f32 (`cvt.f64.f32 %fd4, %f9` then
`cvt.rn.f32.f64 %f12, %fd4`) gives back %f9 exactly, so %f12 becomes %f9. Emmy's f64 kernels
do that for every element: `exp` is taken in f32 and widened to the f64 value the IR names,
which an f32 accumulator then narrows again; ptxas keeps both conversions.
"""

from __future__ import annotations

import re
from collections import Counter

from .ptx import Instr

_MERGED = {"div", "rcp", "sqrt"}
_NO_DESTINATION = ("st.", "red.", "bar.", "bra", "ret", "cp.async", "membar", "fence")
_REG = re.compile(r"%[a-z]+\d+")


def eliminate_common_subexpressions(lines: list) -> list:
    """`lines` without the recomputations described above."""
    writes = Counter(r for line in lines if isinstance(line, Instr) for r in _written(line))
    rename: dict[str, str] = {}
    keep = []
    seen: dict[tuple, tuple[str, int]] = {}  # (opcode, sources) -> (result, position in keep)
    widened: dict[str, tuple[str, int]] = {}  # f64 register -> (the f32 it widened, position)
    for line in lines:
        if not isinstance(line, Instr):
            if line.endswith(":"):  # a label starts a new basic block
                seen.clear()
                widened.clear()
            keep.append(line)
            continue
        if rename:
            line = _renamed(line, rename)
        key = (line.opcode, *line.operands[1:])
        dst = line.operands[0] if line.operands else None
        earlier = seen.get(key) if _mergeable(line, writes) else None
        if line.opcode == "cvt.rn.f32.f64" and _single(line, writes):
            earlier = widened.get(line.operands[1])
        if earlier is not None and not _reads_since(keep, earlier[1], dst):
            rename[dst] = earlier[0]
            continue
        written = _written(line)
        if written:
            seen = {k: v for k, v in seen.items()
                    if v[0] not in written and not any(r in written for r in _sources(k))}
        if line.opcode.startswith(("bra", "ret")):
            seen.clear()
            widened.clear()
        elif _mergeable(line, writes):
            seen[key] = (dst, len(keep))
        elif line.opcode == "cvt.f64.f32" and _single(line, writes) and writes[line.operands[1]] == 1:
            widened[dst] = (line.operands[1], len(keep))
        keep.append(line)
    return keep


def _single(line: Instr, writes: Counter) -> bool:
    """An unguarded instruction whose result register no other instruction writes."""
    return line.guard is None and _REG.fullmatch(line.operands[0]) is not None and writes[line.operands[0]] == 1


def _mergeable(line: Instr, writes: Counter) -> bool:
    return (line.guard is None and line.opcode.split(".")[0] in _MERGED and len(line.operands) > 1
            and _REG.fullmatch(line.operands[0]) is not None and writes[line.operands[0]] == 1)


def _written(line: Instr) -> list[str]:
    if not line.operands or line.opcode.startswith(_NO_DESTINATION):
        return []
    return _REG.findall(line.operands[0])


def _sources(key: tuple) -> list[str]:
    return [r for operand in key[1:] for r in _REG.findall(operand)]


def _reads_since(keep: list, start: int, reg: str) -> bool:
    """Whether an instruction after keep[start] reads `reg` (its previous trip's value, in a loop)."""
    for line in keep[start + 1:]:
        if isinstance(line, Instr):
            operands = line.operands if line.opcode.startswith(_NO_DESTINATION) else line.operands[1:]
            if any(reg in _REG.findall(o) for o in operands) or (line.guard and reg in line.guard):
                return True
    return False


def _renamed(line: Instr, rename: dict[str, str]) -> Instr:
    def sub(text: str) -> str:
        return _REG.sub(lambda m: rename.get(m.group(0), m.group(0)), text)

    return Instr(line.opcode, [sub(o) for o in line.operands], sub(line.guard) if line.guard else None)
