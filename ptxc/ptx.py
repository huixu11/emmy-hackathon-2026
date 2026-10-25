"""PTX text emission: virtual registers, labels, instructions and the module layout.

Registers are virtual and unlimited (%f0, %f1, ...); ptxas performs the real allocation, so
every value simply gets a fresh register and the counts are declared at the top of the entry.
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass

PTX_VERSION = "8.0"
PTX_TARGET = "sm_89"

# register class -> (name prefix, .reg type)
REG_CLASSES = {
    "pred": ("%p", ".pred"),
    "b16": ("%h", ".b16"),
    "b32": ("%r", ".b32"),
    "b64": ("%rd", ".b64"),
    "f32": ("%f", ".f32"),
    "f64": ("%fd", ".f64"),
}


def f32_imm(value: float) -> str:
    """A single-precision immediate spelled as its exact bit pattern: 1.0 -> 0f3F800000."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", value))[0]


def f64_imm(value: float) -> str:
    """A double-precision immediate spelled as its exact bit pattern: 1.0 -> 0d3FF0000000000000."""
    return "0d%016X" % struct.unpack("<Q", struct.pack("<d", value))[0]


def is_register(operand: str) -> bool:
    return operand.startswith("%")


def register_class(reg: str) -> str | None:
    """The class of a virtual register ("f32" for %f3, "b64" for %rd7), None for anything else."""
    m = re.fullmatch(r"(%[a-z]+)\d+", reg)
    return next((cls for cls, (prefix, _) in REG_CLASSES.items() if m and prefix == m.group(1)), None)


def address(base: str, offset: int = 0) -> str:
    """A memory operand: [%rd3], [%rd3+16], [sh_buf+-4] (ptxas accepts the signed form)."""
    return f"[{base}+{offset}]" if offset else f"[{base}]"


def vector(regs: list[str]) -> str:
    """The operand of a vector load/store: one register, or {%f0, %f1, ...}."""
    return regs[0] if len(regs) == 1 else "{" + ", ".join(regs) + "}"


@dataclass
class Instr:
    """One instruction, kept structured so that passes over the stream can rewrite it."""

    opcode: str
    operands: list[str]
    guard: str | None = None

    def render(self) -> str:
        pred = f"@{self.guard} " if self.guard else ""
        args = ", ".join(self.operands)
        return f"\t{pred}{self.opcode}\t{args};" if args else f"\t{pred}{self.opcode};"


class Mark:
    """A place in the stream that can still take instructions after later code is emitted:
    where a loop's preheader or a branch's entry is, for values hoisted out of the body."""

    def __init__(self) -> None:
        self.lines: list[Instr] = []


class PtxBuilder:
    """The instruction stream of one kernel, plus the register and label counters behind it.
    `lines` holds Instr records, Marks and, as plain strings, labels and comments."""

    def __init__(self) -> None:
        self.lines: list[Instr | Mark | str] = []
        self._reg_counts: Counter[str] = Counter()
        self._label_count = 0

    def mark(self) -> Mark:
        mark = Mark()
        self.lines.append(mark)
        return mark

    def flat(self) -> list[Instr | str]:
        """The stream with every Mark replaced by the instructions placed at it."""
        return [line for item in self.lines for line in (item.lines if isinstance(item, Mark) else [item])]

    def reg(self, cls: str) -> str:
        n = self._reg_counts[cls]
        self._reg_counts[cls] += 1
        return f"{REG_CLASSES[cls][0]}{n}"

    def labels(self, *hints: str) -> list[str]:
        """A family of fresh labels sharing one number: $L3_loop, $L3_loop_end."""
        self._label_count += 1
        return [f"$L{self._label_count}_{hint}" for hint in hints]

    def emit(self, opcode: str, *operands, guard: str | None = None, at: Mark | None = None) -> None:
        """Append an instruction, or place it at an earlier Mark."""
        (at.lines if at else self.lines).append(Instr(opcode, [str(o) for o in operands], guard))

    def place(self, label: str) -> None:
        self.lines.append(f"{label}:")

    def comment(self, text: str) -> None:
        # PTX rejects non-ASCII anywhere, comments included.
        self.lines.append("\t// " + text.encode("ascii", "replace").decode("ascii"))

    def reg_declarations(self) -> list[str]:
        return [f"\t.reg {REG_CLASSES[cls][1]}\t{REG_CLASSES[cls][0]}<{n}>;"
                for cls, n in self._reg_counts.items()]


def render_module(entry: str, params: list[str], shared: list[str], body: PtxBuilder,
                  max_threads: int | None = None) -> str:
    """Assemble one .entry and its module-scope shared arrays into a complete PTX file.
    `max_threads`, the launch's block size when known, becomes .maxntid (what CUDA's
    __launch_bounds__ says), so ptxas can budget registers for exactly that block."""
    lines = [f".version {PTX_VERSION}", f".target {PTX_TARGET}", ".address_size 64", ""]
    if shared:
        lines += shared + [""]
    lines.append(f".visible .entry {entry}(")
    lines.append(",\n".join(f"\t.param .u64 {p}" for p in params))
    lines.append(")")
    if max_threads is not None:
        lines.append(f".maxntid {max_threads}, 1, 1")
    lines.append("{")
    lines += body.reg_declarations()
    lines.append("")
    lines += [item.render() if isinstance(item, Instr) else item for item in body.flat()]
    lines.append("}")
    return "\n".join(lines) + "\n"
