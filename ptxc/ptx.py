"""PTX text emission: virtual registers, labels, instructions and the module layout.

Registers are virtual and unlimited (%f0, %f1, ...); ptxas performs the real allocation, so
every value simply gets a fresh register and the counts are declared at the top of the entry.
"""

from __future__ import annotations

import struct
from collections import Counter

PTX_VERSION = "8.0"
PTX_TARGET = "sm_89"

# register class -> (name prefix, .reg type)
REG_CLASSES = {
    "pred": ("%p", ".pred"),
    "b32": ("%r", ".b32"),
    "b64": ("%rd", ".b64"),
    "f32": ("%f", ".f32"),
}


def f32_imm(value: float) -> str:
    """A single-precision immediate spelled as its exact bit pattern: 1.0 -> 0f3F800000."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", value))[0]


def is_register(operand: str) -> bool:
    return operand.startswith("%")


def address(base: str, offset: int = 0) -> str:
    """A memory operand: [%rd3], [%rd3+16], [sh_buf+-4] (ptxas accepts the signed form)."""
    return f"[{base}+{offset}]" if offset else f"[{base}]"


def vector(regs: list[str]) -> str:
    """The operand of a vector load/store: one register, or {%f0, %f1, ...}."""
    return regs[0] if len(regs) == 1 else "{" + ", ".join(regs) + "}"


class PtxBuilder:
    """The instruction stream of one kernel, plus the register and label counters behind it."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._reg_counts: Counter[str] = Counter()
        self._label_count = 0

    def reg(self, cls: str) -> str:
        n = self._reg_counts[cls]
        self._reg_counts[cls] += 1
        return f"{REG_CLASSES[cls][0]}{n}"

    def labels(self, *hints: str) -> list[str]:
        """A family of fresh labels sharing one number: $L3_loop, $L3_loop_end."""
        self._label_count += 1
        return [f"$L{self._label_count}_{hint}" for hint in hints]

    def emit(self, opcode: str, *operands, guard: str | None = None) -> None:
        pred = f"@{guard} " if guard else ""
        args = ", ".join(str(o) for o in operands)
        self.lines.append(f"\t{pred}{opcode}\t{args};" if args else f"\t{pred}{opcode};")

    def place(self, label: str) -> None:
        self.lines.append(f"{label}:")

    def comment(self, text: str) -> None:
        # PTX rejects non-ASCII anywhere, comments included.
        self.lines.append("\t// " + text.encode("ascii", "replace").decode("ascii"))

    def reg_declarations(self) -> list[str]:
        return [f"\t.reg {REG_CLASSES[cls][1]}\t{REG_CLASSES[cls][0]}<{n}>;"
                for cls, n in self._reg_counts.items()]


def render_module(entry: str, params: list[str], shared: list[str], body: PtxBuilder) -> str:
    """Assemble one .entry and its module-scope shared arrays into a complete PTX file."""
    lines = [f".version {PTX_VERSION}", f".target {PTX_TARGET}", ".address_size 64", ""]
    if shared:
        lines += shared + [""]
    lines.append(f".visible .entry {entry}(")
    lines.append(",\n".join(f"\t.param .u64 {p}" for p in params))
    lines += [")", "{"]
    lines += body.reg_declarations()
    lines.append("")
    lines += body.lines
    lines.append("}")
    return "\n".join(lines) + "\n"
