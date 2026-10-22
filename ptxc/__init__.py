"""ptxc: a Kernel IR -> PTX compiler for the Emmy hackathon.

    kernel.py  load the IR, the buffers and the launch contract
    lower.py   walk the IR statements and emit PTX for each
    index.py   integer index arithmetic, predicates, addresses
    fmath.py   f32 elementwise operations
    scope.py   IR name scopes and the reuse cache
    ptx.py     registers, labels, instruction text, module layout
"""

from pathlib import Path

from .errors import CompileError
from .kernel import load_kernel
from .lower import KernelLowering

__all__ = ["CompileError", "compile_file"]


def compile_file(ir_path: str | Path) -> str:
    """Compile one Kernel IR json file to the text of a PTX module."""
    return KernelLowering(load_kernel(Path(ir_path))).run()
