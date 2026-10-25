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

__all__ = ["PTX_LIMIT", "CompileError", "compile_file"]

PTX_LIMIT = 1_900_000  # the runner takes at most 2,000,000 characters of PTX


def compile_file(ir_path: str | Path) -> str:
    """Compile one Kernel IR json file to the text of a PTX module. A module that would be too
    long for the runner (long loops unrolled whole) is compiled again with less unrolling."""
    kernel = load_kernel(Path(ir_path))
    for scale in (1.0, 0.5, 0.25, 0.0):
        ptx = KernelLowering(kernel, unroll_scale=scale).run()
        if len(ptx) <= PTX_LIMIT:
            return ptx
    raise CompileError(f"the PTX is {len(ptx)} characters even without unrolling, over the runner's limit")
