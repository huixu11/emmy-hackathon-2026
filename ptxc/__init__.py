"""ptxc: a Kernel IR -> PTX compiler for the Emmy hackathon.

    kernel.py  load the IR, the buffers and the launch contract
    lower.py   walk the IR statements and emit PTX for each
    index.py   integer index arithmetic, predicates, addresses
    fmath.py   f32 elementwise operations
    scope.py   IR name scopes and the reuse cache
    ptx.py     registers, labels, instruction text, module layout
"""

import os
from pathlib import Path

from . import dtypes as dt
from .errors import CompileError
from .kernel import load_kernel
from .lower import KernelLowering

__all__ = ["PTX_LIMIT", "CompileError", "compile_file"]

PTX_LIMIT = 1_900_000  # the runner takes at most 2,000,000 characters of PTX


def compile_file(ir_path: str | Path, exact_f64: bool | None = None) -> str:
    """Compile one Kernel IR json file to the text of a PTX module. A module that would be too
    long for the runner (long loops unrolled whole) is compiled again with f64 reads as one
    `cvt` each, then with less unrolling (integer f64 reads first again).

    f64 values are held in f32 registers, as emmy's own CUDA holds them, unless exact_f64 (or,
    when it is None, the environment variable PTXC_EXACT_F64=1) asks for f64 arithmetic."""
    if exact_f64 is None:
        exact_f64 = os.environ.get("PTXC_EXACT_F64", "0") not in ("", "0")
    with dt.exact_f64(exact_f64):
        kernel = load_kernel(Path(ir_path))
        tries = [(scale, integer_reads) for scale in (1.0, 0.5, 0.25, 0.0) for integer_reads in (True, False)]
        while tries:
            scale, integer_reads = tries.pop(0)
            with dt.integer_f64_reads(integer_reads):
                lowering = KernelLowering(kernel, unroll_scale=scale)
                ptx = lowering.run()
            if len(ptx) <= PTX_LIMIT:
                return ptx
            if integer_reads and not lowering.b.f64_reads:
                tries.pop(0)  # it read no f64: the same module again
    raise CompileError(f"the PTX is {len(ptx)} characters even without unrolling, over the runner's limit")
