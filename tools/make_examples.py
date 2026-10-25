#!/usr/bin/env python3
"""Regenerate examples/compiled/: a few kernels' IR next to the PTX compile.py makes of it.

    python tools/make_examples.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ptxc import compile_file  # noqa: E402

EXAMPLES = {  # example name -> (problem folder, kernel entry)
    "1_row_sum": ("problems/r04_row_sum", "k_sum_1_reduce"),
    "2_matmul_cp_async": ("problems/m01_matmul_512", None),
    "3_attention": ("problems/a02_attention_128", None),
    "4_tensor_core_matmul": ("tests/problems/t_m_f16_256x256x256", None),
    "5_f16_softmax": ("tests/problems/d_f_f16_softmax", None),
    "6_integer_floor_divide": ("tests/problems/d_p_i32_floordiv", None),
    "7_f16_causal_attention": ("tests/problems/t_s_f16_causal_1x1x64x32", None),
}


def main() -> None:
    out = REPO / "examples/compiled"
    out.mkdir(parents=True, exist_ok=True)
    for name, (folder, entry) in EXAMPLES.items():
        irs = sorted((REPO / folder).glob("*.ir.json"))
        ir = next(p for p in irs if entry is None or p.name == f"{entry}.ir.json")
        shutil.copy(ir.with_suffix("").with_suffix(".ir.txt"), out / f"{name}.ir.txt")
        (out / f"{name}.ptx").write_text(compile_file(ir))
        print(f"examples/compiled/{name}.ptx  <- {ir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
