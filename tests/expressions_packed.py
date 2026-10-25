"""Problems with bf16 or bool buffers, which the GPU runner cannot allocate itself: generated into
tests/problems_packed/ and run with tools/run_packed.py, which passes them as int32 words.

    python tools/generate.py tests/expressions_packed.py --emmy emmy --out tests/problems_packed
    python tools/run_packed.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from generate import bf16  # noqa: E402

f64 = lambda a: np.asarray(a).astype(np.float64)
f32 = lambda a: np.asarray(a).astype(np.float32)

OPEN = [
    dict(id="d_p_bf16_add", code="torch.randn(4096, dtype=torch.bfloat16) + torch.randn(4096, dtype=torch.bfloat16)",
         ref=lambda a, b: [bf16(f32(a) + f32(b))], rtol=0, atol=0),
    dict(id="d_p_bf16_scale", code="torch.randn(4096, dtype=torch.bfloat16) * 2.0",
         ref=lambda x: [bf16(f32(x) * 2)], rtol=0, atol=0),
    dict(id="d_p_bf16_exp", code="torch.exp(torch.randn(4096, dtype=torch.bfloat16))",
         ref=lambda x: [bf16(np.exp(f64(x)))], rtol=1e-2, atol=1e-2),
    dict(id="d_p_bf16_to_f32", code="torch.randn(4096, dtype=torch.bfloat16).float() + 1.0",
         ref=lambda x: [f32(x) + 1], rtol=0, atol=0),
    dict(id="d_p_f32_to_bf16", code="torch.randn(4096).to(torch.bfloat16)",
         ref=lambda x: [bf16(x)], rtol=0, atol=0),
    dict(id="d_r_bf16_sum", code="torch.randn(64, 256, dtype=torch.bfloat16).sum(-1)",
         ref=lambda x: [bf16(f64(x).sum(-1))], rtol=1e-2, atol=1e-2),
    dict(id="d_p_gt_bool", code="torch.randn(4096) > 0.5",
         ref=lambda x: [x > np.float32(0.5)], rtol=0, atol=0),
    dict(id="d_p_isclose_bool", code="x = torch.randn(4096); y = torch.randn(4096); (x - y).abs() <= 0.5",
         ref=lambda x, y: [np.abs(x - y) <= np.float32(0.5)], rtol=0, atol=0),
    dict(id="t_m_bf16_128", code="torch.randn(128, 128, dtype=torch.bfloat16) @ torch.randn(128, 128, dtype=torch.bfloat16)",
         ref=lambda a, b: [bf16(f64(a) @ f64(b))], rtol=1e-2, atol=5e-2),
    dict(id="t_m_bf16_256", code="torch.randn(256, 256, dtype=torch.bfloat16) @ torch.randn(256, 256, dtype=torch.bfloat16)",
         ref=lambda a, b: [bf16(f64(a) @ f64(b))], rtol=1e-2, atol=1e-1),
]

# The open problems with bf16 inputs (tests/expressions_cross.py).
_cross = importlib.util.spec_from_file_location("cross", Path(__file__).resolve().parent / "expressions_cross.py")
_mod = importlib.util.module_from_spec(_cross)
_cross.loader.exec_module(_mod)
OPEN += [s for s in _mod.ALL if s["id"].startswith("c_bf16_")]
