"""The 15 open problems again, with every input in f16, bf16, f64 or i32, and the 49 attention
and reduction problems of tests/expressions_variants.py in f16 and f64: the same kernels in
other data types. The f16, f64 and i32 ones are in tests/problems; the bf16 ones, which the
runner cannot allocate, are in tests/problems_packed (tests/expressions_packed.py takes them
from here).

    python tools/generate.py tests/expressions_cross.py --emmy emmy --out tests/problems

Two combinations are missing because emmy cannot schedule them: bf16 attention (a01, a02)
stops in emmy's own lowering. i32 is left out where torch has no integer form (mean, sqrt,
softmax, division). emmy accumulates f64 matmuls in f32, as its own CUDA does, so those two
are checked at f32 accuracy.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from generate import bf16  # noqa: E402

f64 = lambda a: np.asarray(a).astype(np.float64)


def _softmax(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


REFS = {  # each open problem in float64
    "a01_attention": lambda q, kt, v: _softmax(f64(q) @ f64(kt) / 8.0) @ f64(v),
    "a02_attention_128": lambda q, kt, v: _softmax(f64(q) @ f64(kt) / 5.656854) @ f64(v),
    "f01_rmsnorm": lambda x, w: f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-5) * f64(w),
    "f02_row_softmax": lambda x: _softmax(f64(x)),
    "f03_l2norm": lambda x: f64(x) / np.sqrt((f64(x) ** 2).sum(-1, keepdims=True)),
    "m01_matmul_512": lambda a, b: f64(a) @ f64(b),
    "m02_matmul_rect": lambda a, b: f64(a) @ f64(b),
    "p00_scalar_add": lambda a, b: f64(a) + f64(b),
    "p01_vector_add": lambda a, b: f64(a) + f64(b),
    "p02_scale_shift": lambda x: f64(x) * 2.0 + 1.0,
    "p03_relu_mul": lambda x, y: np.maximum(f64(x), 0) * f64(y),
    "r01_array_sum": lambda x: f64(x).sum(),
    "r02_big_sum": lambda x: f64(x).sum(),
    "r03_mean": lambda x: f64(x).mean(),
    "r04_row_sum": lambda x: f64(x).sum(-1),
}
FLOATS = {  # dtype -> (torch name, rounding of the float64 result, tolerance)
    "f16": ("torch.float16", lambda r: np.asarray(r).astype(np.float16), 2e-2),
    "bf16": ("torch.bfloat16", lambda r: bf16(r), 5e-2),
    "f64": ("torch.float64", lambda r: np.asarray(r).astype(np.float64), 1e-6),
}
INTEGER = ("p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul",
           "r01_array_sum", "r02_big_sum", "r04_row_sum", "m01_matmul_512", "m02_matmul_rect")
NOT_SCHEDULED = {"c_bf16_a01_attention", "c_bf16_a02_attention_128"}
F32_ACCUMULATED = {"c_f64_m01_matmul_512", "c_f64_m02_matmul_rect"}

ALL = []
for meta in sorted(Path(__file__).resolve().parent.parent.glob("problems/*/meta.json")):
    meta = json.loads(meta.read_text())
    pid, code = meta["id"], meta["code"]
    for name, (torch_dtype, rounded, tol) in FLOATS.items():
        sid = f"c_{name}_{pid}"
        if sid in NOT_SCHEDULED:
            continue
        if sid in F32_ACCUMULATED:
            tol = 1e-4
        ALL.append(dict(id=sid, code=re.sub(r"torch\.randn\(([^)]*)\)", rf"torch.randn(\1, dtype={torch_dtype})", code),
                        ref=lambda *a, f=REFS[pid], r=rounded: [r(f(*a))], rtol=tol, atol=tol))
    if pid in INTEGER:
        c = re.sub(r"torch\.randn\(([^)]*)\)", r"torch.randint(-3, 4, (\1,), dtype=torch.int32)", code)
        ALL.append(dict(id=f"c_i32_{pid}", code=c.replace("* 2.0 + 1.0", "* 2 + 1"),
                        ref=lambda *a, f=REFS[pid]: [np.rint(f(*a)).astype(np.int64)], rtol=0, atol=0))

# The attention (`_a_`, `_s_`) and reduction (`_r_`) problems of tests/expressions_variants.py,
# in f16 and f64. Their references compute in float64; f16 is checked at f16 accuracy, f64 at
# the spec's own (f32) accuracy, since emmy keeps some f64 accumulators in f32. An f16 max is
# exact; an f64 one is not, as emmy's tree combine goes through a `float` shared array.
_spec = importlib.util.spec_from_file_location("variants", Path(__file__).resolve().parent / "expressions_variants.py")
_variants = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_variants)
for spec in _variants.OPEN:
    if not re.match(r"[vwxyz]_[asr]_", spec["id"]):
        continue
    for name, torch_dtype in (("f16", "torch.float16"), ("f64", "torch.float64")):
        tol = spec["rtol"], spec["atol"]
        if name == "f16" and max(tol) > 0:
            tol = max(tol[0], 2e-2), max(tol[1], 2e-2)
        elif name == "f64" and max(tol) == 0:
            tol = 1e-6, 1e-6
        ALL.append(dict(spec, id=f"c_{name}_{spec['id']}", rtol=tol[0], atol=tol[1],
                        code=re.sub(r"torch\.randn\(([^)]*)\)", rf"torch.randn(\1, dtype={torch_dtype})", spec["code"])))

OPEN = [s for s in ALL if not s["id"].startswith("c_bf16_")]
