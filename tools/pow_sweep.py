#!/usr/bin/env python3
"""Check pow(x, c) on the GPU runner for many constant exponents c, against float64.

    python tools/pow_sweep.py                  # the default exponents
    python tools/pow_sweep.py 1.5 -0.5 2.2     # just these
    python tools/pow_sweep.py --via-proxy      # as in run_problems.py

Each exponent is put into tests/problems/v_p_pow3's IR (its ConstantOp), compiled with
compile.py and run on 2048 inputs: zeros of both signs, infinities, NaN, subnormals, the
largest floats, then random magnitudes from 6e-6 to 1.6e5 of both signs. Prints the largest
relative error over normal results and every input whose result has the wrong class (NaN,
infinity, zero or its sign) under C's powf rules. The runner returns infinities as NaN, so a
NaN where an infinity is expected cannot be told apart and is not counted.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import submit  # noqa: E402
from run_problems import tunneled_post  # noqa: E402

TEMPLATE = REPO / "tests/problems/v_p_pow3"
EXPONENTS = [0, 1, -1, 2, 3, -2, -3, 7, 64, -64, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 63.5,
             0.3, -0.7, 1 / 3, 2.2, 65, 66, -65, 100, 1e10, 0.125]
SPECIAL = [0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1e-40,
           -1e-40, 3.4e38, -3.4e38, 1e-30, -1e-30, 1e30, -1e30, 1.17549435e-38, 3.0, -3.0]


def c_pow(a: float, c: float) -> float:
    """C's pow in float64 (math.pow), with the domain errors as values. numpy's power is not
    used: it takes exponent 0.5 as sqrt, which gives sqrt(-0) = -0 where C's pow gives +0."""
    try:
        return math.pow(a, c)
    except ValueError:  # negative base, fractional exponent
        return math.nan
    except OverflowError:
        return math.copysign(math.inf, a) if c == int(c) and int(c) % 2 else math.inf


def inputs(n: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    k = n - len(SPECIAL)
    mags = np.exp(rng.uniform(-12, 12, k)) * np.where(rng.random(k) < 0.5, -1.0, 1.0)
    return np.concatenate([np.array(SPECIAL), mags]).astype(np.float32)


def run(post, compiler: str, c: float, x: np.ndarray) -> np.ndarray:
    ir = json.loads(next(TEMPLATE.glob("*.ir.json")).read_text())
    meta = json.loads((TEMPLATE / "meta.json").read_text())
    kernel = meta["kernels"][0]
    const = kernel["arg_order"][0]  # pow_1_c1, the exponent
    ir["nodes"][const]["op_fields"]["value"] = c
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "kernel.ir.json"
        path.write_text(json.dumps(ir))
        ptx = subprocess.run([sys.executable, compiler, str(path)], capture_output=True, text=True, check=True).stdout
    compiled = post(submit.RUNNER, "/compile", {"ptx": ptx})
    if not compiled.get("ok"):
        sys.exit(f"ptxas rejected the kernel for c={c}: {compiled.get('stderr', '')}")
    args = [{"name": const, "dtype": "f32", "shape": [1], "data": [c]},
            {"name": "x", "dtype": "f32", "shape": [len(x)], "data": [float(v) for v in x]},
            {"name": "pow_1", "dtype": "f32", "shape": [len(x)], "output": True}]
    r = post(submit.RUNNER, "/run", {
        "cubin_id": compiled["cubin_id"], "entry": kernel["entry"], "grid": kernel["grid"],
        "block": kernel["block"], "smem_bytes": 0, "zero_outputs": [], "args": args,
        "warmup": 0, "iters": 1, "max_return": len(x)})
    if not r.get("ok"):
        sys.exit(f"launch failed for c={c}: {r.get('stderr', '')}")
    return np.array(r["outputs"]["pow_1"]["values"], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exponents", nargs="*", type=float)
    ap.add_argument("--compiler", default=str(REPO / "compile.py"))
    ap.add_argument("--via-proxy", action="store_true", help="tunnel runner requests through $HTTPS_PROXY")
    args = ap.parse_args()
    post = tunneled_post if args.via_proxy else submit.post

    x = inputs(2048)  # the template launch: 2 blocks of 256 threads, 4 values each
    wrong_total = 0
    for c in args.exponents or EXPONENTS:
        c = float(np.float32(c))
        with np.errstate(all="ignore"):
            ref = np.array([c_pow(float(v), c) for v in x]).astype(np.float32)
        got = run(post, args.compiler, c, x)
        normal = np.isfinite(ref) & np.isfinite(got) & (np.abs(ref) >= np.finfo(np.float32).tiny)
        rel = np.abs(got[normal].astype(np.float64) / ref[normal] - 1).max(initial=0.0)
        same_class = ((np.isnan(got) & np.isnan(ref)) | (np.isfinite(got) & np.isfinite(ref))
                      | ((got == ref) & (np.signbit(got) == np.signbit(ref))))
        exact_zero = (ref == 0) & (got == 0)
        same_class &= ~exact_zero | (np.signbit(got) == np.signbit(ref))
        wrong = ~same_class & ~(np.isinf(ref) & np.isnan(got))
        wrong_total += int(wrong.sum())
        cases = ", ".join(f"pow({x[i]:g}) = {got[i]:g}, not {ref[i]:g}" for i in np.nonzero(wrong)[0][:4])
        print(f"c = {c:<12.6g} max relative error {rel:.2e}   {int(wrong.sum())} wrong special values"
              + (f": {cases}" if cases else ""))
    return 1 if wrong_total else 0


if __name__ == "__main__":
    sys.exit(main())
