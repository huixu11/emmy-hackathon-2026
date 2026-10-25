#!/usr/bin/env python3
"""Run problems with bf16 or bool buffers, which the GPU runner cannot allocate itself.

    python tools/run_packed.py tests/problems_packed/d_p_add_bf16 ...
    python tools/run_packed.py --dir tests/problems_packed --via-proxy

The runner knows f16, f32, f64, i32 and i64 buffers. A bf16 or bool buffer is the same bytes
as an i32 buffer a half (bf16) or a quarter (bool) as long, so this passes it as one: inputs
are seeded the way tools/generate.py seeds them (float32 normals rounded to bfloat16, or their
truth values) and packed into int32 words; outputs come back as int32 words and are unpacked
before the comparison with expected.npz. Kernels run in meta.json's order, chaining the
intermediates, as submit.py does.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import json  # noqa: E402

import submit  # noqa: E402
from generate import bf16  # noqa: E402
from run_problems import tunneled_post  # noqa: E402

_PER_WORD = {"bf16": 2, "bool": 4}


def _seeded(arg: dict) -> np.ndarray:
    rng = np.random.default_rng(arg["seed"])
    values = rng.standard_normal(math.prod(arg["shape"]))
    return bf16(values) if arg["dtype"] == "bf16" else values.astype(np.bool_)


def _pack(values: np.ndarray, dtype: str) -> list[int]:
    if dtype == "bf16":
        raw = (bf16(np.asarray(values, dtype=np.float32)).view(np.uint32) >> 16).astype(np.uint16)
    else:
        raw = np.asarray(values, dtype=np.uint8)
    pad = (-raw.nbytes) % 4
    data = raw.tobytes() + b"\0" * pad
    return np.frombuffer(data, dtype=np.int32).tolist()


def _unpack(words: list, dtype: str, n: int) -> np.ndarray:
    data = np.asarray(words, dtype=np.int64).astype(np.int32).tobytes()
    if dtype == "bf16":
        half = np.frombuffer(data, dtype=np.uint16)[:n].astype(np.uint32) << 16
        return half.view(np.float32)
    return np.frombuffer(data, dtype=np.uint8)[:n].astype(np.float32)


def run(post, pdir: Path, compiler: str) -> tuple[bool, str]:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    produced: dict[str, list] = {}  # intermediate name -> runner values (packed when bf16/bool)
    report = []
    for k in meta["kernels"]:
        ir = pdir / f"{k['entry']}.ir.json"
        r = subprocess.run([sys.executable, compiler, str(ir)], capture_output=True, text=True)
        if r.returncode:
            return False, f"[{k['entry']}] compiler failed: {r.stderr.strip()[-300:]}"
        compiled = post(submit.RUNNER, "/compile", {"ptx": r.stdout})
        if not compiled.get("ok"):
            return False, f"[{k['entry']}] ptxas rejected it: {compiled.get('stderr', '')[:300]}"
        args, packed = [], {}
        for a in k["args"]:
            a = dict(a)
            n = math.prod(a["shape"])
            if a["dtype"] in _PER_WORD:
                packed[a["name"]] = (a["dtype"], n)
                words = -(-n // _PER_WORD[a["dtype"]])
                if a.pop("intermediate", False):
                    a["data"] = produced[a["name"]]
                elif not a.get("output"):
                    a["data"] = _pack(_seeded(a) if "seed" in a else np.asarray(a["data"]), a["dtype"])
                a.pop("seed", None)
                a.update(dtype="i32", shape=[words])
            elif a.pop("intermediate", False):
                a["data"] = produced[a["name"]]
            args.append(a)
        need = max((math.prod(a["shape"]) for a in args if a.get("output")), default=1)
        out = post(submit.RUNNER, "/run", {
            "cubin_id": compiled["cubin_id"], "entry": k["entry"], "grid": k["grid"], "block": k["block"],
            "smem_bytes": k["smem_bytes"], "zero_outputs": k["zero_outputs"], "args": args,
            "warmup": 0, "iters": 1, "max_return": max(4096, need)})
        if not out.get("ok"):
            return False, f"[{k['entry']}] launch failed: {out.get('stderr', '')[:300]}"
        for name, o in out["outputs"].items():
            produced[name] = o["values"]
            if name not in expected:
                continue
            if name in packed:
                got = _unpack(o["values"], *packed[name])
            else:
                got = np.array(o["values"], dtype=np.float32)
            got = got.reshape(expected[name].shape)
            ref = expected[name].astype(np.float32)
            ok = np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"])
            report.append(f"[{k['entry']}] {'PASS' if ok else 'FAIL'}  {name}: max abs err "
                          f"{float(np.max(np.abs(got - ref))):.3e}")
            if not ok:
                return False, " | ".join(report)
    return True, " | ".join(report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path, default=REPO / "tests/problems_packed")
    ap.add_argument("--compiler", default=str(REPO / "compile.py"))
    ap.add_argument("--via-proxy", action="store_true", help="tunnel runner requests through $HTTPS_PROXY")
    args = ap.parse_args()
    post = tunneled_post if args.via_proxy else submit.post
    problems = args.problems or sorted(p for p in args.dir.iterdir() if (p / "meta.json").exists())
    failed = 0
    for pdir in problems:
        ok, line = run(post, pdir, args.compiler)
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {pdir.name:22s} {line}")
    print(f"\n{len(problems) - failed}/{len(problems)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
