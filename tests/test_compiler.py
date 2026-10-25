"""Offline tests of the compiler (no GPU needed).

    python -m unittest discover -s tests -v

Every problem under problems/, tests/problems and tests/problems_packed must compile to PTX
whose entry and parameters match meta.json, and the same PTX on every run. With ptxas on PATH
(or named by $PTXAS) each module must also assemble for sm_89. The checks on the GPU are
tools/run_problems.py and tools/run_packed.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ptxc import CompileError, compile_file  # noqa: E402
from ptxc import dtypes as dt  # noqa: E402

PROBLEM_DIRS = [REPO / "problems", REPO / "tests/problems", REPO / "tests/problems_packed"]
KERNELS = sorted(ir for d in PROBLEM_DIRS if d.exists() for ir in d.glob("*/*.ir.json"))
PTXAS = os.environ.get("PTXAS") or shutil.which("ptxas")


def _launch(ir: Path) -> dict:
    meta = json.loads((ir.parent / "meta.json").read_text())
    entry = ir.name.removesuffix(".ir.json")
    return next(k for k in meta["kernels"] if k["entry"] == entry)


class EveryProblem(unittest.TestCase):
    def test_compiles_to_the_launch_contract(self):
        """The entry is the IR's kernel name and the parameters are meta.json's arg_order."""
        self.assertGreater(len(KERNELS), 150)
        for ir in KERNELS:
            with self.subTest(kernel=f"{ir.parent.name}/{ir.name}"):
                ptx = compile_file(str(ir))
                launch = _launch(ir)
                self.assertIn(".target sm_89", ptx)
                self.assertIn(f".visible .entry {launch['entry']}(", ptx)
                params = re.findall(r"\.param \.u64 param_(\w+)", ptx)
                self.assertEqual(params, launch["arg_order"])
                self.assertIn(f".maxntid {launch['block'][0]}, 1, 1", ptx)

    @unittest.skipUnless(PTXAS, "ptxas not found (set $PTXAS or put it on PATH)")
    def test_assembles_with_ptxas(self):
        with tempfile.TemporaryDirectory() as td:
            for ir in KERNELS:
                with self.subTest(kernel=f"{ir.parent.name}/{ir.name}"):
                    path = Path(td) / "k.ptx"
                    path.write_text(compile_file(str(ir)))
                    r = subprocess.run([PTXAS, "-arch=sm_89", str(path), "-o", os.devnull],
                                       capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stderr[-500:])

    def test_same_ptx_on_every_run(self):
        """Nothing may depend on Python's string hashing, which changes from process to process."""
        picks = [ir for ir in KERNELS if ir.parent.name in
                 ("a01_attention", "v_a_rect", "m01_matmul_512", "t_m_f16_128x128x128", "d_f_f16_softmax")]
        for ir in picks:
            with self.subTest(kernel=ir.parent.name):
                outs = {subprocess.run([sys.executable, str(REPO / "compile.py"), str(ir)], capture_output=True,
                                       text=True, env={**os.environ, "PYTHONHASHSEED": seed}).stdout
                        for seed in ("1", "2", "3")}
                self.assertEqual(len(outs), 1)


class Errors(unittest.TestCase):
    def test_missing_file_is_one_line(self):
        r = subprocess.run([sys.executable, str(REPO / "compile.py"), "no/such/kernel.ir.json"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)

    def test_unsupported_input_stops_with_a_message(self):
        """An IR the compiler cannot translate stops it, never with wrong PTX."""
        ir = next(p for p in KERNELS if p.parent.name == "p00_scalar_add")
        raw = json.loads(ir.read_text())
        for node in raw["nodes"].values():
            for out in node["outputs"]:
                out["dtype"] = "f8e4m3"
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / ir.name
            bad.write_text(json.dumps(raw))
            with self.assertRaises(CompileError):
                compile_file(str(bad))


class DataTypes(unittest.TestCase):
    def test_immediates_round_to_the_dtype(self):
        self.assertEqual(dt.immediate(1.0, dt.F32), "0f3F800000")
        self.assertEqual(dt.immediate(0.1, dt.F16), "0f3DCCC000")    # 0.0999755859375
        self.assertEqual(dt.immediate(0.1, dt.BF16), "0f3DCD0000")   # 0.10009765625
        self.assertEqual(dt.immediate(-1e30, dt.F16), "0fFF800000")  # past f16's range: -inf
        self.assertEqual(dt.immediate(0.1, dt.F64), "0d3FB999999999999A")

    def test_integer_identities_clamp(self):
        self.assertEqual(dt.immediate(-1e30, dt.I32), str(-2 ** 31))
        self.assertEqual(dt.immediate(float("inf"), dt.I64), str(2 ** 63 - 1))

    def test_names(self):
        self.assertIs(dt.dtype("__half"), dt.F16)
        self.assertIs(dt.dtype("float"), dt.F32)
        self.assertIs(dt.common([dt.I32, dt.F16]), dt.F32)
        self.assertIs(dt.common([dt.I32, dt.F64]), dt.F64)
        with self.assertRaises(CompileError):
            dt.dtype("f8e4m3")


if __name__ == "__main__":
    unittest.main()
