"""Offline tests of the compiler (no GPU needed).

    python -m unittest discover -s tests -v

Every problem under problems/, tests/problems and tests/problems_packed must compile to PTX
whose entry and parameters match meta.json, and the same PTX on every run. With ptxas on PATH
(or named by $PTXAS) each module must also assemble for sm_89, the f64 ones in both f64 modes.
The checks on the GPU are tools/run_problems.py and tools/run_packed.py.
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

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ptxc import PTX_LIMIT, CompileError, compile_file  # noqa: E402
from ptxc import dtypes as dt  # noqa: E402
from ptxc.cse import eliminate_common_subexpressions  # noqa: E402
from ptxc.ptx import Instr, PtxBuilder  # noqa: E402

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
                self.assertLessEqual(len(ptx), PTX_LIMIT)  # the runner's limit on a module

    @unittest.skipUnless(PTXAS, "ptxas not found (set $PTXAS or put it on PATH)")
    def test_assembles_with_ptxas(self):
        with tempfile.TemporaryDirectory() as td:
            for ir in KERNELS:
                for exact in (False, True) if "f64" in ir.parent.name else (False,):
                    with self.subTest(kernel=f"{ir.parent.name}/{ir.name}", exact_f64=exact):
                        path = Path(td) / "k.ptx"
                        path.write_text(compile_file(str(ir), exact_f64=exact))
                        r = subprocess.run([PTXAS, "-arch=sm_89", str(path), "-o", os.devnull],
                                           capture_output=True, text=True)
                        self.assertEqual(r.returncode, 0, r.stderr[-500:])

    def test_same_ptx_on_every_run(self):
        """Nothing may depend on Python's string hashing, which changes from process to process."""
        picks = [ir for ir in KERNELS if ir.parent.name in
                 ("a01_attention", "v_a_rect", "m01_matmul_512", "t_m_f16_128x128x128", "d_f_f16_softmax",
                  "t_s_f16_causal_1x1x64x32")]
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

    def test_f64_is_computed_in_f32_unless_exact(self):
        """By default an f64 value is held in f32, as emmy's CUDA holds it; --exact-f64 keeps f64."""
        ir = next(p for p in KERNELS if p.parent.name == "d_p_f64_mul")
        default, exact = compile_file(str(ir), exact_f64=False), compile_file(str(ir), exact_f64=True)
        self.assertIn("cvt.rn.f32.f64", default)
        self.assertIn("mul.rn.f32", default)
        self.assertIn("cvt.f64.f32", default)
        self.assertNotIn("mul.rn.f64", default)
        self.assertIn("mul.rn.f64", exact)
        self.assertNotIn("cvt.rn.f32.f64", exact)

    def test_f64_reads_in_loops_split_between_cvt_and_integers(self):
        """In a loop 2 f64 reads in 5 stay cvt and 3 are rounded in integer instructions (the
        f64 unit and the ALUs run side by side); straight-line code reads with cvt alone."""
        looped = compile_file(str(next(p for p in KERNELS if p.parent.name == "c_f64_m01_matmul_512")))
        self.assertGreater(looped.count("cvt.rn.f32.f64"), 0)
        self.assertGreater(looped.count("shf.r.wrap.b32"), looped.count("cvt.rn.f32.f64"))
        straight = compile_file(str(next(p for p in KERNELS if p.parent.name == "d_p_f64_mul")))
        self.assertIn("cvt.rn.f32.f64", straight)
        self.assertNotIn("shf.r.wrap.b32", straight)

    def test_integer_f64_reads_round_as_cvt(self):
        """dtypes._read_f64's integer form, run instruction by instruction in numpy, against
        numpy's float32(float64): random bit patterns (NaN and inf among them), exact f32 ties,
        both ends of f32's range and the subnormals between."""
        b = PtxBuilder()
        b.f64_reads = 1  # 2 reads in 5 stay cvt, this one is the integer form
        dt._read_f64(b, "%f0", "%fd0", integer=True)
        x = _f64_samples(np.random.default_rng(5), 50_000)
        got = _interpret(b.flat(), {"%fd0": x.view(np.uint64)})["%f0"].astype(np.uint32).view(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            want = x.astype(np.float32)
        nan = np.isnan(want)
        self.assertTrue(np.isnan(got[nan]).all())
        np.testing.assert_array_equal(got[~nan].view(np.uint32), want[~nan].view(np.uint32))

    def test_f64_states_cross_lanes_in_two_halves(self):
        """shfl.sync moves 32 bits: an f64 running max goes over as two halves."""
        ir = next(p for p in KERNELS if p.parent.name == "c_f64_f02_row_softmax")
        ptx = compile_file(str(ir), exact_f64=True)
        self.assertRegex(ptx, r"mov\.b64\s+\{%r\d+, %r\d+\}, %fd\d+;")
        self.assertRegex(ptx, r"mov\.b64\s+%fd\d+, \{%r\d+, %r\d+\};")
        self.assertNotRegex(ptx, r"shfl\.sync\.bfly\.b32\s+%f\d+, %fd")

    def test_names(self):
        self.assertIs(dt.dtype("__half"), dt.F16)
        self.assertIs(dt.dtype("float"), dt.F32)
        self.assertIs(dt.common([dt.I32, dt.F16]), dt.F32)
        self.assertIs(dt.common([dt.I32, dt.F64]), dt.F64)
        with self.assertRaises(CompileError):
            dt.dtype("f8e4m3")


def _f64_samples(rng, n: int) -> np.ndarray:
    """f64 values that exercise a conversion to f32 at every exponent and rounding case."""
    u = np.uint64
    bits = rng.integers(0, 2**64 - 1, n, dtype=np.uint64, endpoint=True)
    edge = (rng.integers(760, 906, n, dtype=np.uint64) << u(52)) | (bits & u(0x800FFFFFFFFFFFFF))  # 2^-263..2^-117
    parts = [bits.view(np.float64), edge.view(np.float64), rng.standard_normal(n)]
    for lo, hi in ((-151, -124), (-20, 20), (126, 129)):
        f = (np.exp2(rng.uniform(lo, hi, n)) * rng.choice([-1, 1], n)).astype(np.float32)
        g = np.nextafter(f, np.float32(np.inf), dtype=np.float32)
        tie = f.astype(np.float64) / 2 + g.astype(np.float64) / 2
        parts += [tie, np.nextafter(tie, -np.inf), np.nextafter(tie, np.inf)]
    parts.append(np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 2.0**-149, 2.0**-150, 2.0**-126, 2.0**128, 5e-324]))
    with np.errstate(over="ignore"):
        return np.concatenate(parts)


def _interpret(lines, regs: dict) -> dict:
    """Run straight-line integer and f32 PTX elementwise over numpy arrays: the instructions
    dtypes._read_f64 emits. 32-bit registers are held in uint64 arrays."""
    m32 = np.uint64(0xFFFFFFFF)

    def val(op):
        if op in regs:
            return regs[op]
        return np.uint64(int(op[2:], 16) if op.startswith("0f") else int(op, 0))

    def f32(v):
        return np.asarray(v, np.uint64).astype(np.uint32).view(np.float32)

    for ins in lines:
        op, (d, *src) = ins.opcode, ins.operands
        if op == "mov.b64" and d.startswith("{"):
            lo, hi = d[1:-1].split(", ")
            regs[lo], regs[hi] = val(src[0]) & m32, val(src[0]) >> np.uint64(32)
            continue
        if op == "mov.b64":
            lo, hi = src[0][1:-1].split(", ")
            regs[d] = (val(hi) << np.uint64(32)) | val(lo)
            continue
        a = [val(s) for s in src]
        if op == "shf.r.wrap.b32":
            r = (((a[1] << np.uint64(32)) | a[0]) >> (a[2] & np.uint64(31))) & m32
        elif op == "shf.l.wrap.b32":
            n = a[2] & np.uint64(31)
            r = ((a[1] << n) | (a[0] >> (np.uint64(32) - n))) & m32
        elif op == "lop3.b32":
            r = np.uint64(0)
            for i in range(8):
                if a[3] >> np.uint64(i) & np.uint64(1):
                    r = r | ((a[0] if i & 4 else ~a[0]) & (a[1] if i & 2 else ~a[1]) & (a[2] if i & 1 else ~a[2]))
            r = r & m32
        elif op == "mul.rn.f32":
            r = (f32(a[0]) * f32(a[1])).view(np.uint32).astype(np.uint64)
        else:
            r = {"and.b32": lambda: a[0] & a[1], "or.b32": lambda: a[0] | a[1], "add.u32": lambda: (a[0] + a[1]) & m32,
                 "max.u32": lambda: np.maximum(a[0], a[1]), "min.u32": lambda: np.minimum(a[0], a[1]),
                 "shr.u32": lambda: a[0] >> a[1], "setp.lt.u32": lambda: a[0] < a[1],
                 "setp.gt.u64": lambda: a[0] > a[1], "selp.b32": lambda: np.where(a[2], a[0], a[1])}[op]()
        regs[d] = r
    return regs


class CommonSubexpressions(unittest.TestCase):
    def _cse(self, *lines):
        out = eliminate_common_subexpressions([Instr(*line) if isinstance(line, tuple) else line for line in lines])
        return [(i.opcode, i.operands) if isinstance(i, Instr) else i for i in out]

    def test_a_recomputation_reads_the_first_result(self):
        out = self._cse(("rcp.rn.f32", ["%f2", "%f1"]), ("st.global.f32", ["[%rd1]", "%f2"]),
                        ("rcp.rn.f32", ["%f3", "%f1"]), ("st.global.f32", ["[%rd1+4]", "%f3"]))
        self.assertEqual(out, [("rcp.rn.f32", ["%f2", "%f1"]), ("st.global.f32", ["[%rd1]", "%f2"]),
                               ("st.global.f32", ["[%rd1+4]", "%f2"])])

    def test_an_f64_round_trip_of_an_f32_is_the_f32(self):
        out = self._cse(("ex2.approx.ftz.f32", ["%f2", "%f1"]), ("cvt.f64.f32", ["%fd1", "%f2"]),
                        ("cvt.rn.f32.f64", ["%f3", "%fd1"]), ("add.rn.f32", ["%f4", "%f3", "%f3"]))
        self.assertEqual(out, [("ex2.approx.ftz.f32", ["%f2", "%f1"]), ("cvt.f64.f32", ["%fd1", "%f2"]),
                               ("add.rn.f32", ["%f4", "%f2", "%f2"])])

    def test_kept_when_a_source_changes_or_a_block_ends(self):
        changed = (("rcp.rn.f32", ["%f2", "%f1"]), ("add.f32", ["%f1", "%f1", "%f4"]), ("rcp.rn.f32", ["%f3", "%f1"]))
        labelled = (("rcp.rn.f32", ["%f2", "%f1"]), "$L1_loop:", ("rcp.rn.f32", ["%f3", "%f1"]))
        twice = (("rcp.rn.f32", ["%f2", "%f1"]), ("rcp.rn.f32", ["%f3", "%f1"]), ("mov.f32", ["%f3", "%f5"]))
        for lines in (changed, labelled, twice):
            with self.subTest(lines=lines):
                self.assertEqual(len(self._cse(*lines)), len(lines))


if __name__ == "__main__":
    unittest.main()
