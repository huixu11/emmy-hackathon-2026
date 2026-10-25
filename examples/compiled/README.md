# Compiled examples

Each pair is a kernel's Kernel IR (`.ir.txt`, the pretty-printed form of the `.ir.json` input) and
the PTX `compile.py` makes of it. Every IR statement appears as a `//` comment above the PTX
that implements it, so the two read side by side. Regenerate with `python tools/make_examples.py`.

| example | from | what to look for |
|---|---|---|
| `1_row_sum` | `problems/r04_row_sum` | The strided loop is gone: the facts prove 4 trips per thread, so the body is copied 4 times with offsets in the address (`[%rd5+512]`). Then a 5-step `shfl.sync.bfly` butterfly, the warp totals through shared memory, and a 2-level tree. `.maxntid 128` is the launch's block size. |
| `2_matmul_cp_async` | `problems/m01_matmul_512` | A ring of 3 tiles filled by `cp.async.cg` (16 B, bypassing L1) into a 72 KB dynamic pool, `cp.async.wait_group 2` and `bar.sync` before each use; the inner product reads 4 values per `ld.shared.v4.f32` (vectorize.py) into `fma.rn.f32`. |
| `3_attention` | `problems/a02_attention_128` | q·k, online softmax and ·v in one kernel: each key's row of k arrives by `ld.global.nc.v4.f32`, the dot product is a chain of `fma.rn.f32`, `exp` is our own (`ex2.approx.ftz` plus a correction), and `acc * alpha + v * beta` is one fma. |
| `4_tensor_core_matmul` | `tests/problems/t_m_f16_256x256x256` | f16 on tensor cores: `cp.async` into B128- and B32-swizzled shared tiles (the `bfe`/`shl`/`xor` triples are the swizzle), `ldmatrix.sync.aligned.m8n8.x4` / `.x2.trans` loads, 64 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, and each lane's f32 C values rounded by `cvt.rn.f16.f32` and stored in pairs (`st.global.v2.b16`). |
| `5_f16_softmax` | `tests/problems/d_f_f16_softmax` | 16-bit floats: `ld.global.nc.b16` + `cvt.f32.f16` on the way in, f32 arithmetic with one `cvt.rn.f16.f32` rounding where the IR's dtype is f16, and the f32 running sum the IR asks for. |
| `6_integer_floor_divide` | `tests/problems/d_p_i32_floordiv` | Integers: `x // 2` with torch's rounding toward minus infinity is `shr.s32` (an arithmetic shift); by other divisors, `div`/`rem` plus a sign fix (dtypes.py). |
