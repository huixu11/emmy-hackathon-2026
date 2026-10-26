> **This submission: `ptxc`, a Kernel IR -> PTX compiler.** The organizers' task description
> follows this box.
>
> **In one minute**
>
> ```
> kernel.ir.json -> kernel.py -> lower.py ----------------------------> cse.py -> vectorize.py -> PTX
>   + meta.json     load the IR   one visit_ per IR statement            drop       merge loads
>                                 loops.py       each loop's shape        repeated   into .v2/.v4
>                                 index.py       addresses                rcp/div/
>                                 facts.py       ranges and divisors      sqrt
>                                 fmath.py       exp, tanh, sin, pow ...
>                                 dtypes.py      f16, bf16, f64, ints
>                                 tensorcore.py  mma.sync, ldmatrix
> ```
>
> 1. `compile.py` reads one Kernel IR file and walks it once, top to bottom. Every IR statement
>    kind has one `visit_` method that writes its PTX, and each IR line goes into the PTX as a
>    comment above the instructions that implement it.
> 2. It works out what it can before the kernel runs. Every integer's range and divisors give
>    loop trip counts (the loop unrolls), alignment (four loads become one `.v4`) and tests that
>    always hold (they are dropped).
> 3. The same parts build all five categories: nothing is special-cased per problem.
>
> | | |
> |---|---|
> | correct | 15/15 open problems, 308/308 and 23/23 generated ones, on the GPU runner |
> | fast kernels | 0.84x the time of Emmy's own CUDA over the f32 problems, 0.54x on attention; none of the 312 problems with a reference time is more than 3% slower |
> | fast compiler | milliseconds: a median of 3 to 6 ms per open-problem kernel, 7 ms over all 358, 3.3 s at most; a run of `compile.py` is mostly Python starting and importing emmy, about 0.4 s |
> | broad | the 5 categories; f32, f16, bf16, f64, i32, i64 and bool; `cp.async` copies; tensor cores (`mma.sync`, `ldmatrix`) |
>
> **Run it**
>
> ```bash
> git clone https://github.com/huixu11/emmy-hackathon-2026 && cd emmy-hackathon-2026
> pip install emmy-ml==0.3.11 numpy requests
> python compile.py problems/r04_row_sum/k_sum_1_reduce.ir.json > r04.ptx   # IR in, PTX out
>
> python -m unittest discover -s tests      # offline: all 358 kernels compile, deterministically
> python tools/run_problems.py               # the 15 open problems on the GPU runner: 15/15
> python tools/run_problems.py --dir tests/problems   # 308 more, all categories and dtypes: 308/308
> python tools/run_packed.py                 # 23 bf16 / bool problems (packed, see below): 23/23
> python tools/run_problems.py --iters 100   # median of 100 launches, as DESIGN.md reports
> ```
>
> The offline tests also assemble every module with `ptxas` when it is on `PATH` or named by
> `$PTXAS` (`pip install nvidia-cuda-nvcc-cu12` provides one under
> `site-packages/nvidia/cuda_nvcc/bin/`). Without `--iters`, the times shown are one cold launch.
>
> **What to look for**
>
> * **All five categories, one lowering.** `compile.py` + `ptxc/`; emmy only loads the IR
>   (`Graph.from_dict`). One `visit_<Statement>` per IR statement (`ptxc/lower.py`).
> * **Data types** (`ptxc/dtypes.py`): f32, f16, bf16, f64, i32, i64 and bool, in loads, stores,
>   arithmetic, reductions and conversions. 16-bit floats round once per op, as torch does;
>   integer `//` and `%` round toward minus infinity, as torch does (Emmy's own CUDA truncates).
>   Every open problem also runs with all its inputs in f16, bf16, f64 or i32, and every
>   attention and reduction problem in f16 and f64 (`c_*`). f64 is computed in f32 by default,
>   as Emmy's own CUDA computes it: sm_89 GPUs run f64 at 1/64 of the f32 rate, and its
>   `cvt.rn.f32.f64` too, so in loops three f64 reads in five are rounded to f32 in integer
>   instructions instead, to the same bits (subnormals, inf and NaN included).
>   `python compile.py --exact-f64 ...` (or `PTXC_EXACT_F64=1`) computes in f64 instead.
> * **Tensor cores** (`ptxc/tensorcore.py`): `mma.sync.aligned.m16n8k16` on f16 and bf16,
>   `ldmatrix` (x4, x2.trans, x4 and x4.trans pairs, x2), tiles staged by `cp.async` into
>   XOR-swizzled shared memory (B32, B64, B128, B128@k), fragments read from global memory with
>   row clamping and K zero-fill for odd shapes, fused epilogues (bias, relu) and split-K.
>   **f16 attention on tensor cores**: q @ k.T and p @ v as mma, the online softmax in between
>   on the C fragments (row max and sum by `shfl.sync.bfly` across the 4 lanes of a row, the
>   causal mask as `setp` + `selp` on each element's row and column, `cvt.rn.f16x2.f32` to turn
>   the f32 scores into the next mma's A operand).
> * **Copy engine:** `cp.async` (`.cg` 16 B and `.ca` 4/8 B), commit and wait groups, multi-stage
>   rings, shared pools over 48 KB. TMA and `wgmma` are not implemented: TMA needs a tensor map
>   built on the host, which the runner cannot pass, and `wgmma` needs sm_90a (the runner is an
>   RTX 5090, sm_120). An IR that uses them stops the compiler with a message.
> * **Optimizations**, each measured (DESIGN.md): trip-count unrolling, vector loads and stores,
>   division by constants without a divide, `.maxntid`, next-key prefetch and one fma per update
>   in attention, accumulators that always hold equal values computed once, scalar constants as
>   immediates, `pow` specialized to its exponent, a common-subexpression pass for what
>   ptxas will not merge (`rcp.rn`, `div.rn`, `sqrt.rn`), and f64 reads split between the f64
>   unit and the integer ALUs.
> * **Exact f32 math**: explicit `.rn` rounding, so pointwise results match numpy bit for bit; our
>   own `exp`, `log`, `tanh`, `erf`, `sin`/`cos`, `pow` with measured accuracy (DESIGN.md).
> * **Never wrong PTX silently**: anything unsupported stops with a one-line error and exit code 1.
>
> **Results.** 15/15 open problems, 308/308 in `tests/problems`, 23/23 in `tests/problems_packed`.
> Kernels take 0.84x the time of Emmy's own CUDA (the `.ref.cu`, compiled with NVRTC) over the 147 f32
> problems with a reference time, 0.54x on attention; 1.00x on the dtype problems and on the 9
> tensor-core matmuls, 0.82x on the 9 f16 attention problems, 0.79x on the attention and reduction
> problems in f16 and f64 (f64 attention 0.60x). None of the 312 problems with a reference time is
> more than 3% slower. Every problem is listed in [TIMINGS.md](TIMINGS.md).
>
> **Tests and examples**
>
> | where | what |
> |---|---|
> | `tests/test_compiler.py` | offline unit tests (`python -m unittest discover -s tests`) |
> | `tests/problems/` | 308 problems from `tests/expressions_variants.py` and `tests/expressions_cross.py` via `tools/generate.py`: new shapes, ops, dtypes (`d_*`), tensor-core matmuls (`t_m_*`), f16 attention (`t_s_*`), and the open problems in f16, f64 and i32 and the attention and reduction problems in f16 and f64 (`c_*`) |
> | `tests/problems_packed/` | bf16 and bool problems, the open problems in bf16 among them: the runner cannot allocate those buffers, so `tools/run_packed.py` passes them as int32 words |
> | `examples/compiled/` | seven kernels' IR next to our PTX, with notes on what to look for |
> | `tools/pow_sweep.py` | `pow` on 27 exponents and the special values, against C's `pow` |
>
> [DESIGN.md](DESIGN.md) explains how each IR statement becomes PTX, and why; [TIMINGS.md](TIMINGS.md)
> has every problem's time against the reference.

# Emmy Hackathon — write a PTX compiler for Kernel IR

Emmy (github.com/cloudrift-ai/emmy) compiles PyTorch to CUDA C++ and builds it with `nvcc`.
**Your task: write a compiler that takes Emmy's Kernel IR and emits PTX directly**, skipping the
CUDA C++ step. `ptxas` turns PTX into the same GPU binary ~19× faster than the full `nvcc`
pipeline — Emmy's autotuner compiles thousands of kernel variants, so that middle step matters.

You do **not** need a GPU. A hosted runner at `http://runner.riftstack.ai:48701` (the default in `submit.py`) compiles
your PTX with `ptxas`, executes it on real hardware, and tells you whether the output is correct. See [docs/API.md](docs/API.md).

## Requirements

- **Python 3.12 or newer.** The Kernel IR only deserializes through Emmy's own classes:

  ```bash
  pip install emmy-ml==0.3.11       # no torch, no CUDA, no GPU needed
  ```

  This alone reads every problem IR. Torch is **not** needed — it is only required to
  *generate* new IRs from PyTorch expressions, which is optional.

- Your submission is a **program**, not PTX files. It must run as:

  ```bash
  python compile.py path/to/kernel.ir.json > out.ptx
  ```

  This contract matters: submissions are evaluated against a **hidden set** of Kernel IRs in the
  same categories as the open problems. Hardcoding the open set scores nothing.

## Five-minute start

```bash
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026 && cd emmy-hackathon-2026
pip install emmy-ml==0.3.11 numpy requests   # no torch needed
python submit.py r01_array_sum examples/k_sum_1_reduce.ptx    # see a pass
python submit.py p00_scalar_add examples/k_sum_1_reduce.ptx   # wrong entry: a clean failure
python submit.py p00_scalar_add --compiler compile.py         # your compiler, once you have one
```

Then open `problems/p00_scalar_add/` and look at three files:

| file | what it is |
|---|---|
| `*.ir.json` | the Kernel IR — **the input to your compiler** |
| `*.ir.txt` | the same IR, pretty-printed for reading |
| `*.ref.cu` | Emmy's generated CUDA C++ — the reference answer, in a different target language |

Load an IR in Python:

```python
import json
from emmy.compiler.graph import Graph
g = Graph.from_dict(json.load(open("problems/r01_array_sum/k_sum_1_reduce.ir.json")))
body = g.nodes["sum_1"].op.body     # a tree of typed statements: Tile, StridedLoop, Load, ...
```

Your compiler walks that tree and prints PTX. Start with the common node kinds in the short table —
see [TASK.md](TASK.md) for the categories and the node-by-node translation table.

## The problems

15 open problems in 5 categories, easy → hard: **pointwise → reductions → matmul → fused →
attention-like**. Each is a folder under `problems/` with the IR, the reference, a launch
descriptor (`meta.json` — you never guess grid/block), and expected outputs (`expected.npz`).

**MVP: your compiler passes categories 1 and 2.** See [MVP.md](MVP.md).

Problems are scheduled for `sm_89`, and the runner is an **RTX 5090 (`sm_120`)**, where PTX written for `sm_89`
runs unchanged; [problems/PROVENANCE.md](problems/PROVENANCE.md)
says how they were generated and checked. The hidden evaluation set contains more problems of the *same kinds* —
different shapes and ops, no surprise concepts.

## Submitting

Submit at [riftstack.ai/hackathon/submit](https://riftstack.ai/hackathon/submit) before
**23:59 AoE on Tuesday, October 27**. A submission is:

- a **public GitHub repository** with `compile.py` at its root, runnable as
  `python compile.py path/to/kernel.ir.json > out.ptx`;
- a **video of 3 to 5 minutes** in which you walk through your compiler in your own voice
  (YouTube, Vimeo or Google Drive, shared so that anyone with the link can watch it).

Teams of up to three submit once and list every member. You can submit again until the
deadline; we review your latest entry. AI coding assistants are allowed, but your video has to
show that you understand how your compiler works — an entry where the understanding seems to
have been handed to an AI is not eligible for a prize.

## How compilers are evaluated

We run your `compile.py` on every kernel of the **hidden set** — the same five categories as the
open problems, with different shapes and ops — then assemble the PTX with `ptxas`, run it on the
runner's GPU under each problem's `meta.json` launch, and compare the outputs with the expected
ones within the problem's tolerance. This is the pipeline `submit.py --compiler` runs, so passing
the open set that way is the best preparation. Keep compiling to seconds per kernel: a time
limit may apply.

1. **Correctness on the hidden set, by category.** Passing categories 1 and 2 is the bar
   ([MVP.md](MVP.md)); each further category is what sets submissions apart.
2. **Code review** of your compiler is the first tiebreak. The IR → PTX lowering has to be your
   own — see the rule in [TASK.md](TASK.md).
3. **Speed** of the generated kernels is the last tiebreak.

The video decides prize eligibility, as described above.

## Repo map

```
problems/     the open set (15 problems)
examples/     hand-written PTX with comments mapping each block back to the IR
submit.py     one-command test: compile + run + check against expected outputs
docs/API.md   the runner API, if you want to call it directly
tools/        how the problems were generated (emmy compile -c ... --ir cuda)
TASK.md       the task, the categories, the CLI contract
MVP.md        the minimum bar
```
