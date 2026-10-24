# ptxc: Kernel IR to PTX

`compile.py` reads one Emmy Kernel IR file and prints a PTX module:

```bash
python compile.py problems/r04_row_sum/k_sum_1_reduce.ir.json > out.ptx
python tools/run_problems.py                     # the 15 open problems on the GPU runner
python tools/run_problems.py --dir tests/problems   # 134 extra hidden-set-style problems
```

Results on the runner (RTX 5090): **15/15 open problems**, all five categories, and
**134/134** of the extra problems in `tests/problems`. Pointwise outputs match numpy bit for bit.
Kernel speed is on par with Emmy's own CUDA path or better: a geometric mean of 0.85x its time
over the 147 problems that have a reference time (see [Performance](#performance)).

The translation is ours end to end. Emmy is used only to deserialize the IR (`Graph.from_dict`);
nothing goes through Emmy's renderer, `nvcc`, or another compiler.

## Pipeline

```
kernel.py   load      IR json -> typed KernelOp body; buffers (shape, dtype); meta.json launch
lower.py    plan      lay out every shared array (static symbols, or one dynamic pool > 48 KiB)
            prologue  ld.param + cvta.to.global for each parameter, in meta.json's arg_order
            walk      one visit_<Statement> per IR statement, emitting PTX as it goes
vectorize.py          merge runs of scalar loads into .v2/.v4 loads where alignment is proven
ptx.py      render    .version/.target, shared declarations, .entry signature, .reg counts, body
```

| module | role |
|---|---|
| `kernel.py` | Load the IR, the buffer table, scalar constants and the launch contract (`arg_order`, grid x block, `smem_bytes`). |
| `lower.py` | Translate statements: launch geometry, loops, memory, barriers, cross-thread combines. |
| `index.py` | Compile integer index expressions, predicates and addresses. |
| `facts.py` | Track a known range and divisor for every index register. |
| `fmath.py` | Emit f32 elementwise ops, from `add.rn` up to `tanh`, `erf`, `log` and `pow`. |
| `scope.py` | Track IR-name scopes, the cache of computed values, and where values can be hoisted to. |
| `vectorize.py` | Merge scalar loads into vector loads after lowering. |
| `ptx.py` | Hand out virtual registers and labels, keep instructions structured, lay out the module. |

Every statement the compiler lowers is written into the PTX as a comment, so the output reads
side by side with the IR:

```
	// for a1 in a1_co..512:128
	mov.f32	%f0, 0f00000000;
	...
$L1_loop:
	// in0 = load x[a0, a1]
	...
	ld.global.nc.f32	%f1, [%rd5];
	// f32 acc0 <- add(acc0, in0)
	add.rn.f32	%f0, %f0, %f1;
```

## Values and names

An IR name maps to one of two things:

* an **integer** (axis coordinate, loop counter, integer `Let`): a `Lin(reg, const)`, the value
  `reg + const`, where `reg` is a `.b32` register or `None` for a compile-time constant;
* an **f32** value: its PTX operand, a register (`%f3`) or an immediate (`0f3F800000` for 1.0).

Registers are virtual and never reused: every result gets a fresh one and ptxas does the real
allocation. That keeps the code generator free of liveness analysis.

Names are **scoped like C blocks** (`scope.py`). Each loop body, branch and butterfly step opens
a scope. A name first defined inside a scope disappears when it closes. Assigning a name that
already exists writes its register in place. That covers accumulators (`acc0 <- add(...)` inside
a loop, read after it) and reduction states (`acc0 = add(acc0, acc0__o)` in a combine program),
the two places the SSA-looking IR really mutates.

## Index expressions (`index.py`)

Index expressions are small trees of `Var`, `Literal`, `BinaryExpr` and `TernaryExpr`. They
compile recursively to `Lin` values, with constant folding:

* `+` adds the constants and emits `add.s32` only when both sides have registers;
* `*` by a constant distributes over the constant, `(r + c) * k = r*k + c*k`;
* `/`, `//` and `%` fold when both sides are constant. By a constant `k` with the dividend
  known non-negative, they need no divide. A power of two becomes `shr` or `and`. Any other
  `k` becomes a multiply-high by a magic reciprocal (Granlund-Montgomery):
  `a / k = mulhi(a, ceil(2^(31+l) / k)) >> (l - 1)`, with `l = ceil(log2 k)`, exact for
  `0 <= a < 2^31`. This matters because ptxas turns `div.u32 x, 128` into a full
  reciprocal-based division;
* comparisons fold (also when the operands' ranges settle them) or emit `setp`, `&&`/`||`
  short-circuit on constants, `?:` becomes `selp`.

Keeping the constant part separate lets it reach the **address immediate**. The unrolled
loads `x[a0*4 + 0..3]` share one address register and differ only in the offset:

```
	ld.global.nc.f32	%f1, [%rd9];
	ld.global.nc.f32	%f5, [%rd9+4];
	ld.global.nc.f32	%f8, [%rd9+8];
```

Every pure integer instruction goes through `IndexCompiler.derive`, which memoizes it. A
repeated `a0 * 4` or address computation reuses the earlier register. The cache is scoped with
the names, so a value computed inside a loop or branch is never reused after that body ends,
where its register might not hold the right value. Loads from read-only inputs are cached the
same way. A kernel never writes its inputs, so the four `load mul_c1[0]` of an unrolled body
become one load, through the non-coherent cache (`ld.global.nc`).

`derive` also **hoists**. An instruction whose operands all come from enclosing scopes is
placed at the outermost point those operands allow. That point is a loop's preheader (ahead of
its entry test) or the spot before a branch, both of which come before every later
instruction of the enclosing code. The result is cached there, so loop-invariant address
arithmetic runs once instead of once per trip or per unrolled copy.

**Facts** (`facts.py`). Each derived register records a range and a known divisor. They start
from the launch (`%tid.x < block`, a Tile coordinate `< extent`, a loop counter is
`start + j*step`) and flow through the arithmetic:

```
a0_co in [0, 127]   ->   a0_co * 4 in [0, 508], a multiple of 4
```

The lowering uses them in three places: to count loop trips, to fold comparisons, and to prove
addresses aligned for vector loads. Every rule is exact or gives up.

Multi-dimensional indices flatten row-major against the buffer's shape from the graph:
`x[a0, a1]` into `(64, 512)` is `a0*512 + a1`.

## Launch geometry: `Tile`

A `Tile` maps one thread to one cell of its iteration space. The global thread id is
`ctaid.x * ntid.x + tid.x`, decoded row-major with the last axis fastest. The first axis needs
no modulo. When the tile rasterizes a 2-D block grid (`raster_group`), the block pair is decoded
with the same grouped swizzle Emmy schedules, so CTAs stay where the IR expects them.

The bounds guard is not in the IR. When the launch carries more threads than the tile has
cells (`p00`: 64 cells, 256 threads), the extra threads branch to the exit. `lane`
(`tid & 31`) and `warp` (`tid >> 5`) are defined only if the body uses them.

## Loops and accumulators

A loop's shape comes from its **trip count**. A `StridedLoop`'s start is usually a thread
coordinate whose range is known, and that often makes every thread's count known.
`r01`'s `for a0 in a0_co..1024:128`, for example, runs exactly 8 trips on every thread.

* **The same small count on every thread:** the body is emitted once per trip with no loop at
  all. Each copy binds the loop variable to `start + k*step`, so index math folds into
  address immediates and all the copies' loads can be in flight together.
* **The same large count:** a real loop that runs 8, 4 or 2 trips per iteration, with the
  factor dividing the count, so no remainder loop is needed (`r02`: 512 trips, 8 per iteration).
* **At most one trip:** a branch (`m02`'s `for _fa in tid..32:256`).
* **Otherwise:** 4 trips per iteration while they fit, then single trips.

A loop with a constant start gives ptxas a constant trip count. Measurements showed that inside
another loop ptxas does best on its own: it unrolls the loop and hoists its invariant loads
out of the outer one. So those loops stay rolled, unless they read shared memory, because
ptxas never merges shared loads across trips. A long loop that is not inside another loop is
unrolled here, as NVVM does for the reference, because ptxas may leave it rolled.

Attention kernels walk the keys in an outer loop around one inner product per query row, and
three more rules come from measuring them in both layouts the IR uses:

* **Inner loops that stream rows are unrolled here.** In SDPA, `k[..., key, d]` is read along
  `d` and changes with every key. ptxas cannot hoist those loads out of the key loop and keeps
  them scalar; unrolled here, `vectorize.py` turns them into `.v4` loads, a quarter of the
  instructions (`w_a_sdpa`: 53.9 -> 39.6 us; a 256-key SDPA: 472 -> 124 us).
* **A short key walk is unrolled whole.** With `q @ kt`, each key's column of `kt` is scattered
  and loads one float at a time. When the walk has at most 64 keys and fits in 16384 IR
  statements, the whole loop is unrolled, so ptxas can issue one key's loads while the
  previous key's softmax update still runs, as NVVM does (`v_a_32x64`: 25.3 -> 13.0 us). An
  SDPA walk stays rolled: its trips already load in vectors, and unrolled it ran up to 1.9x
  slower.
* **A walk that stays rolled prefetches.** A trip cannot start its dot products until its
  key's row of k (or column of kt) and of v arrive. So each trip first issues the next key's
  loads into a second set of registers, then computes with the values the trip before
  loaded, and finally moves the new ones over (the next key is clamped to the last, so the
  final trip's loads stay in bounds). The loads arrive while the previous key's dot products
  and softmax update run. ptxas cannot undo this the way it re-sinks loads moved up inside
  a trip: the values cross the loop's back edge. It applies to a top-level loop with
  constant bounds whose inner loops are all unrolled here, so that every load site reads one
  fixed address per trip. Such a loop unrolls its inner loops whatever their size: with the
  loads a trip ahead, the stalls that made ptxas's own schedule of them better are gone. Over
  the 24 attention kernels this changes, time fell to 0.36x to 0.97x (`a01`: 76.4 -> 43.7 us
  against the reference's 74.4; `y_s_1x2x48x64`: 43.7 -> 19.1 us against 25.2).

Real loops are bottom-tested: one entry test (folded away when the bounds prove it), then
one compare and branch per iteration. The trips always run in order, so every accumulator
folds exactly as the IR says.

A loop with `seed` first sets each accumulator it folds to the fold's **identity**, taken from
the op itself: 0 for add, 1 for multiply, and `-1e30` for maximum. Emmy uses `-1e30` rather than
`-inf`, which matters for online softmax: `exp(-1e30 - m)` is 0, while `-inf - (-inf)` would be
NaN. Accumulators of nested loops are seeded by those loops, at each outer trip.
`Accum(base=...)` folds into `base` rather than the accumulator itself: `acc = base + value`.

**Accumulators that always hold equal values are computed once.** Emmy repeats online
softmax's running max and sum for every output column a thread owns: `acc1__c0_0` and
`acc1__c0_1` are both row 0's max, updated by the same `max` of the same score on every key,
so `a01` computes 16 exps per key where 8 give the same answer. Neither ptxas nor NVVM merges
values that are carried around a loop. Before a seeding loop is emitted, its accumulators are
grouped by identity. The values of one trip are then numbered (`_value_numbers`), on the
assumption that the members of a group are equal when the trip starts, and a group whose
members come out with different values is split. When no group splits, the assumption holds
on every trip, by induction. The first member of a group is the only one computed. It gets
each trip's new value in a fresh register, because the others still read the old value until
their own update; their names then stand for its new value, and it moves back into the loop's
register at the end of the trip. The copies' identical score and exp arithmetic is left to
ptxas, which merges it within the trip. Results are the same bits, from half the exps. Over
the 19 attention kernels it changes, time is 0.91x (`v_a_rect`: 58.0 -> 33.4 us,
`y_s_1x1x256x64`: 111.3 -> 78.5 us, `a01`: 45.5 -> 41.6 us), with one exception:
`y_s_2x2x64x32`, which reloads 8 query rows from L1 on every key, went 45.8 -> 64.2 us
(the reference: 78.5 us).

A `multiply` whose only use is the very next `acc <- add(base, p)` (base is the accumulator
itself when the IR names none) becomes one `fma.rn.f32`. That is what nvcc makes of the
reference's `acc += a * b`, and of online softmax's `acc * alpha + v * beta`; the second
shortens the chain that carries each key's update to the next (`v_a_rect`: 62.1 -> 41.6 us;
over the 38 kernels it changes, 0.975x the time, though `a01` went 43.7 -> 45.8 us as
ptxas scheduled the shorter code differently).
A product by 1.0 (softmax's `one * beta`) is an exact add. Pointwise code keeps separately
rounded `mul.rn` and `add.rn`, as numpy rounds them.

## Memory

| IR | PTX |
|---|---|
| `Load` of an input | `ld.global.nc.f32`, or `.v2`/`.v4` when the IR groups 2 or 4 names |
| `Load` of a scalar constant | no load: the immediate, as for a `Let` (below) |
| `Load` of an output or shared array | `ld.global.f32` / `ld.shared.f32` |
| `Write` | `st.global` / `st.shared`, `.v2`/`.v4` for a grouped write |
| `Write (atomic)` | one `red.global.add.f32` per value |
| `Smem` up to 48 KiB in total | one `.shared .align A .b8 sh_name[bytes]` per array |
| `Smem` beyond 48 KiB | `.extern .shared` pool, arrays packed in declaration order (meta.json's `smem_bytes`) |
| `CpAsyncCopy` | `cp.async.cg.shared.global` (16 B) or `cp.async.ca` (4/8 B), plus the lane's stripe offset |
| `CpAsyncCommit` / `CpAsyncWait` | `cp.async.commit_group` / `cp.async.wait_group N` |

Global addresses are `base + mul.wide.s32(index, 4)`. Shared addresses are 32-bit:
`mad.lo.s32 index, 4, sh_array`.

Scalars of the torch expression (an epsilon, a scale, `mean`'s count, the 1.5 of `x ** 1.5`)
reach the kernel as one-element buffers that the graph fills from a `ConstantOp`. The launch
still passes them, but their value is in the IR, so a `Load` of one becomes the immediate
(`0f3FC00000`) instead of a memory read. Values can then fold: `reciprocal` of a count is
done by ptxas, and `pow` picks its method from the exponent (under f32 math). A value that
meta.json gives differently from the IR is left to be read at run time.

After lowering, `vectorize.py` merges runs of scalar loads through one base register at
consecutive offsets into `.v2`/`.v4` loads. m01's unrolled inner product reads one column of
A per trip; four trips become one `ld.shared.v4.f32`. The base register's divisor from the
facts, together with the offset, must prove the address aligned to the vector's size. Moving a
later load up is safe because nothing in between may write that memory (a store, a
`cp.async`, a barrier, a branch or a label ends the region) or touch its destination register.
m01 goes from 256 scalar shared loads to 96 vector ones, the same count as NVVM's code for the
reference.

## Cross-thread combines

**`WarpShuffle`** is an XOR butterfly over `length` lanes. At distance 16, 8, 4, 2, 1 each lane
fetches its partner's states (`shfl.sync.bfly.b32` under `activemask`) and runs the IR's
combine program on them. The combine is ordinary `Assign`s, so a plain sum and online
softmax's coupled (max, sum) pair go through the same code: the IR says how to combine and
the compiler only routes the values.

**`TreeHalve`** folds one partial per warp through shared memory. At stride
`s = length/2, ..., 1`, entry `i < s` combines entry `i + s` into itself, followed by a
barrier. Every thread then reads entry 0. The combine program is the same as the shuffle's.

`Sync` is `bar.sync id` (or `bar.sync id, count`, or `bar.warp.sync`). `Cond` is `setp` plus a
branch around the body, and around the `else` when there is one. `Select` (a value chosen by a
coordinate predicate: causal masks, the tail of a split reduction) is a chain of `selp`, one per
branch whose predicate the facts leave open; a settled one costs nothing.

## f32 math (`fmath.py`)

Correctness means matching numpy within the problem's tolerance, and the pointwise problems use
`rtol = atol = 0`. So arithmetic uses explicit IEEE rounding: `add.rn`, `mul.rn`, `div.rn`,
`rcp.rn`, `sqrt.rn`. Without the `.rn`, ptxas may contract `a*b + c` into one fma and round
once where numpy rounds twice. `rsqrt` is `rcp.rn(sqrt.rn(x))`, which is exactly what numpy
computes.

PTX has only approximate 2^x and log2 (`ex2.approx`, `lg2.approx`), so the transcendental ops
are built from them, from polynomials, or both:

The last column is the largest relative error measured on the GPU against float64, over a
sweep of each op's input range.

| op | method | max relative error |
|---|---|---|
| `exp` | x clamped to [-104, 89] (NaN passes), `x*log2(e)` split into hi + lo (fma recovers the rounding error), `ex2.approx.ftz(hi) * (1 + lo*ln2)` | < 4e-7 over [-87, 88] |
| `tanh` | `x + x^3 P(x^2)` below 0.55 (polynomial fitted for relative error), `1 - 2/(e^2|x| + 1)` above | < 4e-7 |
| `erf` | `x P(x^2)` below 1, `1 - exp(-x^2) R(|x|)` on [1, 4], then +-1 | < 4e-7 |
| `log` | fdlibm's reduction to [sqrt(2)/2, sqrt(2)) and its coefficients | < 4e-7 |
| `pow` | repeated squaring for integral \|y\| <= 64, `x^k sqrt(x)` for y = k + 1/2, else `exp(y log|x|)`; C `powf`'s sign/NaN/zero/inf rules | < 2.6e-7 for integral \|y\| <= 7 and half-integral \|y\| <= 2.5; 2.6e-6 at y = 64; through exp and log (fractional y, integers past 64) 1.4e-6 to 8.8e-6, growing with y log x; x over [6e-6, 1.6e5] |
| `sigmoid`, `silu` | compositions of the above in numpy's operation order | < 4e-7 |
| `gelu`, `gelu_tanh` | the same; the formula's `1 + erf` cancels in the far negative tail | abs. error < 5e-7 |
| `sin`, `cos` | cephes' `sinf`: reduce by the nearest even multiple of pi/4 (pi/4 in three f32 parts, one fma each), then a polynomial for sin or cos picked by the multiple, which also sets the sign | < 2.4e-7 up to \|x\| = 1e4 |
| comparisons, `where`, `&&`/`\|\|` | 1.0/0.0 masks: `set.<cmp>.f32.f32`, `selp` | exact |

For scale, one f32 ulp is 6e-8 to 1.2e-7 of the value, so 4e-7 is 3 to 7 ulp.

`pow`'s exponent is nearly always a constant of the graph (`x ** 3`, `x ** 1.5`). Then only
the method for that exponent is emitted, with its fix-ups for negative x, zeros and infinities.
`x ** 3` is two multiplies, and `x ** 1.5` is `x * sqrt(x)`, within 1 ulp where
`exp(1.5 log x)` was off by up to 24. About 8 PTX instructions per element replace 100: a
run-time exponent computes every method and selects one. On 2^18 elements `x ** 3` and
`x ** 2.2` take 3.0 us against the reference's 4.8 us.

These are all the f32 ops Emmy's own CUDA renderer accepts (`op_to_expr`), so every
elementwise op a traced problem can contain is covered.

The polynomial coefficients for `tanh` and `erf` were fitted by weighted least squares toward
minimax relative error. Each was checked in float32 Horner arithmetic before use, and then on
the GPU over its whole input range against float64.

## Performance

Speed is measured with `tools/run_problems.py --iters 100`, the median of 100 launches after a
warmup. The baseline is Emmy's reference CUDA (`*.ref.cu`) for the same problem, under the same
launch, compiled to PTX with NVRTC. The NVRTC build was used only for this comparison and is
not part of the compiler.

| problem | first version | now | reference |
|---|---|---|---|
| m01_matmul_512 | 29.4 us | 15.3 us | 15.0 us |
| a01_attention | | 45.6 us | 74.4 us |
| a02_attention_128 | 72.4 us | 29.3 us | 29.3 us |
| r02_big_sum | 21.2 us | 8.9 us | 10.6 us |
| f01_rmsnorm | | 3.3 us | 4.8 us |
| w_f_gelu_mlp | 27.3 us | 10.9 us | 17.1 us |
| w_a_512x64 | 223.8 us | 178.8 us | 354.8 us |
| w_a_sdpa | | 21.2 us | 49.8 us |
| v_a_32x64 | | 13.0 us | 23.2 us |
| v_a_rect | | 58.0 us | 60.1 us |
| y_a_16x64x512x64 | | 201.3 us | 434.7 us |
| z_p_pow_frac_big | | 3.1 us | 4.8 us |
| geometric mean of time / reference, the first 66 problems | 1.12x | 0.91x | 1.00x |
| geometric mean of time / reference, all 147 with a reference | | 0.85x | 1.00x |

By category, over all 147: pointwise 0.99x, reductions 0.94x, matmul 0.95x, fused 0.88x,
attention 0.57x. The "now" and "reference" columns come from one run of both compilers, back
to back: the runner's speed drifts by about 20% between sessions, so figures from different
sessions do not compare.

Two problems have no reference time: NVRTC rejects `v_p_log`'s reference source, and the
reference gets `x_p_fma_5000` (`x * y + z`, checked with zero tolerance) wrong, because nvcc
contracts it into one fma.

What closed the gap, in the order it was found by comparing our SASS with the reference's:

1. Trip-count-driven unrolling (above). The reductions were latency-bound on a single
   load in flight.
2. Load vectorization, enabled by hoisting and by divisor facts on addresses and counters.
3. Division by constants without a divide (above).
4. `.maxntid <block>, 1, 1`: the launch's block size, which CUDA's `__launch_bounds__` gives
   nvcc, lets ptxas budget registers for the real block (a02: 43.7 -> 29.4 us).
5. A cheaper `exp`: clamping the input removes the special-case checks around the correction,
   and `.ftz` removes ptxas's subnormal handling around `ex2`. That is about half the SASS per
   exp, with the same accuracy.
6. The attention loop rules (under Loops): vector loads for streamed rows, short key walks
   unrolled whole, and prefetching the next key's loads in the walks that stay rolled.
7. One fma for online softmax's `acc * alpha + v * beta` (above).
8. Scalar constants as immediates, and `pow` specialized to its exponent (under Memory and
   f32 math). A load fewer on the critical path took `f01_rmsnorm` from 4.8 to 3.3 us.

No problem is slower than the reference beyond the noise. Kernels of about 3 us jump by
0.5 us now and then: `x_p_scalar_bcast` and `v_p_where` read 1.19x in this run, and 1.00x
when run again; `m01`, whose PTX is unchanged, read 15.0 us in the run before.

Attention kernels with few blocks (`v_a_rect` runs 4 blocks of 256 threads) are bound by the
latency of each key's trip, and ptxas's schedule of that trip moves their time by tens of
percent on a one-instruction change. Knowing the scale as an immediate instead of a loaded
value, with the same instruction count, took `v_a_rect` from 41.6 to 58.0 us and
`y_s_1x4x128x64` from 37.5 to 41.6 us, and `y_a_16x64x512x64` from 340.7 to 201.3 us. Over all
problems the geometric mean stayed at 0.85x.

## Testing

* `tools/run_problems.py` runs `submit.py`'s exact check (compile with `compile.py`, assemble,
  run under meta.json's launch, compare with `expected.npz`) over many problems.
* `tests/problems/` holds 134 problems generated by `tools/generate.py` from
  `tests/expressions_variants.py`: new shapes and ops in all five categories. They include odd
  sizes, 3-D and outer broadcasts, every renderable elementwise op (`sin`/`cos` included),
  composite activations (mish, elu, hardswish), unpinned (multi-block, atomic), column,
  middle-axis and 2^20-element reductions, a scan (`cumsum`), batched, skinny, split-K,
  transposed, `einsum` and `linear` matmuls, layer and RMS norms, SwiGLU, SDPA with batch and
  heads, causal SDPA, cross and long-KV attention, and 19 more attention shapes (`y_*`) with
  short and long key walks in both layouts.
* The third batch (`x_*`) found two gaps, both fixed: `sin`/`cos`, and the `Select` statement
  that small matmuls and causal attention use. A line `x = torch.randn(...) * 4` in a spec
  declares the input `x` only: the generator feeds it plain `randn` and drops the arithmetic,
  so specs keep all computation in the final expression.
* The fourth batch (`y_*`) holds the attention shapes the two loop rules above were worked out
  on: both layouts, 16 to 512 keys, 32 to 512 threads per block. Among them was a 256-key SDPA
  that ran 1.7x slower than the reference before the row-streaming rule.
* The fifth batch (`z_*`) takes powers by constants: `x ** 2`, `** -2`, `** -0.5`, `** 0.5`,
  `** 2.2` and `** 3` (the last three on 2^18 elements), the mean of `x ** 1.5`, and RMS and
  L2 norms written with `x.pow(2)`. `tools/pow_sweep.py` runs 27 exponents on the GPU, with
  zeros, infinities, NaN and subnormals among the inputs, and checks `pow`'s special cases
  against C's `pow`.

## Limits

* f32 values and buffers only, which is what every problem uses. Other dtypes stop with a
  clear error.
* The statements marked `*` in TASK.md (tensor-core fragments, TMA, mbarriers, carried
  state) are not implemented; no problem uses them. An unknown statement or op stops the
  compiler with a message on stderr and a nonzero exit, and never produces wrong PTX silently.
