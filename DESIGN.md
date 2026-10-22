# ptxc: Kernel IR to PTX

`compile.py` reads one Emmy Kernel IR file and prints a PTX module:

```bash
python compile.py problems/r04_row_sum/k_sum_1_reduce.ir.json > out.ptx
python tools/run_problems.py                     # the 15 open problems on the GPU runner
python tools/run_problems.py --dir tests/problems   # 52 extra hidden-set-style problems
```

Results on the runner (RTX 5090): **15/15 open problems**, all five categories, and
**52/52** of the extra problems in `tests/problems`. Pointwise outputs match numpy bit for bit.

The translation is ours end to end. Emmy is used only to deserialize the IR (`Graph.from_dict`);
nothing goes through Emmy's renderer, `nvcc`, or another compiler.

## Pipeline

```
kernel.py   load      IR json -> typed KernelOp body; buffers (shape, dtype); meta.json launch
lower.py    plan      lay out every shared array (static symbols, or one dynamic pool > 48 KiB)
            prologue  ld.param + cvta.to.global for each parameter, in meta.json's arg_order
            walk      one visit_<Statement> per IR statement, emitting PTX as it goes
ptx.py      render    .version/.target, shared declarations, .entry signature, .reg counts, body
```

| module | role |
|---|---|
| `kernel.py` | Load the IR, the buffer table and the launch contract (`arg_order`, grid x block, `smem_bytes`). |
| `lower.py` | Translate statements: launch geometry, loops, memory, barriers, cross-thread combines. |
| `index.py` | Compile integer index expressions, predicates and addresses. |
| `fmath.py` | Emit f32 elementwise ops, from `add.rn` up to `tanh`, `erf`, `log` and `pow`. |
| `scope.py` | Track IR-name scopes and the cache of already computed values. |
| `ptx.py` | Hand out virtual registers and labels, format instructions, lay out the module. |

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
* `/`, `//` and `%` fold when both sides are constant, otherwise emit `div`/`rem`;
* comparisons fold or emit `setp`, `&&`/`||` short-circuit on constants, `?:` becomes `selp`.

Keeping the constant part separate lets it reach the **address immediate**. The unrolled
loads `x[a0*4 + 0..3]` share one address register and differ only in the offset:

```
	ld.global.nc.f32	%f1, [%rd9];
	ld.global.nc.f32	%f5, [%rd9+4];
	ld.global.nc.f32	%f8, [%rd9+8];
```

Every pure integer instruction goes through `IndexCompiler.derive`, which memoizes it in the
current scope. A repeated `a0 * 4` or address computation reuses the earlier register. The
cache is scoped with the names, so a value computed inside a loop or branch is never reused
after that body ends, where its register might not hold the right value. Loads from
read-only inputs are cached the same way. A kernel never writes its inputs, so the four
`load mul_c1[0]` of an unrolled body become one load, through the non-coherent cache
(`ld.global.nc`).

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

`StridedLoop` and `Loop` become bottom-tested loops: one entry test (folded away when the
bounds prove it), then one compare and branch per trip.

A loop with `seed` first sets each accumulator it folds to the fold's **identity**, taken from
the op itself: 0 for add, 1 for multiply, and `-1e30` for maximum. Emmy uses `-1e30` rather than
`-inf`, which matters for online softmax: `exp(-1e30 - m)` is 0, while `-inf - (-inf)` would be
NaN. Accumulators of nested loops are seeded by those loops, at each outer trip.
`Accum(base=...)` folds into `base` rather than the accumulator itself: `acc = base + value`.

A loop the IR marks `unroll` is unrolled **at compile time** when its bounds are constant, it
has no inner loop, and it stays under 1024 statements. Each copy binds the counter to a
constant, so its index math folds away. In m01's inner product the shared-memory reads become
`[%r54+24576]`, `[%r54+24580]`, and so on.

A `multiply` whose only use is the very next `acc <- add(acc, p)` becomes one `fma.rn.f32`.
That is what nvcc makes of the reference's `acc += a * b`. Pointwise code keeps separately
rounded `mul.rn` and `add.rn`, as numpy rounds them.

## Memory

| IR | PTX |
|---|---|
| `Load` of an input | `ld.global.nc.f32`, or `.v2`/`.v4` when the IR groups 2 or 4 names |
| `Load` of an output or shared array | `ld.global.f32` / `ld.shared.f32` |
| `Write` | `st.global` / `st.shared`, `.v2`/`.v4` for a grouped write |
| `Write (atomic)` | one `red.global.add.f32` per value |
| `Smem` up to 48 KiB in total | one `.shared .align A .b8 sh_name[bytes]` per array |
| `Smem` beyond 48 KiB | `.extern .shared` pool, arrays packed in declaration order (meta.json's `smem_bytes`) |
| `CpAsyncCopy` | `cp.async.cg.shared.global` (16 B) or `cp.async.ca` (4/8 B), plus the lane's stripe offset |
| `CpAsyncCommit` / `CpAsyncWait` | `cp.async.commit_group` / `cp.async.wait_group N` |

Global addresses are `base + mul.wide.s32(index, 4)`. Shared addresses are 32-bit:
`mad.lo.s32 index, 4, sh_array`.

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
branch around the body, and around the `else` when there is one.

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
| `exp` | `x*log2(e)` split into hi + lo (fma recovers the rounding error), `ex2.approx(hi) * (1 + lo*ln2)` | < 4e-7 over [-87, 88] |
| `tanh` | `x + x^3 P(x^2)` below 0.55 (polynomial fitted for relative error), `1 - 2/(e^2|x| + 1)` above | < 4e-7 |
| `erf` | `x P(x^2)` below 1, `1 - exp(-x^2) R(|x|)` on [1, 4], then +-1 | < 4e-7 |
| `log` | fdlibm's reduction to [sqrt(2)/2, sqrt(2)) and its coefficients | < 4e-7 |
| `pow` | repeated squaring for integral \|y\| <= 64, else `exp(y log|x|)`; C `powf`'s sign/NaN/zero rules | < 4e-7 integral, ~6e-7 fractional |
| `sigmoid`, `silu` | compositions of the above in numpy's operation order | < 4e-7 |
| `gelu`, `gelu_tanh` | the same; the formula's `1 + erf` cancels in the far negative tail | abs. error < 5e-7 |
| comparisons, `where`, `&&`/`\|\|` | 1.0/0.0 masks: `set.<cmp>.f32.f32`, `selp` | exact |

For scale, one f32 ulp is 6e-8 to 1.2e-7 of the value, so 4e-7 is 3 to 7 ulp.

The polynomial coefficients for `tanh` and `erf` were fitted by weighted least squares toward
minimax relative error. Each was checked in float32 Horner arithmetic before use, and then on
the GPU over its whole input range against float64.

## Testing

* `tools/run_problems.py` runs `submit.py`'s exact check (compile with `compile.py`, assemble,
  run under meta.json's launch, compare with `expected.npz`) over many problems.
* `tests/problems/` holds 52 problems generated by `tools/generate.py` from
  `tests/expressions_variants.py`: new shapes and ops in all five categories. They include odd
  sizes, broadcasts, every traceable elementwise op, unpinned (multi-block, atomic) and column
  reductions, batched, skinny and biased matmuls, layer norm, SwiGLU, SDPA and rectangular
  attention.

## Limits

* f32 values and buffers only, which is what every problem uses. Other dtypes stop with a
  clear error.
* The statements marked `*` in TASK.md (tensor-core fragments, TMA, mbarriers, carried
  state) are not implemented; no problem uses them. An unknown statement or op stops the
  compiler with a message on stderr and a nonzero exit, and never produces wrong PTX silently.
