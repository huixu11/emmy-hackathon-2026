"""Statement lowering: walk the Kernel IR body and emit PTX for every statement.

There is one `visit_<Statement>` method per IR statement kind. The walk tracks IR names in
`Scopes`: an integer name (an axis, a loop counter, an integer Let) maps to a `Lin`, an f32 name
maps to its operand, a register such as %f3 or an immediate such as 0f3F800000.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections import Counter
from collections.abc import Iterator

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import FragmentRowReduce, Smem, TreeHalve
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Select, SelectBranch, Write

from . import dtypes as dt
from .cse import eliminate_common_subexpressions
from .errors import CompileError
from .facts import Fact
from .fmath import ONE
from .index import ZERO, IndexCompiler, Lin
from .kernel import Kernel
from .ptx import Mark, PtxBuilder, address, is_register, render_module, vector
from .scope import Scopes
from .tensorcore import FRAGMENT_FIELDS, TensorCores
from .vectorize import vectorize_loads

_F32_BYTES = 4
_STATIC_SMEM_LIMIT = 48 * 1024  # larger shared memory must be the launch's dynamic allocation
_SMEM_POOL = "sh_pool"
_GLOBAL_ALIGN = 16  # buffer base alignment; the IR's own v4 accesses rely on at least this
# Unrolling limits, in IR statements. The enclosing budget was measured on attention kernels:
# when unrolling the inner loops would grow the loop around them past it (several 64-trip
# inner products per trip), ptxas does better with them rolled (it unrolls and hoists itself),
# unless they stream rows that only unrolling here turns into vector loads (_streams_rows), or
# the loop around them prefetches (_prefetch_plan).
_UNROLL_BUDGET = 1024           # a fully unrolled loop the IR marks unroll
_IMPLICIT_UNROLL_BUDGET = 512   # a fully unrolled loop the IR does not mark
_ENCLOSING_UNROLL_BUDGET = 384  # the body of a loop whose inner loops unroll without the IR asking
_FULL_UNROLL_TRIPS = 64         # most trips unrolled completely without the IR asking
_PARTIAL_UNROLL_BUDGET = 128    # one iteration of a partially unrolled loop
_OUTER_UNROLL_BUDGET = 16384    # a short top-level loop with inner loops, unrolled whole


@dataclasses.dataclass(frozen=True)
class SharedArray:
    """One Smem array: its own static symbol, or a byte offset into the dynamic pool."""

    symbol: str
    offset: int
    extents: tuple[int, ...]
    align: int
    dtype: dt.DType = dt.F32

    @property
    def nbytes(self) -> int:
        return math.prod(self.extents) * self.dtype.nbytes


class KernelLowering(TensorCores):
    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.b = PtxBuilder()
        self.scopes = Scopes()
        self.idx = IndexCompiler(self.b, self.scopes)
        self.shared: dict[str, SharedArray] = {}
        self.globals: dict[str, str] = {}  # buffer name -> register with its global address
        self.exit_label: str | None = None
        # Whether a loop may be unrolled without the IR asking; False inside a loop whose body
        # would grow past _ENCLOSING_UNROLL_BUDGET that way.
        self._unroll_room = True
        self._in_loop = False  # inside the body of a loop emitted as a real loop
        self._trips: list[tuple[str, int]] = []  # (variable, trip) of the unrolled copies being emitted
        # Load site -> registers that already hold its values (see _prefetch_plan).
        self._prefetched: dict[tuple, list[str]] | None = None
        # Accumulators computed once for the copies that always equal them
        # (_equal_accumulators): loop body -> {copy: the accumulator it equals}; and, while a
        # trip is emitted, its copies and the accumulators they follow.
        self._merged: dict[int, dict[str, str]] = {}
        self._copies: dict[str, str] = {}
        self._followed: set[str] = set()
        self._types: dict[str, dt.DType] = {}  # value name -> its dtype (f32 when not listed)
        self._frags: dict[str, list[list[str]]] = {}  # fragment array -> registers of each fragment
        # How often each name appears anywhere in the kernel, definitions included.
        self._name_counts = Counter(leaf for node in _walk(kernel.body)
                                    for leaf in _string_leaves(node))

    def run(self) -> str:
        """The complete PTX module for the kernel."""
        shared_decls = self._plan_shared()
        for name in self.kernel.params:
            generic, glob = self.b.reg("b64"), self.b.reg("b64")
            self.b.emit("ld.param.u64", generic, f"[{_param_symbol(name)}]")
            self.b.emit("cvta.to.global.u64", glob, generic)
            self.idx.assume(glob, Fact(div=_GLOBAL_ALIGN))
            self.idx.depth[glob] = 0  # defined in the prologue, ahead of every scope
            self.globals[name] = glob
        self._stmts(self.kernel.body)
        if self.exit_label:
            self.b.place(self.exit_label)
        self.b.emit("ret")
        self.b.lines = vectorize_loads(eliminate_common_subexpressions(self.b.flat()), self._alignment)
        return render_module(
            self.kernel.entry,
            [_param_symbol(name) for name in self.kernel.params],
            shared_decls,
            self.b,
            max_threads=self.kernel.block,
        )

    def _plan_shared(self) -> list[str]:
        """Lay out every Smem array of the kernel up front and return the module declarations.

        Up to 48 KiB, each array is its own static .shared symbol. Beyond that, static shared
        memory is not allowed: the arrays are packed, in declaration order, into the dynamic
        pool the launch allocates (meta.json's smem_bytes).
        """
        smems: dict[str, object] = {}
        for node in _walk(self.kernel.body):
            if isinstance(node, Smem):
                smems.setdefault(node.name, node)
        layout, end = [], 0
        for smem in smems.values():
            esize = dt.dtype(smem.dtype).nbytes
            align = max(smem.align, esize)
            offset = -(-end // align) * align
            layout.append((smem, offset, align))
            end = offset + math.prod(smem.extents) * esize

        if end <= _STATIC_SMEM_LIMIT:
            for smem, _, align in layout:
                self.shared[smem.name] = SharedArray(f"sh_{smem.name}", 0, tuple(smem.extents), align,
                                                     dt.dtype(smem.dtype))
                self.idx.assume(f"sh_{smem.name}", Fact(div=align))
            return [f".shared .align {a.align} .b8 {a.symbol}[{a.nbytes}];" for a in self.shared.values()]

        available = self.kernel.dynamic_smem_bytes
        if available is not None and available < end:
            raise CompileError(f"shared arrays need {end} bytes, the launch provides {available}")
        pool_align = max(align for _, _, align in layout)
        self.idx.assume(_SMEM_POOL, Fact(div=pool_align))
        for smem, offset, align in layout:
            self.shared[smem.name] = SharedArray(_SMEM_POOL, offset, tuple(smem.extents), align,
                                                 dt.dtype(smem.dtype))
        return [f".extern .shared .align {pool_align} .b8 {_SMEM_POOL}[];  // {end} bytes, dynamic"]

    def _alignment(self, operand: str) -> int:
        """The largest power of two an address operand is known to be a multiple of."""
        div = self.idx.facts.get(operand, Fact()).div
        return 1 << 30 if div == 0 else div & -div

    def _stmts(self, body, *, annotate: bool = True) -> None:
        stmts = list(body)
        i = 0
        while i < len(stmts):
            stmt, after = stmts[i], stmts[i + 1] if i + 1 < len(stmts) else None
            if annotate:
                self.b.comment(stmt.pretty()[0].strip())
            if self._is_fma_pair(stmt, after):
                if annotate:
                    self.b.comment(after.pretty()[0].strip())
                self._fma(stmt, after)
                i += 2
                continue
            visit = getattr(self, f"visit_{type(stmt).__name__}", None)
            if visit is None:
                raise CompileError(f"unsupported IR statement {type(stmt).__name__}")
            visit(stmt)
            i += 1

    def _is_fma_pair(self, stmt, after) -> bool:
        """`p = multiply(a, b)` directly followed by `acc <- add(base, p)` (base is acc itself
        when the IR names none), with p used nowhere else: one fma, a single rounding, as nvcc
        contracts the reference's `acc += a * b` and online softmax's `acc * alpha + v * beta`."""
        return (isinstance(stmt, Assign) and stmt.op.name == "multiply"
                and isinstance(after, Accum) and after.op.name == "add" and after.value == stmt.name
                and dt.dtype(stmt.dtype) == dt.dtype(after.dtype) == dt.F32
                and all(self._type(a) == dt.F32 for a in stmt.args)
                and self._name_counts[stmt.name] == 2)  # its definition and this one use

    def _fma(self, product, acc) -> None:
        a, c = (self._operand(name) for name in product.args)
        target = self._accumulator(acc)
        if target is None:
            return
        dst, base = target
        if ONE in (a, c):  # softmax's `one * beta`: exactly an add
            self.b.emit("add.rn.f32", dst, base, c if a == ONE else a)
        else:
            self.b.emit("fma.rn.f32", dst, a, c, base)

    # ---- launch geometry -------------------------------------------------------------------

    def visit_Tile(self, tile) -> None:
        """One thread per cell: decode the cell's axis coordinates from the global thread id."""
        if tile.aux_threads:
            raise CompileError("tiles with a warp-specialized aux band are not supported")
        extents = [axis.extent.as_static() for axis in tile.axes]
        cells = math.prod(extents)
        if self.kernel.block is not None:
            self.idx.assume("%tid.x", Fact(0, self.kernel.block - 1))
            self.idx.assume("%ntid.x", Fact.const(self.kernel.block))
            self.idx.assume("%ctaid.x", Fact(0, self.kernel.grid - 1))
        derive = self.idx.derive
        tid = derive("b32", "mov.u32", "%tid.x")
        gid = derive("b32", "mad.lo.s32", derive("b32", "mov.u32", "%ctaid.x"),
                     derive("b32", "mov.u32", "%ntid.x"), tid)
        launched = self.kernel.launched_threads
        if launched is None or launched > cells:
            # The guard is not in the IR: the launch may carry more threads than cells.
            self.b.emit("bra", self._exit(), guard=derive("pred", "setp.ge.s32", gid, cells))
        self.idx.assume(gid, Fact(0, cells - 1))  # past the guard, or no thread beyond the cells

        with self.scopes.nested():
            for axis, extent, coord in zip(tile.axes, extents, self._decode_cell(tile, extents, gid)):
                if coord.reg is not None and coord.const == 0:
                    self.idx.assume(coord.reg, Fact(0, extent - 1))
                self.scopes.define(axis.name, coord)
            used = _names_used(tile.body)
            if "lane" in used:
                self.scopes.define("lane", Lin(derive("b32", "and.b32", tid, 31)))
            if "warp" in used:
                self.scopes.define("warp", Lin(derive("b32", "shr.u32", tid, 5)))
            self._stmts(tile.body)

    def _decode_cell(self, tile, extents: list[int], gid: str) -> list[Lin]:
        """Every axis coordinate of this thread's cell. The cells are numbered row-major (the
        last axis fastest), except for a rasterized (m, n) block pair: there consecutive CTAs walk
        `raster_group` tiles of the grouped axis first, so they reuse operand slabs from L2."""
        idx = self.idx
        strides = [math.prod(extents[i + 1:]) for i in range(len(extents))]
        coords: dict[int, Lin] = {}

        raster = _raster_pair(tile, extents)
        if raster is not None:
            grouped, other, group = raster
            lo, hi = sorted((grouped, other))
            span = group * extents[other]  # cells of one group of grouped-axis tiles
            sub = self._udiv(Lin(gid), strides[hi])
            if lo != 0:
                sub = self._urem(sub, extents[grouped] * extents[other])
            within = self._urem(sub, span)
            first = idx.mul(self._udiv(sub, span), Lin(None, group))
            if extents[grouped] % group == 0:
                size = Lin(None, group)
            else:  # the last group is ragged
                rest = idx.reg(idx.add(Lin(None, extents[grouped]), idx.neg(first)))
                size = Lin(idx.derive("b32", "min.s32", rest, group))
            coords[grouped] = idx.add(first, idx.arith("%", within, size))
            coords[other] = idx.arith("/", within, size)

        for i, (stride, extent) in enumerate(zip(strides, extents)):
            if i in coords:
                continue
            coord = self._udiv(Lin(gid), stride)
            coords[i] = coord if i == 0 else self._urem(coord, extent)  # gid < cells: no wrap
        return [coords[i] for i in range(len(extents))]

    def _udiv(self, v: Lin, k: int) -> Lin:
        """v / k for a non-negative v (its facts let IndexCompiler avoid a divide)."""
        return self.idx.arith("/", v, Lin(None, k))

    def _urem(self, v: Lin, k: int) -> Lin:
        return self.idx.arith("%", v, Lin(None, k))

    def _exit(self) -> str:
        if self.exit_label is None:
            (self.exit_label,) = self.b.labels("exit")
        return self.exit_label

    # ---- control flow ----------------------------------------------------------------------

    def visit_StridedLoop(self, loop) -> None:
        if loop.end is not None:
            end = self.idx.value(loop.end)
        else:
            end = Lin(None, loop.axis.extent.as_static())
        self._loop(loop.axis.name, self.idx.value(loop.start), end, self.idx.value(loop.step),
                   loop.body, loop.seed, loop.unroll)

    def visit_Loop(self, loop) -> None:
        self._loop(loop.axis.name, ZERO, Lin(None, loop.axis.extent.as_static()), Lin(None, 1),
                   loop.body, loop.seed, loop.unroll)

    def _loop(self, var: str, start: Lin, end: Lin, step: Lin, body, seed: bool, unroll: bool) -> None:
        """for (var = start; var < end; var += step).

        The trip count decides the shape (start is often a thread coordinate whose range is
        known, so every thread's count is). The same count on every thread, and few enough:
        emit the body once per trip with no loop at all. The same count but many: a loop
        that runs a few trips per iteration, the factor dividing the count. At most one
        trip: a branch. Otherwise: 4 trips per iteration while they fit, then single trips.
        Copies of the body bind `var` to counter + k*step, so their index arithmetic differs
        only in address immediates, and their loads can all be in flight together. The trips
        still run in order, so every accumulator folds exactly as the IR says.
        """
        if seed:
            self._merged[id(body)] = self._seed_accumulators(body, self._equal_accumulators(body))
        trips = self._trip_range(start, end, step)
        if trips == (0, 0):
            return
        if _indexes_fragments(var, body):  # registers cannot be indexed at run time
            if trips is None or trips[0] != trips[1] or step.reg is not None:
                raise CompileError(f"loop {var!r} over fragments needs the same constant trip count on every thread")
            self._unrolled(var, start, step.const, trips[0], body)
            return
        has_loop = any(isinstance(node, (Loop, StridedLoop)) for node in _walk(body))
        size = sum(1 for _ in _straight_line(body))
        if trips is not None and trips[0] == trips[1]:
            n = trips[0]
            if self._unrolls(n, size, has_loop, unroll):
                self._unrolled(var, start, step.const, n, body)
                return
            # A short top-level loop around inner loops, such as attention's walk over 16 to 64
            # keys of q @ kt, is unrolled whole, so ptxas can issue one key's scattered column
            # loads while the previous key's softmax update still runs (NVVM unrolls these too).
            # A loop whose inner loops stream rows (SDPA's k) stays rolled: its trips already
            # load in vectors, and unrolled it measured up to 1.9x slower. So does attention on
            # tensor cores, whose trips each hold dozens of fragments: unrolled, they spill.
            if (has_loop and not self._in_loop and n <= _FULL_UNROLL_TRIPS
                    and n * _expanded_size(body) <= _OUTER_UNROLL_BUDGET and not _streams_rows(var, body)
                    and not any(isinstance(node, FragmentRowReduce) for node in _walk(body))):
                self._unrolled(var, start, step.const, n, body)
                return
            # A constant start leaves ptxas a constant trip count. Inside another loop it does
            # best itself (unrolling the inner loop and hoisting its invariant loads out of the
            # outer one); elsewhere it may leave a long loop rolled, and it never merges shared
            # loads across trips. So those are unrolled here, as NVVM does, and vectorize.py
            # turns a column walk into .v4 loads.
            ours = start.reg is not None or self._reads_shared(body) or not self._in_loop
            factor = 1 if has_loop or not ours else next(
                (f for f in (8, 4, 2) if n % f == 0 and f * size <= _PARTIAL_UNROLL_BUDGET), 1)
            self._counted_loop(var, start, end, step, body, factor, uniform=True)
        elif trips is not None and trips[1] == 1:
            self._at_most_once(var, start, end, body)
        else:
            constant = end.reg is None and step.reg is None
            factor = 4 if constant and not has_loop and 4 * size <= _PARTIAL_UNROLL_BUDGET else 1
            self._counted_loop(var, start, end, step, body, factor, uniform=False)

    def _unrolls(self, n: int, size: int, has_loop: bool, unroll: bool) -> bool:
        """Whether a loop of n trips on every thread, `size` statements, becomes n copies."""
        budget = _UNROLL_BUDGET if unroll else _IMPLICIT_UNROLL_BUDGET
        wanted = unroll or (self._unroll_room and n <= _FULL_UNROLL_TRIPS)
        return n == 1 or (not has_loop and wanted and n * size <= budget)

    def _reads_shared(self, body) -> bool:
        return any(isinstance(node, Load) and node.input in self.shared for node in _walk(body))

    def _trip_range(self, start: Lin, end: Lin, step: Lin) -> tuple[int, int] | None:
        """(fewest, most) trips any thread makes, when the bounds and start's range tell."""
        first = self.idx.fact(start)
        if end.reg is not None or step.reg is not None or step.const <= 0 or None in (first.lo, first.hi):
            return None
        def trips(s: int) -> int:
            return max(0, -(-(end.const - s) // step.const))
        return trips(first.hi), trips(first.lo)

    def _copy(self, var: str, value: Lin, body, annotate: bool, hoist: Mark | None = None) -> None:
        """One copy of a loop body with `var` bound to `value`. Index arithmetic that does not
        depend on the copy moves to `hoist` (by default, the start of this copy) and is shared
        with the copies that follow. An accumulator that merged copies follow gets its trip's
        value in a fresh register (_accumulator), moved back at the end of the trip."""
        merged = self._merged.get(id(body), {})
        saved = self._copies, self._followed
        with self.scopes.nested(hoist=hoist or self.b.mark()):
            self.scopes.define(var, value)
            carried = {name: self.scopes.lookup(name) for name in dict.fromkeys(merged.values())}
            self._copies, self._followed = {**saved[0], **merged}, saved[1] | set(carried)
            try:
                self._stmts(body, annotate=annotate)
            finally:
                self._copies, self._followed = saved
            for name, reg in carried.items():
                self.b.emit("mov.f32", reg, self.scopes.lookup(name))

    def _unrolled(self, var: str, start: Lin, step: int, n: int, body) -> None:
        for k in range(n):
            self.b.comment(f"{var}: trip {k} of {n}")
            self._trips.append((var, k))
            try:
                self._copy(var, Lin(start.reg, start.const + k * step), body, annotate=k == 0)
            finally:
                self._trips.pop()

    def _at_most_once(self, var: str, start: Lin, end: Lin, body) -> None:
        enter = self.idx.compare("lt", start, end)
        if enter is True:
            self._copy(var, start, body, annotate=True)
            return
        before = self.b.mark()
        (skip,) = self.b.labels("once_end")
        self.b.emit("bra", skip, guard="!" + enter)
        self._copy(var, start, body, annotate=True, hoist=before)
        self.b.place(skip)

    def _counted_loop(self, var: str, start: Lin, end: Lin, step: Lin, body, factor: int, uniform: bool) -> None:
        """A real loop, `factor` trips per iteration; then, unless every thread's trip count is
        a multiple of the factor, a loop of single trips for the rest. Tested at the bottom."""
        saved = self._unroll_room, self._in_loop
        self._unroll_room = _expanded_size(body) <= _ENCLOSING_UNROLL_BUDGET or _streams_rows(var, body)
        self._in_loop = True
        try:
            last = None
            if factor == 1 and uniform and not saved[1]:
                # Prefetching is worth unrolling the inner loops for, whatever their size: it
                # removes the stalls that made ptxas's own schedule of them better (a01:
                # 76.4 -> 43.7 us). Only when it then applies, so no loop unrolls without it.
                room = self._unroll_room
                self._unroll_room = True
                last = self._prefetch_plan(var, start, end, step, body)
                if last is None:
                    self._unroll_room = room
            self._emit_counted_loop(var, start, end, step, body, factor, uniform, last)
        finally:
            self._unroll_room, self._in_loop = saved

    def _prefetch_plan(self, var: str, start: Lin, end: Lin, step: Lin, body) -> int | None:
        """The value of `var` on the last trip, when the loop's loads are prefetched.

        Attention's walk over the keys reads each key's row of k (or column of kt) and of v,
        and the trip's work cannot start until they arrive. So each trip first loads the next
        key's values into a second set of registers, then computes with the values the trip
        before loaded, and finally moves the new ones over. The loads then arrive while the
        previous key's dot products and softmax update run (24 attention kernels measured
        0.36x to 0.97x the time). This needs a top-level loop with constant bounds and every
        inner loop unrolled here, so that each load site reads one fixed address per trip.
        """
        if start.reg is not None or end.reg is not None or step.reg is not None or step.const <= 0:
            return None
        n = -(-(end.const - start.const) // step.const)
        if n < 2:
            return None
        for stmt in _straight_line(body):
            if isinstance(stmt, StridedLoop):
                return None
            if isinstance(stmt, Loop):
                inner_size = sum(1 for _ in _straight_line(stmt.body))
                nested = any(isinstance(node, (Loop, StridedLoop)) for node in _walk(stmt) if node is not stmt)
                if not self._unrolls(stmt.axis.extent.as_static(), inner_size, nested, stmt.unroll):
                    return None
        if not any(isinstance(node, Load) and self._streamed(node, var) for node in _walk(body)):
            return None
        return start.const + (n - 1) * step.const

    def _streamed(self, load, var: str) -> bool:
        """A load from a read-only input whose address changes with `var`."""
        if load.input in self.shared or self.kernel.buffers[load.input].is_output:
            return False
        return any(var in _names_used(e) for e in load.index)

    def _prefetch_loads(self, var: str, value: Lin, body) -> dict[tuple, list[str]]:
        """Load every streamed site of `body` with `var` bound to `value`: site -> registers.
        A site is a Load statement in one unrolled copy, keyed like _trips."""
        sites: dict[tuple, list[str]] = {}
        with self.scopes.nested(hoist=self.b.mark()):
            self.scopes.define(var, value)
            self._prefetch_walk(body, var, sites, {})
        return sites

    def _prefetch_walk(self, body, var: str, sites: dict, loaded: dict) -> None:
        for stmt in body:
            if isinstance(stmt, Loop):  # unrolled in the body too (_prefetch_plan)
                for k in range(stmt.axis.extent.as_static()):
                    self._trips.append((stmt.axis.name, k))
                    try:
                        with self.scopes.nested(hoist=self.b.mark()):
                            self.scopes.define(stmt.axis.name, Lin(None, k))
                            self._prefetch_walk(stmt.body, var, sites, loaded)
                    finally:
                        self._trips.pop()
            elif isinstance(stmt, Load) and self._streamed(stmt, var):
                names = sorted({n for e in stmt.index for n in _names_used(e)})
                if not all(self.scopes.find(n) is not None for n in names):
                    continue  # an index the body computes itself: loaded there as usual
                # Sites with the same index under the same values read one address for every
                # value of var (attention's rows all read k[key, d]), so they share registers,
                # alike in every trip. An address match alone could hold for one value only.
                same = (stmt.input, repr(stmt.index), tuple((n, repr(self.scopes.lookup(n))) for n in names))
                if same not in loaded:
                    flat = self._flat(stmt.input, stmt.index)
                    regs, pos = [], 0
                    for width in _vector_widths(len(stmt.names)):
                        _, addr = self._address(stmt.input, Lin(flat.reg, flat.const + pos))
                        regs += self._load(f"ld.global.nc{_vec(width)}.f32", addr, width)
                        pos += width
                    loaded[same] = regs
                sites[id(stmt), tuple(self._trips)] = loaded[same]

    def _emit_counted_loop(self, var: str, start: Lin, end: Lin, step: Lin, body, factor: int, uniform: bool,
                           last: int | None = None) -> None:
        counter = self.b.reg("b32")
        self.b.emit("mov.s32", counter, self.idx.operand(start))
        if step.reg is None and step.const > 0:
            # The counter steps from start by step (by factor * step when only the unrolled
            # loop runs), so in the body it is a multiple of what both are multiples of: what
            # proves m01-style column walks aligned for vector loads.
            first = self.idx.fact(start)
            stride = step.const * factor if uniform else step.const
            top = None
            if end.reg is None:  # in the body, counter < the bound of the loop that runs it
                top = (end.const - (factor - 1) * step.const if uniform else end.const) - 1
            self.idx.assume(counter, Fact(first.lo, top, math.gcd(first.div, stride)))
        if factor > 1:
            last_start = end.const - (factor - 1) * step.const  # the iteration's last trip fits
            self._loop_iterations(var, counter, last_start, step.const, factor, body, test_entry=not uniform)
        if factor == 1 or not uniform:
            enter = self.idx.compare("lt", start, end) if factor == 1 else None
            self._loop_iterations(var, counter, self.idx.operand(end), self.idx.operand(step), 1, body,
                                  test_entry=enter is not True,
                                  prefetch=(start, last) if last is not None and factor == 1 else None)

    def _loop_iterations(self, var: str, counter: str, bound, step, factor: int, body, test_entry: bool,
                         prefetch: tuple[Lin, int] | None = None) -> None:
        """while counter < bound: `factor` copies of the body; counter += factor * step.
        Loop-invariant index arithmetic goes to the preheader, ahead of the entry test.
        With `prefetch` = (first, last value of var), the body's streamed loads come from the
        trip before (_prefetch_plan)."""
        if prefetch:  # ahead of the preheader mark, which what the body hoists may use
            current = self._prefetch_loads(var, prefetch[0], body)
        preheader = self.b.mark()
        head, done = self.b.labels("loop", "loop_end")
        if test_entry:
            skip = self.b.reg("pred")  # not derive(): the counter changes every trip
            self.b.emit("setp.ge.s32", skip, counter, bound)
            self.b.emit("bra", done, guard=skip)
        self.b.place(head)
        with self.scopes.nested(hoist=preheader):  # one iteration: what depends on the counter
            self.idx.depth[counter] = self.scopes.current.depth
            if prefetch:
                # The next trip's value of var, held at the last one so the loads stay in bounds.
                following = self.b.reg("b32")
                self.b.emit("add.s32", following, counter, step)
                self.b.emit("min.s32", following, following, prefetch[1])
                self.idx.assume(following, Fact(self.idx.fact(Lin(counter)).lo, prefetch[1],
                                                self.idx.fact(Lin(counter)).div))
                self.idx.depth[following] = self.scopes.current.depth
                upcoming = self._prefetch_loads(var, Lin(following), body)
                self._prefetched = current
            try:
                for k in range(factor):  # factor > 1 only with a constant step
                    self._copy(var, Lin(counter, k * step if k else 0), body, annotate=k == 0)
            finally:
                self._prefetched = None
            if prefetch:
                moved: set[str] = set()
                for site, regs in current.items():
                    for reg, value in zip(regs, upcoming[site]):
                        if reg not in moved:
                            moved.add(reg)
                            self.b.emit("mov.f32", reg, value)
        again = self.b.reg("pred")
        self.b.emit("add.s32", counter, counter, step * factor if factor > 1 else step)
        self.b.emit("setp.lt.s32", again, counter, bound)
        self.b.emit("bra", head, guard=again)
        self.b.place(done)

    def _seed_accumulators(self, body, same: dict[str, str]) -> dict[str, str]:
        """A seeding loop starts each accumulator it folds at the fold's identity (0 for add,
        -1e30 for maximum, ...). Accumulators inside nested loops are those loops' business.
        A copy in `same` (_equal_accumulators) gets no register: its name stands for the
        accumulator it equals. Returns the copies merged that way; one already defined
        outside the loop keeps its own register, which code after the loop reads."""
        merged = {}
        for stmt in _straight_line(body):
            if isinstance(stmt, Accum):
                if stmt.op.identity is None:
                    raise CompileError(f"accumulator {stmt.name!r}: {stmt.op.name} has no identity")
                d = dt.dtype(stmt.dtype)
                self._types[stmt.name] = d
                if stmt.name in same and self.scopes.find(stmt.name) is None:
                    self.scopes.define(stmt.name, self.scopes.lookup(same[stmt.name]))
                    merged[stmt.name] = same[stmt.name]
                else:
                    dt.move(self.b, self._register(stmt.name, d), dt.immediate(stmt.op.identity, d), d)
        return merged

    def _equal_accumulators(self, body) -> dict[str, str]:
        """Accumulators of a seeding loop that hold equal values after every trip: copy -> the
        accumulator it equals, the first of them the body updates.

        Emmy repeats online softmax's running max and sum for each output column a thread
        owns (acc1__c0_0 and acc1__c0_1 are one row's max), and neither ptxas nor NVVM merges
        values carried around a loop. Accumulators are equal when they start at the same
        identity and each trip folds equal values into them. That is checked by numbering
        the values of one trip, given that the accumulators of a class are equal at its start,
        and splitting the classes whose new values differ until none does: the optimistic
        rule of value numbering for loops. A body with statements other than plain values,
        loads, writes, branches and loops is left alone, and so is an accumulator written
        twice or inside a nested statement.
        """
        plain = (Assign, Accum, Let, Select, Load, Write, Loop, StridedLoop, Cond)
        defined: Counter = Counter()
        for stmt in _statements(body):
            if not isinstance(stmt, plain):
                return {}
            defined.update(_defined(stmt))
        folds = [s for s in body if isinstance(s, Accum) and defined[s.name] == 1 and s.op.identity is not None]
        identities: dict[str, int] = {}
        cls = {s.name: identities.setdefault((repr(float(s.op.identity)), dt.dtype(s.dtype).name), len(identities))
               for s in folds}
        while True:
            values = self._value_numbers(body, cls)
            keys: dict[tuple, int] = {}
            refined = {name: keys.setdefault((c, values[name]), len(keys)) for name, c in cls.items()}
            if len(keys) == len(set(cls.values())):
                break
            cls = refined
        first: dict[int, str] = {}
        return {name: first[c] for name, c in cls.items() if first.setdefault(c, name) != name}

    def _value_numbers(self, body, cls: dict[str, int]) -> dict[str, tuple]:
        """A token for each name after one trip of `body`: names with equal tokens hold equal
        values, given that the accumulators of one class in `cls` are equal at the start."""
        tokens: dict[str, tuple] = {name: ("start", c) for name, c in cls.items()}
        fresh = itertools.count()

        def of(name: str) -> tuple:
            return tokens.get(name, ("outside", name))

        for stmt in body:
            if isinstance(stmt, Assign):
                tokens[stmt.name] = (stmt.op.name, dt.dtype(stmt.dtype).name, *map(of, stmt.args))
            elif isinstance(stmt, Let):
                tokens[stmt.name] = ("let", repr(stmt.value), *map(of, sorted(_names_used(stmt.value))))
            elif (isinstance(stmt, Load) and stmt.input not in self.shared
                  and not self.kernel.buffers[stmt.input].is_output):
                where = ("load", stmt.input, repr(stmt.index), *map(of, sorted(_names_used(stmt.index))))
                for k, name in enumerate(stmt.names):
                    tokens[name] = (*where, k)
            elif isinstance(stmt, Accum) and stmt.name in cls:
                tokens[stmt.name] = ("fold", stmt.op.name, dt.dtype(stmt.dtype).name, of(stmt.base or stmt.name),
                                     of(stmt.value))
            else:  # values this does not follow: each its own
                for inner in _statements((stmt,)):
                    for name in _defined(inner):
                        tokens[name] = ("opaque", next(fresh))
        return tokens

    def visit_Cond(self, cond) -> None:
        p = self.idx.pred(cond.cond)
        if isinstance(p, bool):
            with self.scopes.nested():
                self._stmts(cond.body if p else cond.else_body)
            return
        before = self.b.mark()  # index arithmetic of either branch can be computed up here
        else_label, end_label = self.b.labels("else", "endif")
        self.b.emit("bra", else_label if cond.else_body else end_label, guard="!" + p)
        with self.scopes.nested(hoist=before):
            self._stmts(cond.body)
        if cond.else_body:
            self.b.emit("bra", end_label)
            self.b.place(else_label)
            with self.scopes.nested(hoist=before):
                self._stmts(cond.else_body)
        self.b.place(end_label)

    # ---- scalar compute --------------------------------------------------------------------

    def visit_Let(self, let) -> None:
        d = dt.dtype(let.dtype) if let.dtype is not None else None
        if d is not None and not d.is_int and isinstance(let.value, Literal):
            self._types[let.name] = d
            self.scopes.define(let.name, dt.immediate(float(let.value.value), d))
        else:  # an integer: an index value, which data arithmetic reads as i32
            self.scopes.define(let.name, self.idx.value(let.value))

    def visit_Init(self, init) -> None:
        d = dt.dtype(init.dtype)
        self._types[init.name] = d
        dt.move(self.b, self._register(init.name, d), dt.immediate(float(init.identity), d), d)

    def visit_Assign(self, assign) -> None:
        """name = op(args) in the statement's dtype, each argument converted to it first (a
        comparison compares in the type its arguments meet in). A bool lives in an f32 register
        as 1.0 / 0.0, the representation the comparison and logic ops produce and `where` tests."""
        d = dt.dtype(assign.dtype)
        op = assign.op.name
        self._types[assign.name] = d
        if op in dt.COMPARISONS:
            meet = dt.common([self._type(a) for a in assign.args])
            args = [self._value(a, meet) for a in assign.args]
            dt.compare(self.b, op, self._register(assign.name, d), *args, meet)
        elif op == "where" and d.reg != "f32":
            cond, a, c = assign.args
            dt.select(self.b, self._register(assign.name, d), self._value(cond, dt.BOOL),
                      self._value(a, d), self._value(c, d), d)
        else:
            want = [dt.BOOL] + [d] * (len(assign.args) - 1) if op == "where" else [d] * len(assign.args)
            args = [self._value(a, w) for a, w in zip(assign.args, want)]
            self._set(assign.name, d, lambda dst: dt.emit(self.b, op, dst, args, d))

    def _set(self, name: str, d: dt.DType, emit) -> None:
        """name = what emit(register) computes, rounded to d when d is a 16-bit float."""
        if not d.is_half:
            emit(self._register(name, d))
            return
        raw = self.b.reg("f32")
        emit(raw)
        value = dt.round_to(self.b, raw, d)
        if self.scopes.find(name) is None:
            self.scopes.define(name, value)
        else:
            self.b.emit("mov.f32", self._register(name, d), value)

    def visit_Select(self, select) -> None:
        """name = the value of the first branch whose predicate holds; the last branch is the
        default. A predicate the facts settle drops out, so a known choice costs nothing."""
        d = dt.common([self._type(b.value) for b in select.branches])
        self._types[select.name] = d
        kind = "b" + str(d.bits) if d.is_int else ("f64" if d.reg == "f64" else "f32")
        choice = self._value(select.branches[-1].value, d)
        for branch in reversed(select.branches[:-1]):
            holds = self.idx.pred(branch.select)
            if holds is True:
                choice = self._value(branch.value, d)
            elif holds is not False:
                reg = self.b.reg(d.reg)
                self.b.emit(f"selp.{kind}", reg, self._value(branch.value, d), choice, holds)
                choice = reg
        if self.scopes.find(select.name) is None:
            self.scopes.define(select.name, choice)
        else:
            dt.move(self.b, self._register(select.name, d), choice, d)

    def visit_Accum(self, acc) -> None:
        """acc = op(base, value), where base defaults to the accumulator itself; in the
        accumulator's dtype, as C converts `acc += x` (a 16-bit one rounds each fold)."""
        d = dt.dtype(acc.dtype)
        value = self._value(acc.value, d)
        target = self._accumulator(acc)
        if target is None:
            return
        dst, base = target
        if not d.is_half:
            dt.emit(self.b, acc.op.name, dst, [base, value], d)
            return
        raw = self.b.reg("f32")
        dt.emit(self.b, acc.op.name, raw, [base, value], d)
        self.b.emit("mov.f32", dst, dt.round_to(self.b, raw, d))

    def _accumulator(self, acc) -> tuple[str, str] | None:
        """(the register acc's new value goes to, the base it folds into), or None for a copy
        merged into an accumulator the trip has already updated: then acc's name now stands
        for that one's new value. An accumulator copies follow takes a fresh register, since
        they read its old value until their own update (_copy moves the new one back)."""
        same = self._copies.get(acc.name)
        if same is not None:
            self.scopes.define(acc.name, self._operand(same))
            return None
        dst = self.scopes.find(acc.name)
        if dst is None:
            raise CompileError(f"accumulator {acc.name!r} is folded before it is seeded")
        d = dt.dtype(acc.dtype)
        base = self._value(acc.base, d) if acc.base else dst
        if acc.name in self._followed:
            dst = self.b.reg(d.reg)
            self.scopes.define(acc.name, dst)
        return dst, base

    def visit_Load(self, load) -> None:
        d = self._buffer_dtype(load.input)
        for name in load.names:
            self._types[name] = d
        value = self.kernel.constants.get(load.input)
        if value is not None:  # a constant of the graph: an immediate, like a Let
            for name in load.names:
                bound = self.scopes.find(name)
                if isinstance(bound, str) and is_register(bound):  # a name written in place
                    dt.move(self.b, bound, dt.immediate(value, d), d)
                else:
                    self.scopes.define(name, dt.immediate(value, d))
            return
        if self._prefetched is not None and all(self.scopes.find(n) is None for n in load.names):
            regs = self._prefetched.get((id(load), tuple(self._trips)))
            if regs is not None:  # loaded by the trip before
                for name, reg in zip(load.names, regs):
                    self.scopes.define(name, reg)
                return
        flat = self._flat(load.input, load.index)
        # Inputs are never written by the kernel: read them through the non-coherent cache and
        # reuse a value already loaded from the same address.
        readonly = load.input not in self.shared and not self.kernel.buffers[load.input].is_output
        pos = 0
        for width in _vector_widths(len(load.names), d):
            names = load.names[pos:pos + width]
            space, addr = self._address(load.input, Lin(flat.reg, flat.const + pos))
            opcode = f"ld.{space}{'.nc' if readonly else ''}{_vec(width)}.{d.mem}"
            if readonly and all(self.scopes.find(n) is None for n in names):
                regs = self.scopes.memo((opcode, addr), lambda: self._load(opcode, addr, width, d))
                for name, reg in zip(names, regs):
                    self.scopes.define(name, reg)
            elif d == dt.F32:
                self.b.emit(opcode, vector([self._register(n) for n in names]), addr)
            else:
                for name, reg in zip(names, self._load(opcode, addr, width, d)):
                    dt.move(self.b, self._register(name, d), reg, d)
            pos += width

    def _load(self, opcode: str, addr: str, width: int, d: dt.DType = dt.F32) -> list[str]:
        regs = [self.b.reg(d.mem_reg) for _ in range(width)]
        self.b.emit(opcode, vector(regs), addr)
        return dt.from_memory(self.b, regs, d)

    def visit_Write(self, write) -> None:
        """buf[i] = (v0, v1, ...) as vector stores, or one red.add per value when atomic; each
        value converted to the buffer's dtype."""
        if str(write.swizzle) != "NONE":
            raise CompileError(f"swizzled writes ({write.swizzle}) are not supported")
        d = self._buffer_dtype(write.output)
        values = [dt.to_memory(self.b, self._value(name, dt.store_type(d)), d) for name in write.values]
        flat = self._flat(write.output, write.index)
        widths = [1] * len(values) if write.atomic else _vector_widths(len(values), d)
        pos = 0
        for width in widths:
            space, addr = self._address(write.output, Lin(flat.reg, flat.const + pos))
            if write.atomic:
                self.b.emit(f"red.{space}.{dt.atomic_add(d)}", addr, values[pos])
            else:
                self.b.emit(f"st.{space}{_vec(width)}.{d.mem}", addr, vector(values[pos:pos + width]))
            pos += width

    # ---- shared memory, barriers, cross-thread combines ------------------------------------

    def visit_Smem(self, smem) -> None:
        pass  # laid out up front by _plan_shared

    def visit_CpAsyncCopy(self, copy) -> None:
        """One asynchronous global -> shared copy of 4, 8 or 16 bytes (16: .cg, bypassing L1),
        to the swizzled place of its first element when the array is swizzled (tensorcore.py)."""
        if str(copy.swizzle) != "NONE" and self.shared[copy.smem].dtype.nbytes != 2:
            raise CompileError(f"swizzled cp.async ({copy.swizzle}) into a non-16-bit array")
        if copy.nbytes not in (4, 8, 16):
            raise CompileError(f"cp.async of {copy.nbytes} bytes")
        flat = self._flat(copy.smem, copy.smem_index)
        if copy.lane_index is not None:  # smem_index is a stripe base; add the lane's (row, col)
            row, col = (self.idx.value(e) for e in copy.lane_index)
            cols = Lin(None, self.shared[copy.smem].extents[-1])
            flat = self.idx.add(flat, self.idx.add(self.idx.mul(row, cols), col))
        _, dst = self._address(copy.smem, self._swizzle(flat, copy.swizzle))
        _, src = self._address(copy.src, self._flat(copy.src, copy.src_index))
        self.b.emit(f"cp.async.{'cg' if copy.nbytes == 16 else 'ca'}.shared.global", dst, src, copy.nbytes)

    def visit_CpAsyncCommit(self, _) -> None:
        self.b.emit("cp.async.commit_group")

    def visit_CpAsyncWait(self, wait) -> None:
        self.b.emit("cp.async.wait_group", wait.group)

    def visit_Sync(self, sync) -> None:
        if sync.warp:
            self.b.emit("bar.warp.sync", "0xffffffff")
        else:
            self._barrier(sync.barrier_id, sync.count)

    def _barrier(self, barrier_id: int, count: int | None) -> None:
        if count is None:
            self.b.emit("bar.sync", barrier_id)
        else:
            self.b.emit("bar.sync", barrier_id, count)

    def visit_WarpShuffle(self, ws) -> None:
        """XOR butterfly: at distance length/2, ..., 2, 1 every lane fetches its partner's
        states (named state_b) and folds them into its own with the combine program."""
        _require_f32_held(ws.dtype, "WarpShuffle")
        mask = self.b.reg("b32")
        self.b.emit("activemask.b32", mask)
        distance = ws.length // 2
        while distance >= 1:
            self.b.comment(f"butterfly distance {distance}")
            with self.scopes.nested():
                for state, partner in zip(ws.state, ws.state_b):
                    self._types[partner] = self._type(state)
                    self.b.emit("shfl.sync.bfly.b32", self._define_new(partner),
                                self._operand(state), distance, "0x1f", mask)
                self._stmts(ws.combine_states, annotate=False)
            distance //= 2

    def visit_TreeHalve(self, tree) -> None:
        """Fold `length` partials kept in shared memory (one per warp): at stride s = length/2,
        ..., 1 entry i < s combines entry i + s into itself; then every thread reads entry 0."""
        if tree.inner is not None:
            raise CompileError("TreeHalve with an inner tree is not supported")
        _require_f32_held(tree.dtype, "TreeHalve")
        tid = self.scopes.lookup(tree.tid_var)
        arrays = [self.shared[buf] for buf in tree.bufs]
        stride = tree.length // 2
        while stride >= 1:
            self.b.comment(f"tree stride {stride}")
            active = self.idx.compare("lt", tid, Lin(None, stride))
            before = self.b.mark()
            (skip,) = self.b.labels("tree_skip")
            self.b.emit("bra", skip, guard="!" + active)
            with self.scopes.nested(hoist=before):
                mine = self.idx.shared_address
                for array, state, partner in zip(arrays, tree.state, tree.state_b):
                    self._types[partner] = self._type(state)
                    self.b.emit("ld.shared.f32", self._define_new(state),
                                mine(array.symbol, tid, _F32_BYTES))
                    self.b.emit("ld.shared.f32", self._define_new(partner),
                                mine(array.symbol, Lin(tid.reg, tid.const + stride), _F32_BYTES))
                self._stmts(tree.combine_states, annotate=False)
                for array, state in zip(arrays, tree.state):
                    self.b.emit("st.shared.f32", mine(array.symbol, tid, _F32_BYTES),
                                self._operand(state))
            self.b.place(skip)
            self._barrier(tree.barrier_id, tree.barrier_count)
            stride //= 2
        for array, state in zip(arrays, tree.state):
            self.b.emit("ld.shared.f32", self._register(state), address(array.symbol))

    # ---- names and memory ------------------------------------------------------------------

    def _operand(self, name: str) -> str:
        value = self.scopes.lookup(name)
        if isinstance(value, Lin):
            raise CompileError(f"integer {name!r} used as an f32 value")
        return value

    def _type(self, name: str) -> dt.DType:
        """The dtype of a value name; an index (a Lin) reads as i32."""
        if name in self._types:
            return self._types[name]
        return dt.I32 if isinstance(self.scopes.find(name), Lin) else dt.F32

    def _value(self, name: str, want: dt.DType) -> str:
        """The value of `name` as dtype `want`, converted if it has another."""
        value = self.scopes.lookup(name)
        if isinstance(value, Lin):
            return dt.convert(self.b, self.idx.operand(value), dt.I32, want)
        return dt.convert(self.b, value, self._type(name), want)

    def _register(self, name: str, d: dt.DType = dt.F32) -> str:
        """The f32 register `name` is written to. Assigning a name that already exists (an
        accumulator, a reduction state) updates it in place, as in C; a new name gets a fresh
        register in the current scope."""
        value = self.scopes.find(name)
        if value is None:
            return self._define_new(name, d)
        if not (isinstance(value, str) and is_register(value)):
            raise CompileError(f"cannot assign to {name!r}")
        return value

    def _define_new(self, name: str, d: dt.DType = dt.F32) -> str:
        """A fresh register for `name`, shadowing any outer definition until the scope ends."""
        reg = self.b.reg(d.reg)
        self.scopes.define(name, reg)
        return reg

    def _in_register(self, name: str, d: dt.DType = dt.F32) -> str:
        operand = self._value(name, d)
        if is_register(operand):
            return operand
        reg = self.b.reg(d.reg)
        dt.move(self.b, reg, operand, d)
        return reg

    def _flat(self, buffer: str, index) -> Lin:
        if buffer in self.shared:
            shape = self.shared[buffer].extents
        elif buffer in self.kernel.buffers:
            shape = self.kernel.buffers[buffer].shape
        else:
            raise CompileError(f"unknown buffer {buffer!r}")
        return self.idx.flat(tuple(index), shape)

    def _address(self, buffer: str, flat: Lin) -> tuple[str, str]:
        """(state space, memory operand) of element `flat` of a buffer or shared array."""
        if buffer in self.shared:
            array = self.shared[buffer]
            esize = array.dtype.nbytes
            in_pool = Lin(flat.reg, flat.const + array.offset // esize)
            return "shared", self.idx.shared_address(array.symbol, in_pool, esize)
        if buffer not in self.globals:
            raise CompileError(f"buffer {buffer!r} is not a kernel parameter")
        return "global", self.idx.global_address(self.globals[buffer], flat, self._buffer_dtype(buffer).nbytes)

    def _buffer_dtype(self, buffer: str) -> dt.DType:
        if buffer in self.shared:
            return self.shared[buffer].dtype
        return dt.dtype(self.kernel.buffers[buffer].dtype)


# ---- helpers ---------------------------------------------------------------------------------

def _param_symbol(name: str) -> str:
    return f"param_{name}"


def _vec(width: int) -> str:
    return "" if width == 1 else f".v{width}"


def _vector_widths(n: int, d: dt.DType = dt.F32) -> list[int]:
    """Split n consecutive values into vector accesses (the IR only groups aligned runs). Bytes
    (bools) go one at a time: PTX's vector byte accesses want 16-bit registers."""
    if d.nbytes == 1:
        return [1] * n
    if n in (1, 2, 4):
        return [n]
    width = 4 if n % 4 == 0 else 2 if n % 2 == 0 else 1
    return [width] * (n // width)


def _require_f32_held(dtype, what: str) -> None:
    """Cross-thread combines move 32-bit registers: f32, and the 16-bit floats held in f32."""
    if dt.dtype(dtype).reg != "f32":
        raise CompileError(f"{what}: dtype {getattr(dtype, 'name', dtype)} is not supported")


def _indexes_fragments(var: str, body) -> bool:
    """Whether a loop body indexes a fragment array with the loop variable (`_b[_r0]`)."""
    index = f"[{var}]"
    for node in _walk(body):
        for f in FRAGMENT_FIELDS:
            value = getattr(node, f, None)
            if any(isinstance(v, str) and index in v for v in (value if isinstance(value, tuple) else (value,))):
                return True
    return False


def _statements(body) -> Iterator:
    """Every statement of a body, those in nested branches and loops included."""
    for stmt in body:
        yield stmt
        for field in ("body", "else_body"):
            yield from _statements(getattr(stmt, field, None) or ())


def _defined(stmt) -> tuple[str, ...]:
    """The names a plain statement defines."""
    if isinstance(stmt, Load):
        return tuple(stmt.names)
    if isinstance(stmt, (Assign, Accum, Let, Select)):
        return (stmt.name,)
    return ()


def _straight_line(body) -> Iterator:
    """The statements of a body, descending into branches but not into nested loops."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, Cond):
            yield from _straight_line(stmt.body)
            yield from _straight_line(stmt.else_body)


def _raster_pair(tile, extents: list[int]) -> tuple[int, int, int] | None:
    """(grouped axis, other axis, group size) when the tile rasterizes its CTA order. That needs
    a group of at least 2 and the (m, n) block axes adjacent, m first."""
    if not (tile.raster_group and tile.raster_axes):
        return None
    names = [axis.name for axis in tile.axes]
    m_axis, n_axis = tile.raster_axes
    if m_axis not in names or n_axis not in names:
        return None
    m, n = names.index(m_axis), names.index(n_axis)
    if n != m + 1:
        return None
    grouped, other = (m, n) if tile.raster_orient == "m" else (n, m)
    group = min(tile.raster_group, extents[grouped])
    if extents[grouped] <= 1 or group <= 1:
        return None
    return grouped, other, group


def _constant_trips(loop) -> int | None:
    extent = loop.axis.extent.as_static()
    if isinstance(loop, Loop):
        return extent
    if loop.end is None and isinstance(loop.start, Literal) and isinstance(loop.step, Literal):
        return max(0, -(-(extent - int(loop.start.value)) // int(loop.step.value)))
    return None


def _expanded_size(body) -> int:
    """The statements of a loop body, counting each inner loop with a small constant trip
    count as unrolled."""
    total = 0
    for stmt in _straight_line(body):
        if isinstance(stmt, (Loop, StridedLoop)):
            inner = sum(1 for _ in _straight_line(stmt.body))
            trips = _constant_trips(stmt)
            total += inner * trips if trips is not None and trips <= _FULL_UNROLL_TRIPS else inner
        else:
            total += 1
    return total


def _streams_rows(outer: str, body) -> bool:
    """Whether an inner loop of `body` reads, at unit stride along its own axis, a row that
    changes with every trip of the loop over `outer`: attention's k[..., key, d] under
    `for key`. ptxas cannot hoist those loads out of the outer loop and keeps them scalar;
    unrolled here, vectorize.py merges them into .v4 loads, a quarter of the instructions."""
    for stmt in _straight_line(body):
        if not isinstance(stmt, Loop):
            continue
        inner = stmt.axis.name
        for node in _walk(stmt):
            if (isinstance(node, Load) and node.index and _unit_stride(node.index[-1], inner)
                    and not any(inner in _names_used(e) for e in node.index[:-1])
                    and any(outer in _names_used(e) for e in node.index)):
                return True
    return False


def _unit_stride(e, name: str) -> bool:
    """Whether `e` is `name` plus terms that do not use it."""
    if isinstance(e, Var):
        return e.name == name
    if isinstance(e, BinaryExpr) and e.op == "+":
        return ((_unit_stride(e.left, name) and name not in _names_used(e.right))
                or (_unit_stride(e.right, name) and name not in _names_used(e.left)))
    return False


def _walk(node) -> Iterator:
    """Every IR node (statement or expression) under `node`, parents before children."""
    stack = [node]
    while stack:
        item = stack.pop()
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            yield item
            stack.extend(reversed([getattr(item, f.name) for f in dataclasses.fields(item)]))
        elif isinstance(item, (tuple, list)):
            stack.extend(reversed(item))


def _string_leaves(node) -> Iterator[str]:
    """The string fields of one IR node, looking inside tuples (names, args, values) and the
    records they hold (a Select's branches)."""
    for f in dataclasses.fields(node):
        value = getattr(node, f.name)
        if isinstance(value, str):
            yield value
        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, str):
                    yield v
                elif isinstance(v, SelectBranch):
                    yield from _string_leaves(v)


def _names_used(node) -> set[str]:
    """Every name an IR subtree refers to: its Vars, plus TreeHalve's tid variable (a string)."""
    found: set[str] = set()
    for item in _walk(node):
        if isinstance(item, Var):
            found.add(item.name)
        elif isinstance(item, TreeHalve):
            found.add(item.tid_var)
    return found
