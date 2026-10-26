"""Loops: how each IR loop becomes PTX, and the accumulators loops fold.

The trip count decides a loop's shape (_loop): the same small count on every thread unrolls
it, copies of the body binding the variable to counter + k*step, so that their index
arithmetic differs only in address immediates; at most one trip becomes a branch; the rest are
real counted loops, a few trips per iteration where the count allows. A top-level walk with
constant bounds whose inner loops unroll prefetches the next trip's loads (attention's key
walk, _prefetch_plan). A seeding loop starts each accumulator at its fold's identity, and
copies that always hold equal values are computed once (_equal_accumulators).

`Loops` is a mixin of KernelLowering: it emits through the lowering's builder, scopes and index
compiler, and keeps its state there (see KernelLowering.__init__).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import FragmentRowReduce
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Select, Write

from . import dtypes as dt
from .errors import CompileError
from .facts import Fact
from .index import Lin
from .irtree import names_defined, names_used, statements, straight_line, walk
from .ptx import Mark
from .tensorcore import FRAGMENT_FIELDS

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
_READ_ONCE = 64                 # f64 values a prefetching loop loads once for all its trips


class Loops:
    """Loop shapes, prefetching and accumulator seeding: methods of KernelLowering."""

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
        has_loop = any(isinstance(node, (Loop, StridedLoop)) for node in walk(body))
        size = sum(1 for _ in straight_line(body))
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
                    and n * _expanded_size(body) <= _OUTER_UNROLL_BUDGET * self.unroll_scale
                    and not _streams_rows(var, body)
                    and not any(isinstance(node, FragmentRowReduce) for node in walk(body))):
                self._unrolled(var, start, step.const, n, body)
                return
            # A constant start leaves ptxas a constant trip count. Inside another loop it does
            # best itself (unrolling the inner loop and hoisting its invariant loads out of the
            # outer one); elsewhere it may leave a long loop rolled, and it never merges shared
            # loads across trips. So those are unrolled here, as NVVM does, and vectorize.py
            # turns a column walk into .v4 loads.
            ours = start.reg is not None or self._reads_shared(body) or not self._in_loop
            factor = 1 if has_loop or not ours else next(
                (f for f in (8, 4, 2) if n % f == 0 and f * size <= _PARTIAL_UNROLL_BUDGET * self.unroll_scale), 1)
            self._counted_loop(var, start, end, step, body, factor, uniform=True)
        elif trips is not None and trips[1] == 1:
            self._at_most_once(var, start, end, body)
        else:
            constant = end.reg is None and step.reg is None
            factor = 4 if constant and not has_loop and 4 * size <= _PARTIAL_UNROLL_BUDGET * self.unroll_scale else 1
            self._counted_loop(var, start, end, step, body, factor, uniform=False)

    def _unrolls(self, n: int, size: int, has_loop: bool, unroll: bool) -> bool:
        """Whether a loop of n trips on every thread, `size` statements, becomes n copies."""
        budget = (_UNROLL_BUDGET if unroll else _IMPLICIT_UNROLL_BUDGET) * self.unroll_scale
        wanted = unroll or (self._unroll_room and n <= _FULL_UNROLL_TRIPS)
        return n == 1 or (not has_loop and wanted and n * size <= budget)

    def _reads_shared(self, body) -> bool:
        return any(isinstance(node, Load) and node.input in self.shared for node in walk(body))

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
        self._unroll_room = (_expanded_size(body) <= _ENCLOSING_UNROLL_BUDGET * self.unroll_scale
                             or _streams_rows(var, body))
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
        for stmt in straight_line(body):
            if isinstance(stmt, StridedLoop):
                return None
            if isinstance(stmt, Loop):
                inner_size = sum(1 for _ in straight_line(stmt.body))
                nested = any(isinstance(node, (Loop, StridedLoop)) for node in walk(stmt) if node is not stmt)
                if not self._unrolls(stmt.axis.extent.as_static(), inner_size, nested, stmt.unroll):
                    return None
        if not any(isinstance(node, Load) and self._streamed(node, var) for node in walk(body)):
            return None
        return start.const + (n - 1) * step.const

    def _streamed(self, load, var: str) -> bool:
        """A load from a read-only input whose address changes with `var`."""
        if load.input in self.shared or self.kernel.buffers[load.input].is_output:
            return False
        return any(var in names_used(e) for e in load.index)

    def _prefetch_loads(self, var: str, value: Lin, body, fixed: dict | None = None) -> dict[tuple, list[str]]:
        """Load every streamed site of `body` with `var` bound to `value`: site -> registers.
        A site is a Load statement in one unrolled copy, keyed like _trips. With `fixed`, the
        f64 sites that read the same address on every trip go there, loaded this once."""
        sites: dict[tuple, list[str]] = {}
        with self.scopes.nested(hoist=self.b.mark()):
            self.scopes.define(var, value)
            self._prefetch_walk(body, var, sites, {}, fixed)
        return sites

    def _prefetch_walk(self, body, var: str, sites: dict, loaded: dict, fixed: dict | None = None) -> None:
        for stmt in body:
            if isinstance(stmt, Loop):  # unrolled in the body too (_prefetch_plan)
                for k in range(stmt.axis.extent.as_static()):
                    self._trips.append((stmt.axis.name, k))
                    try:
                        with self.scopes.nested(hoist=self.b.mark()):
                            self.scopes.define(stmt.axis.name, Lin(None, k))
                            self._prefetch_walk(stmt.body, var, sites, loaded, fixed)
                    finally:
                        self._trips.pop()
            elif isinstance(stmt, Load) and (self._streamed(stmt, var) or self._read_once(stmt, var, fixed)):
                names = sorted({n for e in stmt.index for n in names_used(e)})
                if not all(self.scopes.find(n) is not None for n in names):
                    continue  # an index the body computes itself: loaded there as usual
                # Sites with the same index under the same values read one address for every
                # value of var (attention's rows all read k[key, d]), so they share registers,
                # alike in every trip. An address match alone could hold for one value only.
                same = (stmt.input, repr(stmt.index), tuple((n, repr(self.scopes.lookup(n))) for n in names))
                if same not in loaded:
                    d = self._buffer_dtype(stmt.input)
                    flat = self._flat(stmt.input, stmt.index)
                    regs, pos = [], 0
                    for width in dt.vector_widths(len(stmt.names), d):
                        _, addr = self._address(stmt.input, Lin(flat.reg, flat.const + pos))
                        regs += self._load(f"ld.global.nc{dt.vec_suffix(width)}.{d.mem}", addr, width, d)
                        pos += width
                    loaded[same] = (regs, d)
                (sites if self._streamed(stmt, var) else fixed)[id(stmt), tuple(self._trips)] = loaded[same]

    def _read_once(self, load, var: str, fixed: dict | None) -> bool:
        """An f64 read held in f32 whose address is the same on every trip of the loop over var
        (attention's q row, under `for key`): it costs a conversion each time it is loaded,
        and ptxas does not always hoist it (it did not for c_f64_a01_attention once most reads
        were converted in integer instructions), so the first trip's prefetch loads it for all
        of them, up to _READ_ONCE values per loop."""
        if fixed is None or self._buffer_dtype(load.input) is not dt.F64_AS_F32:
            return False
        if load.input in self.shared or self.kernel.buffers[load.input].is_output:
            return False
        return len({r for regs, _ in fixed.values() for r in regs}) + len(load.names) <= _READ_ONCE

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
            fixed: dict[tuple, tuple[list[str], dt.DType]] = {}
            current = self._prefetch_loads(var, prefetch[0], body, fixed)
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
                self._prefetched = {**current, **fixed}
            try:
                for k in range(factor):  # factor > 1 only with a constant step
                    self._copy(var, Lin(counter, k * step if k else 0), body, annotate=k == 0)
            finally:
                self._prefetched = None
            if prefetch:
                moved: set[str] = set()
                for site, (regs, d) in current.items():
                    for reg, value in zip(regs, upcoming[site][0]):
                        if reg not in moved:
                            moved.add(reg)
                            dt.move(self.b, reg, value, d)
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
        for stmt in straight_line(body):
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
        for stmt in statements(body):
            if not isinstance(stmt, plain):
                return {}
            defined.update(names_defined(stmt))
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
                tokens[stmt.name] = ("let", repr(stmt.value), *map(of, sorted(names_used(stmt.value))))
            elif (isinstance(stmt, Load) and stmt.input not in self.shared
                  and not self.kernel.buffers[stmt.input].is_output):
                where = ("load", stmt.input, repr(stmt.index), *map(of, sorted(names_used(stmt.index))))
                for k, name in enumerate(stmt.names):
                    tokens[name] = (*where, k)
            elif isinstance(stmt, Accum) and stmt.name in cls:
                tokens[stmt.name] = ("fold", stmt.op.name, dt.dtype(stmt.dtype).name, of(stmt.base or stmt.name),
                                     of(stmt.value))
            else:  # values this does not follow: each its own
                for inner in statements((stmt,)):
                    for name in names_defined(inner):
                        tokens[name] = ("opaque", next(fresh))
        return tokens


# ---- helpers ---------------------------------------------------------------------------------


def _indexes_fragments(var: str, body) -> bool:
    """Whether a loop body indexes a fragment array with the loop variable (`_b[_r0]`)."""
    index = f"[{var}]"
    for node in walk(body):
        for f in FRAGMENT_FIELDS:
            value = getattr(node, f, None)
            if any(isinstance(v, str) and index in v for v in (value if isinstance(value, tuple) else (value,))):
                return True
    return False


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
    for stmt in straight_line(body):
        if isinstance(stmt, (Loop, StridedLoop)):
            inner = sum(1 for _ in straight_line(stmt.body))
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
    for stmt in straight_line(body):
        if not isinstance(stmt, Loop):
            continue
        inner = stmt.axis.name
        for node in walk(stmt):
            if (isinstance(node, Load) and node.index and _unit_stride(node.index[-1], inner)
                    and not any(inner in names_used(e) for e in node.index[:-1])
                    and any(outer in names_used(e) for e in node.index)):
                return True
    return False


def _unit_stride(e, name: str) -> bool:
    """Whether `e` is `name` plus terms that do not use it."""
    if isinstance(e, Var):
        return e.name == name
    if isinstance(e, BinaryExpr) and e.op == "+":
        return ((_unit_stride(e.left, name) and name not in names_used(e.right))
                or (_unit_stride(e.right, name) and name not in names_used(e.left)))
    return False
