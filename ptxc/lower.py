"""Statement lowering: walk the Kernel IR body and emit PTX for every statement.

There is one `visit_<Statement>` method per IR statement kind. The walk tracks IR names in
`Scopes`: an integer name (an axis, a loop counter, an integer Let) maps to a `Lin`, an f32 name
maps to its operand, a register such as %f3 or an immediate such as 0f3F800000.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import TreeHalve
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.leaves import Accum

from .errors import CompileError
from .fmath import emit_elementwise
from .index import ZERO, IndexCompiler, Lin
from .kernel import Kernel
from .ptx import PtxBuilder, address, f32_imm, is_register, render_module, vector
from .scope import Scopes

_F32_BYTES = 4
_SMEM_DTYPES = {"float": "f32"}


@dataclasses.dataclass(frozen=True)
class SharedArray:
    symbol: str
    extents: tuple[int, ...]
    align: int

    def declaration(self) -> str:
        nbytes = math.prod(self.extents) * _F32_BYTES
        return f".shared .align {self.align} .b8 {self.symbol}[{nbytes}];"


class KernelLowering:
    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.b = PtxBuilder()
        self.scopes = Scopes()
        self.idx = IndexCompiler(self.b, self.scopes)
        self.shared: dict[str, SharedArray] = {}
        self.globals: dict[str, str] = {}  # buffer name -> register with its global address
        self.exit_label: str | None = None

    def run(self) -> str:
        """The complete PTX module for the kernel."""
        for name in self.kernel.params:
            generic, glob = self.b.reg("b64"), self.b.reg("b64")
            self.b.emit("ld.param.u64", generic, f"[{_param_symbol(name)}]")
            self.b.emit("cvta.to.global.u64", glob, generic)
            self.globals[name] = glob
        self._stmts(self.kernel.body)
        if self.exit_label:
            self.b.place(self.exit_label)
        self.b.emit("ret")
        return render_module(
            self.kernel.entry,
            [_param_symbol(name) for name in self.kernel.params],
            [array.declaration() for array in self.shared.values()],
            self.b,
        )

    def _stmts(self, body, *, annotate: bool = True) -> None:
        for stmt in body:
            if annotate:
                self.b.comment(stmt.pretty()[0].strip())
            visit = getattr(self, f"visit_{type(stmt).__name__}", None)
            if visit is None:
                raise CompileError(f"unsupported IR statement {type(stmt).__name__}")
            visit(stmt)

    # ---- launch geometry -------------------------------------------------------------------

    def visit_Tile(self, tile) -> None:
        """One thread per cell: decode the cell's axis coordinates from the global thread id."""
        if tile.raster_axes is not None or tile.aux_threads:
            raise CompileError("rasterized tiles are not supported yet")
        extents = [axis.extent.as_static() for axis in tile.axes]
        cells = math.prod(extents)
        derive = self.idx.derive
        tid = derive("b32", "mov.u32", "%tid.x")
        gid = derive("b32", "mad.lo.s32", derive("b32", "mov.u32", "%ctaid.x"),
                     derive("b32", "mov.u32", "%ntid.x"), tid)
        launched = self.kernel.launched_threads
        if launched is None or launched > cells:
            # The guard is not in the IR: the launch may carry more threads than cells.
            self.b.emit("bra", self._exit(), guard=derive("pred", "setp.ge.s32", gid, cells))

        with self.scopes.nested():
            stride = cells  # row-major: the last axis varies fastest
            for i, (axis, extent) in enumerate(zip(tile.axes, extents)):
                stride //= extent
                coord = gid if stride == 1 else derive("b32", "div.u32", gid, stride)
                if i > 0:  # the first axis needs no wrap: gid < cells
                    coord = derive("b32", "rem.u32", coord, extent)
                self.scopes.define(axis.name, ZERO if extent == 1 else Lin(coord))
            used = _names_used(tile.body)
            if "lane" in used:
                self.scopes.define("lane", Lin(derive("b32", "and.b32", tid, 31)))
            if "warp" in used:
                self.scopes.define("warp", Lin(derive("b32", "shr.u32", tid, 5)))
            self._stmts(tile.body)

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
                   loop.body, loop.seed)

    def visit_Loop(self, loop) -> None:
        self._loop(loop.axis.name, ZERO, Lin(None, loop.axis.extent.as_static()), Lin(None, 1),
                   loop.body, loop.seed)

    def _loop(self, var: str, start: Lin, end: Lin, step: Lin, body, seed: bool) -> None:
        """for (var = start; var < end; var += step): tested at the bottom, one branch per trip."""
        if seed:
            self._seed_accumulators(body)
        enter = self.idx.compare("lt", start, end)
        if enter is False:
            return
        end_op, step_op = self.idx.operand(end), self.idx.operand(step)  # computed once, outside
        counter = self.b.reg("b32")
        self.b.emit("mov.s32", counter, self.idx.operand(start))
        head, done = self.b.labels("loop", "loop_end")
        if enter is not True:
            self.b.emit("bra", done, guard="!" + enter)
        self.b.place(head)
        with self.scopes.nested():
            self.scopes.define(var, Lin(counter))
            self._stmts(body)
        again = self.b.reg("pred")  # not derive(): the counter changes every trip
        self.b.emit("add.s32", counter, counter, step_op)
        self.b.emit("setp.lt.s32", again, counter, end_op)
        self.b.emit("bra", head, guard=again)
        self.b.place(done)

    def _seed_accumulators(self, body) -> None:
        """A seeding loop starts each accumulator it folds at the fold's identity (0 for add,
        -1e30 for maximum, ...). Accumulators inside nested loops are those loops' business."""
        for stmt in _straight_line(body):
            if isinstance(stmt, Accum):
                if stmt.op.identity is None:
                    raise CompileError(f"accumulator {stmt.name!r}: {stmt.op.name} has no identity")
                self.b.emit("mov.f32", self._register(stmt.name), f32_imm(stmt.op.identity))

    def visit_Cond(self, cond) -> None:
        p = self.idx.pred(cond.cond)
        if isinstance(p, bool):
            with self.scopes.nested():
                self._stmts(cond.body if p else cond.else_body)
            return
        else_label, end_label = self.b.labels("else", "endif")
        self.b.emit("bra", else_label if cond.else_body else end_label, guard="!" + p)
        with self.scopes.nested():
            self._stmts(cond.body)
        if cond.else_body:
            self.b.emit("bra", end_label)
            self.b.place(else_label)
            with self.scopes.nested():
                self._stmts(cond.else_body)
        self.b.place(end_label)

    # ---- scalar compute --------------------------------------------------------------------

    def visit_Let(self, let) -> None:
        if _is_float(let.dtype):
            self.scopes.define(let.name, f32_imm(float(let.value.value)))
        else:
            self.scopes.define(let.name, self.idx.value(let.value))

    def visit_Init(self, init) -> None:
        _require_f32(init.dtype, init.name)
        self.b.emit("mov.f32", self._register(init.name), f32_imm(float(init.identity)))

    def visit_Assign(self, assign) -> None:
        _require_f32(assign.dtype, assign.name)
        args = [self._operand(name) for name in assign.args]
        emit_elementwise(self.b, assign.op.name, self._register(assign.name), args)

    def visit_Accum(self, acc) -> None:
        """acc = op(base, value), where base defaults to the accumulator itself."""
        _require_f32(acc.dtype, acc.name)
        dst = self.scopes.find(acc.name)
        if dst is None:
            raise CompileError(f"accumulator {acc.name!r} is folded before it is seeded")
        lhs = self._operand(acc.base) if acc.base else dst
        emit_elementwise(self.b, acc.op.name, dst, [lhs, self._operand(acc.value)])

    def visit_Load(self, load) -> None:
        _require_f32(load.dtype, load.input)
        flat = self._flat(load.input, load.index)
        # Inputs are never written by the kernel: read them through the non-coherent cache and
        # reuse a value already loaded from the same address.
        readonly = load.input not in self.shared and not self.kernel.buffers[load.input].is_output
        pos = 0
        for width in _vector_widths(len(load.names)):
            names = load.names[pos:pos + width]
            space, addr = self._address(load.input, Lin(flat.reg, flat.const + pos))
            opcode = f"ld.{space}{'.nc' if readonly else ''}{_vec(width)}.f32"
            if readonly and all(self.scopes.find(n) is None for n in names):
                regs = self.scopes.memo((opcode, addr), lambda: self._load(opcode, addr, width))
                for name, reg in zip(names, regs):
                    self.scopes.define(name, reg)
            else:
                self.b.emit(opcode, vector([self._register(n) for n in names]), addr)
            pos += width

    def _load(self, opcode: str, addr: str, width: int) -> list[str]:
        regs = [self.b.reg("f32") for _ in range(width)]
        self.b.emit(opcode, vector(regs), addr)
        return regs

    def visit_Write(self, write) -> None:
        """buf[i] = (v0, v1, ...) as vector stores, or one red.add per value when atomic."""
        _require_f32(write.value_dtype, write.output)
        if str(write.swizzle) != "NONE":
            raise CompileError(f"swizzled writes ({write.swizzle}) are not supported")
        values = [self._in_register(name) for name in write.values]
        flat = self._flat(write.output, write.index)
        widths = [1] * len(values) if write.atomic else _vector_widths(len(values))
        pos = 0
        for width in widths:
            space, addr = self._address(write.output, Lin(flat.reg, flat.const + pos))
            if write.atomic:
                self.b.emit(f"red.{space}.add.f32", addr, values[pos])
            else:
                self.b.emit(f"st.{space}{_vec(width)}.f32", addr, vector(values[pos:pos + width]))
            pos += width

    # ---- shared memory, barriers, cross-thread combines ------------------------------------

    def visit_Smem(self, smem) -> None:
        if _SMEM_DTYPES.get(smem.dtype) != "f32":
            raise CompileError(f"shared array {smem.name!r} of type {smem.dtype} is not supported")
        array = SharedArray(f"sh_{smem.name}", tuple(smem.extents), max(smem.align, _F32_BYTES))
        self.shared.setdefault(smem.name, array)

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
        _require_f32(ws.dtype, "WarpShuffle")
        mask = self.b.reg("b32")
        self.b.emit("activemask.b32", mask)
        distance = ws.length // 2
        while distance >= 1:
            self.b.comment(f"butterfly distance {distance}")
            with self.scopes.nested():
                for state, partner in zip(ws.state, ws.state_b):
                    self.b.emit("shfl.sync.bfly.b32", self._define_new(partner),
                                self._operand(state), distance, "0x1f", mask)
                self._stmts(ws.combine_states, annotate=False)
            distance //= 2

    def visit_TreeHalve(self, tree) -> None:
        """Fold `length` partials kept in shared memory (one per warp): at stride s = length/2,
        ..., 1 entry i < s combines entry i + s into itself; then every thread reads entry 0."""
        if tree.inner is not None:
            raise CompileError("TreeHalve with an inner tree is not supported")
        _require_f32(tree.dtype, "TreeHalve")
        tid = self.scopes.lookup(tree.tid_var)
        arrays = [self.shared[buf] for buf in tree.bufs]
        stride = tree.length // 2
        while stride >= 1:
            self.b.comment(f"tree stride {stride}")
            active = self.idx.compare("lt", tid, Lin(None, stride))
            (skip,) = self.b.labels("tree_skip")
            self.b.emit("bra", skip, guard="!" + active)
            with self.scopes.nested():
                mine = self.idx.shared_address
                for array, state, partner in zip(arrays, tree.state, tree.state_b):
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

    def _register(self, name: str) -> str:
        """The f32 register `name` is written to. Assigning a name that already exists (an
        accumulator, a reduction state) updates it in place, as in C; a new name gets a fresh
        register in the current scope."""
        value = self.scopes.find(name)
        if value is None:
            return self._define_new(name)
        if not (isinstance(value, str) and is_register(value)):
            raise CompileError(f"cannot assign to {name!r}")
        return value

    def _define_new(self, name: str) -> str:
        """A fresh register for `name`, shadowing any outer definition until the scope ends."""
        reg = self.b.reg("f32")
        self.scopes.define(name, reg)
        return reg

    def _in_register(self, name: str) -> str:
        operand = self._operand(name)
        if is_register(operand):
            return operand
        reg = self.b.reg("f32")
        self.b.emit("mov.f32", reg, operand)
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
            return "shared", self.idx.shared_address(self.shared[buffer].symbol, flat, _F32_BYTES)
        if self.kernel.buffers[buffer].dtype != "f32":
            raise CompileError(f"buffer {buffer!r} has dtype {self.kernel.buffers[buffer].dtype}")
        if buffer not in self.globals:
            raise CompileError(f"buffer {buffer!r} is not a kernel parameter")
        return "global", self.idx.global_address(self.globals[buffer], flat, _F32_BYTES)


# ---- helpers ---------------------------------------------------------------------------------

def _param_symbol(name: str) -> str:
    return f"param_{name}"


def _vec(width: int) -> str:
    return "" if width == 1 else f".v{width}"


def _vector_widths(n: int) -> list[int]:
    """Split n consecutive values into vector accesses (the IR only groups aligned runs)."""
    if n in (1, 2, 4):
        return [n]
    width = 4 if n % 4 == 0 else 2 if n % 2 == 0 else 1
    return [width] * (n // width)


def _is_float(dtype) -> bool:
    return getattr(dtype, "name", dtype) in ("f32", "float")


def _require_f32(dtype, what: str) -> None:
    if dtype is not None and not _is_float(dtype):
        raise CompileError(f"{what}: dtype {getattr(dtype, 'name', dtype)} is not supported (f32 only)")


def _straight_line(body) -> Iterator:
    """The statements of a body, descending into branches but not into nested loops."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, Cond):
            yield from _straight_line(stmt.body)
            yield from _straight_line(stmt.else_body)


def _names_used(node) -> set[str]:
    """Every name an IR subtree refers to: its Vars, plus TreeHalve's tid variable (a string)."""
    found: set[str] = set()
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, Var):
            found.add(item.name)
        elif isinstance(item, TreeHalve):
            found.add(item.tid_var)
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            stack.extend(getattr(item, f.name) for f in dataclasses.fields(item))
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
    return found
