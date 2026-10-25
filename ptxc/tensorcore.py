"""Tensor cores: register fragments, ldmatrix, mma.sync and fragment stores (m16n8k16).

One `mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}` multiplies a 16x16 A tile by a 16x8 B
tile into a 16x8 f32 C tile, the three spread over the 32 lanes of a warp as register
fragments: per lane, A is 4 b32 registers (8 packed 16-bit values), B is 2 and C is 4 f32.
Lane l holds C's rows g and g + 8, columns 2t and 2t + 1, where g = l / 4 and t = l % 4.

A fragment array is a list of such register sets, indexed by a constant: the loops that walk
it (`for _r0 in 0..2`) are always unrolled (lower.py). A shared tile is read with
`ldmatrix`, each lane addressing one 16-byte row of an 8x8 matrix; a tile left in global
memory is read element by element in the same lane map, clamping rows past M or N to the
last one and zero-filling K past its end (an mma sums over K, so a duplicate would count).
Shared tiles are swizzled: element e lives at e ^ (((e >> s) & m) << 3), the XOR that moves
16-byte chunks across banks, applied on the fill (cp.async) and on every ldmatrix read alike.
"""

from __future__ import annotations

import re

from . import dtypes as dt
from .errors import CompileError
from .facts import Fact
from .index import Lin

# swizzle name -> (shift, mask) of the bits XORed into bits 3.. of a 16-bit element index
_SWIZZLES = {"B32": (6, 1), "B64": (6, 3), "B128": (6, 7), "B128@7": (7, 7)}
_NREGS = {"a": 4, "b": 2, "c": 4}


class TensorCores:
    """visit_ methods of KernelLowering for the tensor-core statements."""

    def visit_RegFragment(self, frag) -> None:
        if tuple(frag.shape) != (16, 8, 16):
            raise CompileError(f"fragment {frag.name}: shape {frag.shape} is not supported (m16n8k16 only)")
        n = frag.nregs or _NREGS[frag.role]
        cls = "f32" if frag.role == "c" else "b32"
        regs = [[self.b.reg(cls) for _ in range(n)] for _ in range(frag.count)]
        if frag.role == "c":  # C starts at zero
            for r in (r for f in regs for r in f):
                self.b.emit("mov.f32", r, dt.immediate(0.0, dt.F32))
        self._frags[frag.name] = regs

    def _fragment(self, ref: str) -> list[str]:
        """The registers of a fragment reference: `_a0`, `_b[1]`, `_c0_[_r0]` (a trip constant)."""
        m = re.fullmatch(r"(\w+)(?:\[(\w+)\])?", ref)
        if m is None or m.group(1) not in self._frags:
            raise CompileError(f"unknown fragment {ref!r}")
        regs = self._frags[m.group(1)]
        index = m.group(2)
        if index is None:
            k = 0
        elif index.isdigit():
            k = int(index)
        else:
            v = self.scopes.lookup(index)
            if not isinstance(v, Lin) or v.reg is not None:
                raise CompileError(f"fragment index {index!r} is not a constant")
            k = v.const
        return regs[k]

    def visit_MmaSyncPtx(self, mma) -> None:
        if tuple(mma.shape) != (16, 8, 16) or mma.c_dtype != "f32" or mma.ab_dtype not in ("f16", "bf16"):
            raise CompileError(f"mma {mma.shape} {mma.ab_dtype}->{mma.c_dtype} is not supported")
        if mma.b_row_major or mma.sfa_frag or mma.sfb_frag:
            raise CompileError("row-major B and block-scaled mma are not supported")
        c, a, b = (self._fragment(f) for f in (mma.c_frag, mma.a_frag, mma.b_frag))
        t = mma.ab_dtype
        self.b.emit(f"mma.sync.aligned.m16n8k16.row.col.f32.{t}.{t}.f32",
                    _regs(c), _regs(a), _regs(b), _regs(c))

    # ---- fragment loads ----------------------------------------------------------------------

    def visit_LdmatrixLoad(self, ld) -> None:
        if ld.byte_slab or ld.scale_buffer or ld.fragment_layout != "m16n8k16":
            raise CompileError("byte-slab and block-scaled fragment loads are not supported")
        regs = self._fragment(ld.frag)
        pair = self._fragment(ld.pair_frag) if ld.pair_frag else None
        if ld.staged:
            self._ldmatrix(ld, regs, pair)
        else:
            if pair is not None:
                raise CompileError("a paired fragment load from global memory is not supported")
            self._fragment_from_global(ld, regs)

    def _ldmatrix(self, ld, regs: list[str], pair: list[str] | None) -> None:
        """One ldmatrix: each lane gives the address of one 8-element row of an 8x8 matrix."""
        array = self.shared.get(ld.src_buffer)
        if array is None or array.dtype.nbytes != 2:
            raise CompileError(f"ldmatrix from {ld.src_buffer!r}: a 16-bit shared array is needed")
        lane = self._lane()
        if ld.role == "a":  # rows 0-15 at k 0 (lanes 0-15), then at k 8: a0..a3
            offset, count, trans = self._lane_rows(lane % 16, ld.ldm, lane // 16), 4, False
        elif not ld.b_trans and pair is not None:  # K rows of two column-adjacent B fragments
            offset, count, trans = self._lane_rows(lane % 16, ld.ldm, lane // 16), 4, True
        elif not ld.b_trans:  # the 16 K rows of one B fragment, transposed into place
            offset, count, trans = self._lane_rows(lane % 16, ld.ldm, None), 2, True
        elif pair is None:  # B stored N x K (x @ w.T): its rows are the fragment's columns
            offset, count, trans = self._lane_rows(lane % 8, ld.ldm, (lane // 8) % 2), 2, False
        else:
            raise CompileError("a paired ldmatrix of an N x K operand is not supported")
        element = self.idx.add(self._flat(ld.src_buffer, ld.src_index), offset)
        element = self._swizzle(element, ld.swizzle)
        _, addr = self._address(ld.src_buffer, element)
        dst = regs + (pair or []) if count == 4 and pair is not None else regs
        if len(dst) != count:
            raise CompileError(f"ldmatrix x{count} into a fragment of {len(dst)} registers")
        self.b.emit(f"ldmatrix.sync.aligned.m8n8.x{count}{'.trans' if trans else ''}.shared.b16",
                    _regs(dst), addr)

    def _lane_rows(self, row, ldm: int, col8) -> Lin:
        """row * ldm + col8 * 8, the element a lane addresses (row, col8: _Lane expressions)."""
        offset = self.idx.mul(row.lin, Lin(None, ldm))
        return offset if col8 is None else self.idx.add(offset, self.idx.mul(col8.lin, Lin(None, 8)))

    def _fragment_from_global(self, ld, regs: list[str]) -> None:
        """A fragment read straight from global memory, two 16-bit values per register, in
        mma's lane map. Rows (A) or columns (B) past the guard read the last valid one; K
        past k_zero reads 0."""
        d = self._buffer_dtype(ld.src_buffer)
        if d.nbytes != 2:
            raise CompileError(f"fragment from {ld.src_buffer!r}: a 16-bit buffer is needed")
        lane = self._lane()
        g, t2 = (lane // 4).lin, self.idx.mul((lane % 4).lin, Lin(None, 2))
        base = self._flat(ld.src_buffer, ld.src_index)
        ldm = ld.ldm
        last = None
        if ld.gmem_guard is not None:  # in-range rows (or columns) from the tile base, minus one
            coord, bound = (self.idx.value(e) for e in ld.gmem_guard)
            last = self.idx.add(self.idx.add(bound, self.idx.neg(coord)), Lin(None, -1))
        k_left = None
        if ld.k_zero is not None:
            coord, bound = (self.idx.value(e) for e in ld.k_zero)
            k_left = self.idx.add(bound, self.idx.neg(coord))
        clamp = (lambda v: v) if last is None else (lambda v: self._clamp(v, last))
        pairs = []
        if ld.role == "a":
            for i in range(4):  # row g (+8 for odd i), k 2t (+8 for i >= 2)
                row = clamp(self.idx.add(g, Lin(None, 8 * (i & 1))))
                k = self.idx.add(t2, Lin(None, 8 * (i >> 1)))
                first = self.idx.add(base, self.idx.add(self.idx.mul(row, Lin(None, ldm)), k))
                pairs.append(((first, k), (self.idx.add(first, Lin(None, 1)), self.idx.add(k, Lin(None, 1)))))
        else:
            n = clamp(g)
            for i in range(2):  # column g, k 2t (+8 for i = 1)
                k = self.idx.add(t2, Lin(None, 8 * i))
                if ld.b_trans:  # B stored N x K
                    first = self.idx.add(base, self.idx.add(self.idx.mul(n, Lin(None, ldm)), k))
                    second = self.idx.add(first, Lin(None, 1))
                else:
                    first = self.idx.add(base, self.idx.add(self.idx.mul(k, Lin(None, ldm)), n))
                    second = self.idx.add(first, Lin(None, ldm))
                pairs.append(((first, k), (second, self.idx.add(k, Lin(None, 1)))))
        for reg, halves in zip(regs, pairs):
            h = [self._half_from_global(ld.src_buffer, element, k, k_left) for element, k in halves]
            self.b.emit("mov.b32", reg, "{" + f"{h[0]}, {h[1]}" + "}")

    def _half_from_global(self, buffer: str, element: Lin, k: Lin, k_left: Lin | None) -> str:
        h = self.b.reg("b16")
        _, addr = self._address(buffer, element)
        live = True if k_left is None else self.idx.compare("lt", k, k_left)
        if live is True:
            self.b.emit("ld.global.nc.b16", h, addr)
        else:
            self.b.emit("mov.b16", h, "0")
            if live is not False:
                self.b.emit("ld.global.nc.b16", h, addr, guard=live)
        return h

    def _clamp(self, v: Lin, last: Lin) -> Lin:
        """min(v, max(last, 0)): a row past the operand's end reads its last row."""
        top = self.idx.derive("b32", "max.s32", self.idx.reg(last), 0)
        out = self.idx.derive("b32", "min.s32", self.idx.reg(v), top)
        self.idx.assume(out, Fact(0, self.idx.fact(v).hi))
        return Lin(out)

    # ---- fragment stores ---------------------------------------------------------------------

    def visit_RegStore(self, st) -> None:
        """Store a C fragment: lane (g, t) writes rows g, g + 8 and columns 2t, 2t + 1 of the
        16x8 tile at dst_index, after the optional epilogue, converted to the buffer's dtype."""
        if st.atomic or st.volta_interleaved or str(st.swizzle) != "NONE" or st.extra_frags:
            raise CompileError("atomic, swizzled, interleaved or multi-fragment stores are not supported")
        if st.fragment_layout != "m16n8k16":
            raise CompileError(f"fragment layout {st.fragment_layout} is not supported")
        regs = self._fragment(st.frag)
        d = self._buffer_dtype(st.dst_buffer)
        shape = self.kernel.buffers[st.dst_buffer].shape
        coords = [self.idx.value(e) for e in st.dst_index]
        rdim = st.row_dim if st.row_dim is not None else len(coords) - 2
        cdim = st.col_dim if st.col_dim is not None else len(coords) - 1
        lane = self._lane()
        g, t2 = (lane // 4).lin, self.idx.mul((lane % 4).lin, Lin(None, 2))
        m_bound = n_bound = None
        if st.m_guard is not None:
            m_bound = [self.idx.value(e) for e in st.m_guard]
        if st.n_guard is not None:
            n_bound = [self.idx.value(e) for e in st.n_guard]
        for half, (r_off, values) in enumerate(((g, regs[0:2]), (self.idx.add(g, Lin(None, 8)), regs[2:4]))):
            cells = []
            for j, value in enumerate(values):
                c_off = self.idx.add(t2, Lin(None, j))
                if st.epilogue is not None:
                    value = self._epilogue(st.epilogue, value, r_off, c_off, d)
                else:
                    value = dt.convert(self.b, value, dt.F32, dt.store_type(d))
                where = list(coords)
                where[rdim] = self.idx.add(where[rdim], r_off)
                where[cdim] = self.idx.add(where[cdim], c_off)
                guard = self._store_guard(m_bound, r_off, n_bound, c_off)
                cells.append((self._flat_lin(where, shape), dt.to_memory(self.b, value, d), guard))
            self._store_pair(st.dst_buffer, cells, d)

    def _store_guard(self, m_bound, r_off: Lin, n_bound, c_off: Lin) -> bool | str:
        guard: bool | str = True
        for bound, off in ((m_bound, r_off), (n_bound, c_off)):
            if bound is None:
                continue
            p = self.idx.compare("lt", self.idx.add(bound[0], off), bound[1])
            if p is False:
                return False
            if p is not True:
                if guard is True:
                    guard = p
                else:
                    both = self.b.reg("pred")
                    self.b.emit("and.pred", both, guard, p)
                    guard = both
        return guard

    def _store_pair(self, buffer: str, cells: list, d: dt.DType) -> None:
        """Two column-adjacent values: one vector store when they share a guard and the first
        one's address is aligned to the pair, else one store each."""
        (first, a, ga), (_, b, gb) = cells
        space, addr = self._address(buffer, first)
        if ga == gb and d.nbytes in (2, 4, 8) and self.idx.fact(first).div % 2 == 0:
            if ga is not False:
                self.b.emit(f"st.{space}.v2.{d.mem}", addr, "{" + f"{a}, {b}" + "}",
                            guard=None if ga is True else ga)
            return
        for flat, value, guard in cells:
            if guard is False:
                continue
            space, addr = self._address(buffer, flat)
            self.b.emit(f"st.{space}.{d.mem}", addr, value, guard=None if guard is True else guard)

    def _epilogue(self, fn, value: str, r_off: Lin, c_off: Lin, d: dt.DType) -> str:
        """The store's pointwise tail on one element: its parameters are the element (f32),
        then __M__ / __N__ (its row and column in the tile) and names already in scope."""
        with self.scopes.nested():
            self.scopes.define(fn.params[0], value)
            self._types[fn.params[0]] = dt.F32
            for p in fn.params[1:]:
                if p == "__N__":
                    self.scopes.define(p, c_off)
                elif p == "__M__":
                    self.scopes.define(p, r_off)
            self._stmts(fn.body, annotate=False)
            if len(fn.results) != 1:
                raise CompileError("an epilogue with several results is not supported")
            return self._value(fn.results[0], dt.store_type(d))

    def _flat_lin(self, coords: list[Lin], shape: tuple[int, ...]) -> Lin:
        flat, stride = Lin(None, 0), 1
        for c, extent in reversed(list(zip(coords, shape))):
            flat = self.idx.add(flat, self.idx.mul(c, Lin(None, stride)))
            stride *= extent
        return flat

    # ---- lanes and swizzles ------------------------------------------------------------------

    def _lane(self) -> _Lane:
        """This thread's lane within its warp, %tid.x & 31."""
        tid = self.idx.derive("b32", "mov.u32", "%tid.x")
        reg = self.idx.derive("b32", "and.b32", tid, 31)
        self.idx.assume(reg, Fact(0, 31))
        return _Lane(self.idx, Lin(reg))

    def _swizzle(self, element: Lin, swizzle) -> Lin:
        """The physical index of logical 16-bit element `element` in a swizzled shared array."""
        name = str(swizzle)
        if name == "NONE":
            return element
        if name not in _SWIZZLES:
            raise CompileError(f"swizzle {name} is not supported")
        shift, mask = _SWIZZLES[name]
        if element.reg is None:
            e = element.const
            return Lin(None, e ^ (((e >> shift) & mask) << 3))
        # A register that is a multiple of 2^(shift + 3) leaves the XORed bits to the constant.
        if self.idx.fact(Lin(element.reg)).div % (1 << (shift + 3)) == 0 and element.const >= 0:
            e = element.const
            return Lin(element.reg, e ^ (((e >> shift) & mask) << 3))
        e = self.idx.reg(element)
        bits = self.idx.derive("b32", "bfe.u32", e, shift, mask.bit_length())
        moved = self.idx.derive("b32", "shl.b32", bits, 3)
        return Lin(self.idx.derive("b32", "xor.b32", e, moved))


class _Lane:
    """The lane, or an expression of it, written as lane % 16, lane // 8, ... (Lin underneath)."""

    def __init__(self, idx, lin: Lin) -> None:
        self.idx, self.lin = idx, lin

    def __mod__(self, k: int) -> _Lane:
        return _Lane(self.idx, self.idx.arith("%", self.lin, Lin(None, k)))

    def __floordiv__(self, k: int) -> _Lane:
        return _Lane(self.idx, self.idx.arith("/", self.lin, Lin(None, k)))


def _regs(regs: list[str]) -> str:
    return "{" + ", ".join(regs) + "}"


FRAGMENT_FIELDS = ("frag", "pair_frag", "c_frag", "a_frag", "b_frag")
