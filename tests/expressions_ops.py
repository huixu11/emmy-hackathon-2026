"""Ops that no other test uses: the data movement emmy traces (cat, stack, roll, transpose,
expand, repeat_interleave, tril/triu, chunk), indexing by computed integers (index_select,
embedding, gather), conv1d, and the comparisons and bool combines the other problems leave out
(>=, <, ==, !=, &, |), in f32, f16, f64 and i32 and inside reductions.

Left out because emmy's own trace computes something else, so its reference fails as well:
`torch.cat` along dim 0 stops in the tracer, `torch.arange` and `torch.eye` come back as the
input they are combined with (`x * arange(n)` traces as `x * x`), and a 3-d `permute` traces as
a reshape.

    python tools/generate.py tests/expressions_ops.py --emmy emmy --out tests/problems

As in tests/expressions_variants.py, `x = torch.randn(...)` only declares an input; the
computation is the last expression.
"""
import numpy as np

f32 = lambda a: np.asarray(a).astype(np.float32)
f64 = lambda a: np.asarray(a).astype(np.float64)


def _sig(x):
    return 1 / (1 + np.exp(-f64(x)))


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _conv1d(x, w):  # x: (1, C, L), w: (O, C, K), no padding, stride 1
    k = w.shape[2]
    cols = np.stack([f64(x)[0, :, i:i + k] for i in range(x.shape[2] - k + 1)], axis=-1)  # (C, K, L')
    return np.einsum("ock,ckl->ol", f64(w), cols)[None]


OPEN = [
    # ---- data movement --------------------------------------------------------------------
    dict(id="u_p_cat_cols", code="a = torch.randn(64, 32); b = torch.randn(64, 48); torch.cat([a, b], dim=1) * 2.0",
         ref=lambda a, b: [f32(np.concatenate([a, b], 1) * 2)], rtol=0, atol=0),
    dict(id="u_p_stack", code="a = torch.randn(64, 32); b = torch.randn(64, 32); torch.stack([a, b], dim=1) * 3.0",
         ref=lambda a, b: [f32(np.stack([a, b], 1) * f32(3))], rtol=0, atol=0),
    dict(id="u_p_roll", code="x = torch.randn(4096); torch.roll(x, 7, 0) - x",
         ref=lambda x: [f32(np.roll(x, 7) - x)], rtol=0, atol=0),
    dict(id="u_p_roll_2d", code="x = torch.randn(64, 64); torch.roll(x, -5, 1) * 2.0",
         ref=lambda x: [f32(np.roll(x, -5, 1) * 2)], rtol=0, atol=0),
    dict(id="u_p_transpose_3d", code="x = torch.randn(8, 16, 32); x.transpose(0, 2).contiguous() + 1.0",
         ref=lambda x: [f32(np.transpose(x, (2, 1, 0)) + f32(1))], rtol=0, atol=0),
    dict(id="u_p_expand", code="x = torch.randn(64, 1, 32); y = torch.randn(64, 16, 32); x.expand(64, 16, 32) * y",
         ref=lambda x, y: [f32(np.broadcast_to(x, (64, 16, 32)) * y)], rtol=0, atol=0),
    dict(id="u_p_repeat_interleave", code="x = torch.randn(128, 32); torch.repeat_interleave(x, 2, dim=0) + 1.0",
         ref=lambda x: [f32(np.repeat(x, 2, axis=0) + f32(1))], rtol=0, atol=0),
    dict(id="u_p_tril", code="x = torch.randn(64, 64); torch.tril(x) * 2.0",
         ref=lambda x: [f32(np.tril(x) * 2)], rtol=0, atol=0),
    dict(id="u_p_triu_diag", code="x = torch.randn(48, 80); torch.triu(x, diagonal=1) + 1.0",
         ref=lambda x: [f32(np.triu(x, 1) + f32(1))], rtol=0, atol=0),
    dict(id="u_p_chunk_glu", code="x = torch.randn(64, 128); x.chunk(2, dim=1)[0] * torch.sigmoid(x.chunk(2, dim=1)[1])",
         ref=lambda x: [f32(f64(x[:, :64]) * _sig(x[:, 64:]))], rtol=1e-5, atol=1e-6),
    # ---- indexing by computed integers --------------------------------------------------
    dict(id="u_p_index_select", code="w = torch.randn(8, 32); x = torch.randn(64); torch.index_select(w, 0, (x > 0).long())",
         ref=lambda w, x: [f32(w[(x > 0).astype(np.int64)])], rtol=0, atol=0),
    dict(id="u_p_embedding",  # the reference takes the inputs in emmy's order, here x first
         code="w = torch.randn(16, 64); x = torch.randn(256); torch.nn.functional.embedding((x > 0).long() + (x > 1).long(), w)",
         ref=lambda x, w: [f32(w[(x > 0).astype(np.int64) + (x > 1).astype(np.int64)])], rtol=0, atol=0),
    dict(id="u_p_gather", code="x = torch.randn(64, 32); torch.gather(x, 1, (x > 0).long())",
         ref=lambda x: [f32(np.take_along_axis(x, (x > 0).astype(np.int64), 1))], rtol=0, atol=0),
    # ---- convolution --------------------------------------------------------------------
    dict(id="u_m_conv1d", code="x = torch.randn(1, 4, 64); w = torch.randn(8, 4, 3); torch.nn.functional.conv1d(x, w)",
         ref=lambda x, w: [f32(_conv1d(x, w))], rtol=1e-4, atol=1e-4),
    # ---- comparisons and bool combines ----------------------------------------------------
    dict(id="u_p_ge_where", code="x = torch.randn(4096); y = torch.randn(4096); torch.where(x >= y, x - y, y - x)",
         ref=lambda x, y: [f32(np.where(x >= y, x - y, y - x))], rtol=0, atol=0),
    dict(id="u_p_lt_scalar", code="x = torch.randn(4096); torch.where(x < 0.5, x * 2.0, -x)",
         ref=lambda x: [f32(np.where(x < 0.5, x * f32(2), -x))], rtol=0, atol=0),
    dict(id="u_p_and_mask", code="x = torch.randn(4096); y = torch.randn(4096); torch.where((x > 0) & (y < 0), x, y)",
         ref=lambda x, y: [f32(np.where((x > 0) & (y < 0), x, y))], rtol=0, atol=0),
    dict(id="u_p_or_mask", code="x = torch.randn(4096); torch.where((x > 1) | (x < -1), x, x * 0.5)",
         ref=lambda x: [f32(np.where((x > 1) | (x < -1), x, x * f32(0.5)))], rtol=0, atol=0),
    dict(id="u_p_i32_eq", code="x = torch.randint(-3, 4, (4096,), dtype=torch.int32); torch.where(x == 0, x + 5, x)",
         ref=lambda x: [np.where(x == 0, x + 5, x).astype(np.int64)], rtol=0, atol=0),
    dict(id="u_p_i32_ne", code="x = torch.randint(-3, 4, (4096,), dtype=torch.int32); torch.where(x != 1, x * 2, x)",
         ref=lambda x: [np.where(x != 1, x * 2, x).astype(np.int64)], rtol=0, atol=0),
    dict(id="u_p_f16_leaky", code="x = torch.randn(4096, dtype=torch.float16); torch.where(x >= 0, x, x * 0.25)",
         ref=lambda x: [np.where(x >= 0, x, (f32(x) * f32(0.25)).astype(np.float16))], rtol=0, atol=0),
    dict(id="u_r_masked_sum", code="x = torch.randn(64, 256); (x * (x > 0)).sum(-1)",
         ref=lambda x: [f32((f64(x) * (x > 0)).sum(-1))], rtol=1e-5, atol=1e-5),
    dict(id="u_r_f64_masked_sum", code="x = torch.randn(64, 256, dtype=torch.float64); (x * (x < 0.5)).sum(-1)",
         ref=lambda x: [(f64(x) * (x < 0.5)).sum(-1)], rtol=1e-5, atol=1e-5),
    dict(id="u_f_masked_softmax", code="x = torch.randn(64, 128); torch.softmax(x.masked_fill(x < -1.0, -1e9), dim=-1)",
         ref=lambda x: [f32(_softmax(np.where(x < -1, -1e9, f64(x))))], rtol=1e-5, atol=1e-6),
]
