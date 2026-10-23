"""Hidden-set-style variants of the five problem categories: new shapes and ops, same concepts.

The generated problems live in tests/problems/ (run them with tools/run_problems.py --dir
tests/problems). To regenerate them, with torch installed:

    python tools/generate.py tests/expressions_variants.py --emmy emmy --out tests/problems

`x = torch.randn(...)` in a spec only declares the input x (it is fed plain seeded randn data,
and any arithmetic on that line is dropped), so all computation goes in the final expression.
"""
import math

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"
f64 = lambda a: a.astype(np.float64)
f32 = lambda a: np.asarray(a).astype(np.float32)
erf = np.vectorize(math.erf)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _ln(x, w, b, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * w + b


OPEN = [

    # ---- pointwise --------------------------------------------------------
    dict(id="v_p_add_odd", code="torch.randn(1000) + torch.randn(1000)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="v_p_bcast_row", code="x = torch.randn(64, 96); b = torch.randn(96); x * b + 1.5",
         ref=lambda x, b: [f32(f64(x) * f64(b) + 1.5)], rtol=1e-6, atol=1e-6),
    dict(id="v_p_bcast_col", code="x = torch.randn(48, 80); c = torch.randn(48, 1); x - c",
         ref=lambda x, c: [x - c], rtol=0, atol=0),
    dict(id="v_p_exp", code="torch.exp(torch.randn(4096))",
         ref=lambda x: [f32(np.exp(f64(x)))], rtol=1e-5, atol=1e-6),
    dict(id="v_p_tanh", code="torch.tanh(torch.randn(3000) * 2.0)",
         ref=lambda x: [f32(np.tanh(f64(x) * 2.0))], rtol=1e-5, atol=1e-6),
    dict(id="v_p_sigmoid", code="torch.sigmoid(torch.randn(2048)) * 3.0",
         ref=lambda x: [f32(_sig(f64(x)) * 3.0)], rtol=1e-5, atol=1e-6),
    dict(id="v_p_gelu", code="torch.nn.functional.gelu(torch.randn(4096))",
         ref=lambda x: [f32(0.5 * f64(x) * (1 + erf(f64(x) / math.sqrt(2))))], rtol=1e-5, atol=1e-6),
    dict(id="v_p_gelu_tanh", code="torch.nn.functional.gelu(torch.randn(2048), approximate='tanh')",
         ref=lambda x: [f32(0.5 * f64(x) * (1 + np.tanh(math.sqrt(2 / math.pi) * (f64(x) + 0.044715 * f64(x) ** 3))))],
         rtol=1e-5, atol=1e-6),
    dict(id="v_p_silu", code="torch.nn.functional.silu(torch.randn(4096))",
         ref=lambda x: [f32(f64(x) * _sig(f64(x)))], rtol=1e-5, atol=1e-6),
    dict(id="v_p_softplus", code="torch.nn.functional.softplus(torch.randn(2048))",
         ref=lambda x: [f32(np.logaddexp(0, f64(x)))], rtol=1e-5, atol=1e-6),
    dict(id="v_p_clamp", code="torch.clamp(torch.randn(2048), -0.5, 0.75)",
         ref=lambda x: [np.clip(x, -0.5, 0.75)], rtol=0, atol=0),
    dict(id="v_p_pow3", code="torch.randn(2048) ** 3",
         ref=lambda x: [f32(f64(x) ** 3)], rtol=1e-5, atol=1e-6),
    dict(id="v_p_div_abs", code="x = torch.randn(2048); y = torch.randn(2048); x / (torch.abs(y) + 1.0)",
         ref=lambda x, y: [f32(f64(x) / (np.abs(f64(y)) + 1.0))], rtol=1e-6, atol=1e-6),
    dict(id="v_p_where", code="x = torch.randn(2048); torch.where(x > 0, x, x * 0.1)",
         ref=lambda x: [f32(np.where(x > 0, f64(x), f64(x) * 0.1))], rtol=1e-6, atol=1e-6),
    dict(id="v_p_sqrt_abs", code="torch.sqrt(torch.abs(torch.randn(2048)))",
         ref=lambda x: [f32(np.sqrt(np.abs(f64(x))))], rtol=1e-6, atol=1e-6),
    dict(id="v_p_rsqrt", code="x = torch.randn(2048); torch.rsqrt(x * x + 1.0)",
         ref=lambda x: [f32(1 / np.sqrt(f64(x) ** 2 + 1))], rtol=1e-5, atol=1e-6),
    dict(id="v_p_log", code="x = torch.randn(2048); torch.log(x * x + 1.0)",
         ref=lambda x: [f32(np.log(f64(x) ** 2 + 1))], rtol=1e-5, atol=1e-6),
    dict(id="v_p_max2", code="x = torch.randn(1536); y = torch.randn(1536); torch.maximum(x, y) - torch.minimum(x, y)",
         ref=lambda x, y: [np.maximum(x, y) - np.minimum(x, y)], rtol=0, atol=0),
    # ---- reductions -------------------------------------------------------
    dict(id="v_r_sum_odd", code="torch.randn(3000).sum()",
         ref=lambda x: [f32(f64(x).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="v_r_mean_rows", code="torch.randn(16, 1000).mean(-1)",
         ref=lambda x: [f32(f64(x).mean(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="v_r_sumsq", code="x = torch.randn(4096); (x * x).sum()",
         ref=lambda x: [f32((f64(x) ** 2).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="v_r_colsum", code="torch.randn(256, 64).sum(0)",
         ref=lambda x: [f32(f64(x).sum(0))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="v_r_rowmax_wide", code="torch.randn(8, 4096).amax(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    # ---- matmul -------------------------------------------------------------
    dict(id="v_m_256x128x256", code="torch.randn(256, 128) @ torch.randn(128, 256)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="v_m_64", code="torch.randn(64, 64) @ torch.randn(64, 64)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="v_m_odd", code="torch.randn(96, 72) @ torch.randn(72, 80)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="v_m_tall", code="torch.randn(1024, 32) @ torch.randn(32, 128)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="v_m_deepk", code="torch.randn(64, 1024) @ torch.randn(1024, 64)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="v_m_bias_relu", code="x = torch.randn(128, 64); w = torch.randn(64, 96); b = torch.randn(96); torch.relu(x @ w + b)",
         ref=lambda x, w, b: [f32(np.maximum(f64(x) @ f64(w) + f64(b), 0))], rtol=1e-3, atol=1e-3),
    # ---- fused --------------------------------------------------------------
    dict(id="v_f_layernorm", code="torch.nn.functional.layer_norm(torch.randn(32, 512), (512,))",
         ref=lambda x: [f32((f64(x) - f64(x).mean(-1, keepdims=True)) / np.sqrt(f64(x).var(-1, keepdims=True) + 1e-5))],
         rtol=1e-3, atol=1e-4),
    dict(id="v_f_softmax_wide", code="torch.softmax(torch.randn(16, 1024) * 2.0, dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x) * 2.0))], rtol=1e-3, atol=1e-5),
    dict(id="v_f_rmsnorm_w", code="x = torch.randn(32, 768); w = torch.randn(768); x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=lambda x, w: [f32(f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-6) * f64(w))],
         rtol=1e-3, atol=1e-4),
    dict(id="v_f_center", code="x = torch.randn(64, 256); x - x.mean(-1, keepdim=True)",
         ref=lambda x: [f32(f64(x) - f64(x).mean(-1, keepdims=True))], rtol=1e-3, atol=1e-4),
    # ---- attention-like -----------------------------------------------------
    dict(id="v_a_32x64", code="q = torch.randn(32, 64); kt = torch.randn(64, 32); v = torch.randn(32, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(f64(q) @ f64(kt) / 8.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="v_a_256x16", code="q = torch.randn(256, 16); kt = torch.randn(16, 256); v = torch.randn(256, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(f64(q) @ f64(kt) / 4.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="v_a_rect", code="q = torch.randn(64, 32); kt = torch.randn(32, 128); v = torch.randn(128, 48); torch.softmax(q @ kt * 0.17, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(f64(q) @ f64(kt) * 0.17) @ f64(v))], rtol=1e-3, atol=1e-4),
    # ---- batch 2: unpinned schedules, bigger shapes, batched and chained kernels ----

    # reductions without pinned knobs: emmy picks the schedule (maybe split + atomics)
    dict(id="w_r_sum_unpinned", code="torch.randn(65536).sum()",
         ref=lambda x: [f32(f64(x).sum())], rtol=1e-3, atol=1e-3),
    dict(id="w_r_rows_unpinned", code="torch.randn(512, 512).sum(-1)",
         ref=lambda x: [f32(f64(x).sum(-1))], rtol=1e-4, atol=1e-4),
    dict(id="w_r_amax_rows", code="torch.randn(100, 300).amax(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0),
    # matmul
    dict(id="w_m_bmm", code="torch.randn(4, 64, 32) @ torch.randn(4, 32, 64)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="w_m_linear", code="x = torch.randn(32, 256); w = torch.randn(256, 512); b = torch.randn(512); x @ w + b",
         ref=lambda x, w, b: [f32(f64(x) @ f64(w) + f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="w_m_skinny", code="torch.randn(8, 2048) @ torch.randn(2048, 16)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    # fused / chained
    dict(id="w_f_softmax_dim0", code="torch.softmax(torch.randn(256, 32), dim=0)",
         ref=lambda x: [f32(_softmax(f64(x), axis=0))], rtol=1e-3, atol=1e-5),
    dict(id="w_f_gelu_mlp", code="x = torch.randn(64, 128); w = torch.randn(128, 256); b = torch.randn(256); torch.nn.functional.gelu(x @ w + b)",
         ref=lambda x, w, b: [f32((lambda h: 0.5 * h * (1 + erf(h / math.sqrt(2))))(f64(x) @ f64(w) + f64(b)))],
         rtol=1e-3, atol=1e-3),
    dict(id="w_f_swiglu", code="x = torch.randn(32, 64); w1 = torch.randn(64, 128); w2 = torch.randn(64, 128); torch.nn.functional.silu(x @ w1) * (x @ w2)",
         ref=lambda x, w1, w2: [f32((lambda a, b: a * _sig(a) * b)(f64(x) @ f64(w1), f64(x) @ f64(w2)))],
         rtol=1e-3, atol=1e-3),
    dict(id="w_f_layernorm_affine", code="x = torch.randn(48, 384); w = torch.randn(384); b = torch.randn(384); torch.nn.functional.layer_norm(x, (384,), w, b)",
         ref=lambda x, w, b: [f32(_ln(f64(x), f64(w), f64(b)))], rtol=1e-3, atol=1e-4),
    dict(id="w_f_softmax_temp", code="x = torch.randn(32, 512); torch.softmax(x / 0.7, dim=-1) * 2.0",
         ref=lambda x: [f32(_softmax(f64(x) / 0.7) * 2.0)], rtol=1e-3, atol=1e-5),
    # attention
    dict(id="w_a_512x64", code="q = torch.randn(512, 64); kt = torch.randn(64, 512); v = torch.randn(512, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(f64(q) @ f64(kt) / 8.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="w_a_sdpa", code="q = torch.randn(1, 2, 64, 32); k = torch.randn(1, 2, 64, 32); v = torch.randn(1, 2, 64, 32); torch.nn.functional.scaled_dot_product_attention(q, k, v)",
         ref=lambda q, k, v: [f32(_softmax(f64(q) @ np.swapaxes(f64(k), -1, -2) / math.sqrt(32)) @ f64(v))],
         rtol=1e-3, atol=1e-4),
    # pointwise
    dict(id="w_p_3d", code="torch.randn(4, 16, 32) * torch.randn(32)",
         ref=lambda x, w: [x * w], rtol=0, atol=0),
    dict(id="w_p_chain", code="x = torch.randn(4096); torch.tanh(x * 2.0 + 1.0) * x",
         ref=lambda x: [f32(np.tanh(f64(x) * 2.0 + 1.0) * f64(x))], rtol=1e-5, atol=1e-6),
    dict(id="w_p_two_out_like", code="x = torch.randn(64, 64); y = torch.randn(64, 64); (x + y) * (x - y)",
         ref=lambda x, y: [f32((f64(x) + f64(y)) * (f64(x) - f64(y)))], rtol=1e-6, atol=1e-6),
    # ---- batch 3: odd sizes, 3-D shapes, composite activations, normalizers, more attention ----

    # pointwise
    dict(id="x_p_fma_5000", code="x = torch.randn(5000); y = torch.randn(5000); z = torch.randn(5000); x * y + z",
         ref=lambda x, y, z: [x * y + z], rtol=0, atol=0),
    dict(id="x_p_outer_bcast", code="torch.randn(8, 1, 64) + torch.randn(1, 32, 64)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="x_p_scalar_bcast", code="torch.randn(1) * torch.randn(1000)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="x_p_div_const", code="torch.randn(64, 33) / 3.0",
         ref=lambda x: [f32(f64(x) / 3.0)], rtol=1e-6, atol=1e-7),
    dict(id="x_p_elu", code="x = torch.randn(3000); torch.where(x > 0, x, torch.exp(x) - 1.0)",
         ref=lambda x: [f32(np.where(x > 0, f64(x), np.exp(f64(x)) - 1.0))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_hardswish", code="x = torch.randn(2048); x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0",
         ref=lambda x: [f32(f64(x) * np.clip(f64(x) + 3.0, 0.0, 6.0) / 6.0)], rtol=1e-5, atol=1e-6),
    dict(id="x_p_mish", code="x = torch.randn(2048); x * torch.tanh(torch.nn.functional.softplus(x))",
         ref=lambda x: [f32(f64(x) * np.tanh(np.logaddexp(0, f64(x))))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_gauss", code="x = torch.randn(4096); torch.exp(-x * x)",
         ref=lambda x: [f32(np.exp(-f64(x) ** 2))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_sigmoid_tails", code="torch.sigmoid(torch.randn(4096) * 10.0)",
         ref=lambda x: [f32(_sig(f64(x) * 10.0))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_tanh_tails", code="torch.tanh(torch.randn(4096) * 8.0)",
         ref=lambda x: [f32(np.tanh(f64(x) * 8.0))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_recip", code="1.0 / (torch.randn(2048) ** 2 + 0.5)",
         ref=lambda x: [f32(1.0 / (f64(x) ** 2 + 0.5))], rtol=1e-6, atol=1e-6),
    dict(id="x_p_pow_frac", code="torch.abs(torch.randn(2048)) ** 1.5",
         ref=lambda x: [f32(np.abs(f64(x)) ** 1.5)], rtol=1e-5, atol=1e-6),
    dict(id="x_p_sin", code="torch.sin(torch.randn(2048))",
         ref=lambda x: [f32(np.sin(f64(x)))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_sin_wide", code="torch.sin(torch.randn(4096) * 20.0)",
         ref=lambda x: [f32(np.sin(f64(x * np.float32(20.0))))], rtol=1e-5, atol=1e-6),  # torch rounds x*20 to f32 first
    dict(id="x_p_cos", code="torch.cos(torch.randn(2048) * 3.0)",
         ref=lambda x: [f32(np.cos(f64(x * np.float32(3.0))))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_sincos", code="x = torch.randn(64, 32); t = torch.randn(64, 32); x * torch.cos(t) - x * torch.sin(t)",
         ref=lambda x, t: [f32(f64(x) * np.cos(f64(t)) - f64(x) * np.sin(f64(t)))], rtol=1e-5, atol=1e-6),
    dict(id="x_p_softsign", code="torch.nn.functional.softsign(torch.randn(2048))",
         ref=lambda x: [f32(f64(x) / (1 + np.abs(f64(x))))], rtol=1e-6, atol=1e-7),
    dict(id="x_p_masked_fill", code="x = torch.randn(32, 32); x.masked_fill(x > 0.5, 0.0)",
         ref=lambda x: [np.where(x > 0.5, np.float32(0), x)], rtol=0, atol=0),
    # reductions
    dict(id="x_r_mid_axis", code="torch.randn(8, 64, 16).sum(1)",
         ref=lambda x: [f32(f64(x).sum(1))], rtol=1e-4, atol=1e-4),
    dict(id="x_r_rows_odd", code="torch.randn(37, 333).sum(-1)",
         ref=lambda x: [f32(f64(x).sum(-1))], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="x_r_l1", code="torch.abs(torch.randn(5000)).sum()",
         ref=lambda x: [f32(np.abs(f64(x)).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="x_r_sumexp", code="torch.exp(torch.randn(4096)).sum()",
         ref=lambda x: [f32(np.exp(f64(x)).sum())], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="x_r_keepdim", code="torch.randn(32, 200).sum(-1, keepdim=True)",
         ref=lambda x: [f32(f64(x).sum(-1, keepdims=True))], rtol=1e-4, atol=1e-4),
    dict(id="x_r_mean_cols", code="torch.randn(500, 40).mean(0)",
         ref=lambda x: [f32(f64(x).mean(0))], rtol=1e-4, atol=1e-4),
    dict(id="x_r_big_unpinned", code="torch.randn(1048576).sum()",
         ref=lambda x: [f32(f64(x).sum())], rtol=1e-3, atol=1e-2),
    dict(id="x_r_rowmax_narrow", code="torch.randn(256, 7).amax(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0),
    dict(id="x_r_last_3d", code="torch.randn(4, 8, 256).sum(-1)",
         ref=lambda x: [f32(f64(x).sum(-1))], rtol=1e-4, atol=1e-4),
    dict(id="x_r_cumsum", code="torch.randn(16, 64).cumsum(-1)",
         ref=lambda x: [f32(np.cumsum(f64(x), -1))], rtol=1e-4, atol=1e-5),
    # matmul
    dict(id="x_m_100x50x70", code="torch.randn(100, 50) @ torch.randn(50, 70)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_8", code="torch.randn(8, 8) @ torch.randn(8, 8)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_splitk", code="torch.randn(32, 4096) @ torch.randn(4096, 32)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_bmm_bcast", code="torch.randn(3, 32, 48) @ torch.randn(48, 40)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_residual", code="x = torch.randn(64, 64); w = torch.randn(64, 64); x @ w + x",
         ref=lambda x, w: [f32(f64(x) @ f64(w) + f64(x))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_vecmat", code="torch.randn(128) @ torch.randn(128, 96)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_scaled", code="(torch.randn(128, 64) @ torch.randn(64, 128)) * 0.5",
         ref=lambda a, b: [f32(f64(a) @ f64(b) * 0.5)], rtol=1e-3, atol=1e-3),
    dict(id="x_m_wide", code="torch.randn(16, 64) @ torch.randn(64, 1024)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_bt", code="a = torch.randn(64, 48); b = torch.randn(80, 48); a @ b.t()",
         ref=lambda a, b: [f32(f64(a) @ f64(b).T)], rtol=1e-3, atol=1e-3),
    dict(id="x_m_einsum", code="torch.einsum('ij,jk->ik', torch.randn(32, 16), torch.randn(16, 24))",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], rtol=1e-3, atol=1e-3),
    dict(id="x_m_linear", code="torch.nn.functional.linear(torch.randn(32, 16), torch.randn(24, 16))",
         ref=lambda x, w: [f32(f64(x) @ f64(w).T)], rtol=1e-3, atol=1e-3),
    dict(id="x_m_linear_bias", code="torch.nn.functional.linear(torch.randn(64, 128), torch.randn(96, 128), torch.randn(96))",
         ref=lambda x, w, b: [f32(f64(x) @ f64(w).T + f64(b))], rtol=1e-3, atol=1e-3),
    # fused
    dict(id="x_f_softmax_100", code="torch.softmax(torch.randn(40, 100), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="x_f_div_absmax", code="x = torch.randn(64, 256); x / x.abs().amax(-1, keepdim=True)",
         ref=lambda x: [f32(f64(x) / np.abs(f64(x)).max(-1, keepdims=True))], rtol=1e-5, atol=1e-6),
    dict(id="x_f_div_sum", code="x = torch.randn(32, 300); torch.abs(x) / torch.abs(x).sum(-1, keepdim=True)",
         ref=lambda x: [f32(np.abs(f64(x)) / np.abs(f64(x)).sum(-1, keepdims=True))], rtol=1e-4, atol=1e-6),
    dict(id="x_f_layernorm_200", code="torch.nn.functional.layer_norm(torch.randn(48, 200), (200,))",
         ref=lambda x: [f32((f64(x) - f64(x).mean(-1, keepdims=True)) / np.sqrt(f64(x).var(-1, keepdims=True) + 1e-5))],
         rtol=1e-3, atol=1e-4),
    dict(id="x_f_rmsnorm_1000", code="x = torch.randn(24, 1000); x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)",
         ref=lambda x: [f32(f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-5))], rtol=1e-3, atol=1e-4),
    dict(id="x_f_softmax_3d", code="torch.softmax(torch.randn(4, 16, 64), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-5),
    dict(id="x_f_silu_dot", code="x = torch.randn(64, 128); (torch.nn.functional.silu(x) * x).sum(-1)",
         ref=lambda x: [f32((f64(x) * _sig(f64(x)) * f64(x)).sum(-1))], rtol=1e-4, atol=1e-4),
    dict(id="x_f_colnorm", code="x = torch.randn(128, 32); x / torch.sqrt((x * x).sum(0, keepdim=True))",
         ref=lambda x: [f32(f64(x) / np.sqrt((f64(x) ** 2).sum(0, keepdims=True)))], rtol=1e-4, atol=1e-6),
    # attention
    dict(id="x_a_16", code="q = torch.randn(16, 16); kt = torch.randn(16, 16); v = torch.randn(16, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(f64(q) @ f64(kt) / 4.0) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_48x80x40", code="q = torch.randn(48, 32); kt = torch.randn(32, 80); v = torch.randn(80, 40); torch.softmax(q @ kt * 0.2, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(f64(q) @ f64(kt) * 0.2) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_sdpa_b2h4", code="q = torch.randn(2, 4, 32, 16); k = torch.randn(2, 4, 32, 16); v = torch.randn(2, 4, 32, 16); torch.nn.functional.scaled_dot_product_attention(q, k, v)",
         ref=lambda q, k, v: [f32(_softmax(f64(q) @ np.swapaxes(f64(k), -1, -2) / math.sqrt(16)) @ f64(v))],
         rtol=1e-3, atol=1e-4),
    dict(id="x_a_causal", code="q = torch.randn(1, 2, 64, 32); k = torch.randn(1, 2, 64, 32); v = torch.randn(1, 2, 64, 32); torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)",
         ref=lambda q, k, v: [f32(_softmax(np.where(np.tril(np.ones((64, 64))) > 0, f64(q) @ np.swapaxes(f64(k), -1, -2) / math.sqrt(32), -np.inf)) @ f64(v))],
         rtol=1e-3, atol=1e-4),
    dict(id="x_a_cross", code="q = torch.randn(64, 32); k = torch.randn(128, 32); v = torch.randn(128, 32); torch.softmax(q @ k.t() / 5.66, dim=-1) @ v",
         ref=lambda q, k, v: [f32(_softmax(f64(q) @ f64(k).T / 5.66) @ f64(v))], rtol=1e-3, atol=1e-4),
    dict(id="x_a_long_kv", code="q = torch.randn(32, 64); kt = torch.randn(64, 1024); v = torch.randn(1024, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax(f64(q) @ f64(kt) / 8.0) @ f64(v))], rtol=1e-3, atol=1e-4),
]
