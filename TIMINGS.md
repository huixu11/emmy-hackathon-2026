# Per-problem timings

ptxc's kernel times against Emmy's own CUDA for the same problems, on the GPU runner (RTX 5090). A ratio below 1 means ptxc is faster; "faster" and "slower" mean by more than 3%, about the run-to-run noise.

| set | problems | geometric mean, ours / reference | faster | within 3% | slower |
|---|---|---|---|---|---|
| pointwise | 49 (47 with a reference) | 0.98× | 2 | 45 | 0 |
| reductions | 23 | 0.94× | 6 | 17 | 0 |
| matmul | 23 | 0.94× | 4 | 19 | 0 |
| fused | 22 | 0.88× | 7 | 15 | 0 |
| attention | 32 | 0.54× | 29 | 3 | 0 |
| data types | 19 (16 with a reference) | 1.00× | 0 | 16 | 0 |
| f16 matmul on tensor cores | 9 | 1.00× | 0 | 9 | 0 |
| f16 attention on tensor cores | 9 | 0.79× | 4 | 5 | 0 |
| the open problems in f16, f64 and i32 | 39 (36 with a reference) | 0.91× | 9 | 27 | 0 |
| attention and reductions in f16 and f64 | 98 (95 with a reference) | 0.78× | 57 | 38 | 0 |
| ops no other problem uses | 24 | 1.00× | 0 | 24 | 0 |
| **all f32** (the five categories) | 149 (147 with a reference) | 0.84× | 48 | 99 | 0 |
| **all** | 347 (336 with a reference) | 0.85× | 118 | 218 | 0 |

The rows do not overlap: the 347 are the 149 f32 problems of the five categories plus 198 in other data types, on tensor cores or in the ops batch, and all 347 pass. The 23 bf16 and bool problems in `tests/problems_packed` pass too, but the runner cannot time them. The figures in brackets count the problems with a reference time: the reference itself fails 11 (see their rows). None is more than 3% slower than the reference.

The tables below list each set's problems from the slowest relative to the reference to the fastest.

## Pointwise (49 problems, 0.98×)

Every pointwise kernel takes about 3 µs, mostly launch time, and kernels this short jump by 0.5 µs now and then: `x_p_outer_bcast` (and `x_m_bmm_bcast`, among the matmuls) once read 1.18× and 1.16×, and 1.01× here. Where the work shows, the `pow` rows gain: `z_p_pow_frac_big` and `z_p_cube_big` (2^18 elements) take 3.1 and 3.0 µs against 4.7 µs.

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `p00_scalar_add` | 3.01 | 2.98 | 1.01× |
| `x_p_outer_bcast` | 3.01 | 2.98 | 1.01× |
| `v_p_rsqrt` | 3.04 | 3.01 | 1.01× |
| `w_p_two_out_like` | 3.04 | 3.01 | 1.01× |
| `x_p_tanh_tails` | 3.04 | 3.01 | 1.01× |
| `x_p_mish` | 3.10 | 3.07 | 1.01× |
| `v_p_gelu_tanh` | 3.23 | 3.20 | 1.01× |
| `p02_scale_shift` | 3.04 | 3.04 | 1.00× |
| `p03_relu_mul` | 3.01 | 3.01 | 1.00× |
| `v_p_bcast_col` | 3.01 | 3.01 | 1.00× |
| `v_p_bcast_row` | 3.07 | 3.07 | 1.00× |
| `v_p_clamp` | 3.04 | 3.04 | 1.00× |
| `v_p_div_abs` | 3.07 | 3.07 | 1.00× |
| `v_p_exp` | 2.98 | 2.98 | 1.00× |
| `v_p_gelu` | 3.14 | 3.14 | 1.00× |
| `v_p_max2` | 3.01 | 3.01 | 1.00× |
| `v_p_sigmoid` | 3.01 | 3.01 | 1.00× |
| `v_p_silu` | 3.01 | 3.01 | 1.00× |
| `v_p_sqrt_abs` | 2.98 | 2.98 | 1.00× |
| `w_p_3d` | 3.01 | 3.01 | 1.00× |
| `w_p_chain` | 3.07 | 3.07 | 1.00× |
| `x_p_cos` | 3.01 | 3.01 | 1.00× |
| `x_p_div_const` | 3.01 | 3.01 | 1.00× |
| `x_p_gauss` | 2.98 | 2.98 | 1.00× |
| `x_p_pow_frac` | 3.01 | 3.01 | 1.00× |
| `x_p_sigmoid_tails` | 3.01 | 3.01 | 1.00× |
| `x_p_sin` | 2.98 | 2.98 | 1.00× |
| `x_p_sincos` | 3.01 | 3.01 | 1.00× |
| `x_p_softsign` | 3.01 | 3.01 | 1.00× |
| `z_p_pow2` | 2.98 | 2.98 | 1.00× |
| `z_p_pow_neg2` | 3.04 | 3.04 | 1.00× |
| `z_p_sqrt_big` | 2.98 | 2.98 | 1.00× |
| `x_p_hardswish` | 3.20 | 3.22 | 1.00× |
| `x_p_elu` | 3.06 | 3.07 | 0.99× |
| `x_p_masked_fill` | 3.04 | 3.06 | 0.99× |
| `v_p_add_odd` | 2.99 | 3.01 | 0.99× |
| `v_p_softplus` | 3.07 | 3.10 | 0.99× |
| `v_p_where` | 3.04 | 3.07 | 0.99× |
| `x_p_recip` | 3.04 | 3.07 | 0.99× |
| `x_p_sin_wide` | 3.04 | 3.07 | 0.99× |
| `v_p_pow3` | 3.01 | 3.04 | 0.99× |
| `p01_vector_add` | 2.98 | 3.01 | 0.99× |
| `x_p_scalar_bcast` | 2.98 | 3.01 | 0.99× |
| `z_p_rsqrt_pow` | 2.98 | 3.01 | 0.99× |
| `v_p_tanh` | 3.01 | 3.07 | 0.98× |
| `z_p_pow_frac_big` | 3.10 | 4.74 | 0.66× |
| `z_p_cube_big` | 3.04 | 4.74 | 0.64× |
| `v_p_log` | 3.01 | — | no reference: NVRTC rejects its reference source |
| `x_p_fma_5000` | 3.04 | — | no reference time: the reference fails the check (nvcc fuses x*y+z) |

## Reductions (23 problems, 0.94×)

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `x_r_rows_odd` | 3.01 | 2.98 | 1.01× |
| `w_r_amax_rows` | 2.99 | 2.98 | 1.01× |
| `w_r_rows_unpinned` | 2.99 | 2.98 | 1.01× |
| `r01_array_sum` | 2.98 | 2.98 | 1.00× |
| `r04_row_sum` | 3.01 | 3.01 | 1.00× |
| `v_r_colsum` | 3.01 | 3.01 | 1.00× |
| `v_r_mean_rows` | 4.70 | 4.70 | 1.00× |
| `v_r_sum_odd` | 4.70 | 4.70 | 1.00× |
| `x_r_l1` | 4.70 | 4.70 | 1.00× |
| `x_r_last_3d` | 3.23 | 3.23 | 1.00× |
| `x_r_mid_axis` | 3.01 | 3.01 | 1.00× |
| `x_r_rowmax_narrow` | 2.98 | 2.98 | 1.00× |
| `r03_mean` | 6.75 | 6.78 | 1.00× |
| `x_r_big_unpinned` | 6.53 | 6.56 | 1.00× |
| `x_r_keepdim` | 3.25 | 3.26 | 1.00× |
| `w_r_sum_unpinned` | 3.23 | 3.25 | 1.00× |
| `v_r_sumsq` | 2.94 | 2.99 | 0.98× |
| `x_r_mean_cols` | 3.30 | 3.42 | 0.96× |
| `x_r_sumexp` | 2.98 | 3.10 | 0.96× |
| `r02_big_sum` | 8.80 | 10.77 | 0.82× |
| `x_r_cumsum` | 4.74 | 6.78 | 0.70× |
| `z_r_pow_mean` | 3.23 | 4.74 | 0.68× |
| `v_r_rowmax_wide` | 2.98 | 4.70 | 0.63× |

## Matmul (23 problems, 0.94×)

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `x_m_bmm_bcast` | 3.14 | 3.10 | 1.01× |
| `v_m_256x128x256` | 7.07 | 7.04 | 1.00× |
| `v_m_tall` | 7.10 | 7.07 | 1.00× |
| `v_m_deepk` | 11.20 | 11.17 | 1.00× |
| `v_m_64` | 3.14 | 3.14 | 1.00× |
| `v_m_bias_relu` | 4.74 | 4.74 | 1.00× |
| `v_m_odd` | 4.74 | 4.74 | 1.00× |
| `w_m_bmm` | 5.06 | 5.06 | 1.00× |
| `w_m_linear` | 4.74 | 4.74 | 1.00× |
| `x_m_8` | 3.30 | 3.30 | 1.00× |
| `x_m_bt` | 4.99 | 4.99 | 1.00× |
| `x_m_linear` | 3.01 | 3.01 | 1.00× |
| `x_m_residual` | 4.70 | 4.70 | 1.00× |
| `x_m_splitk` | 4.83 | 4.83 | 1.00× |
| `x_m_vecmat` | 3.01 | 3.01 | 1.00× |
| `x_m_100x50x70` | 9.09 | 9.12 | 1.00× |
| `x_m_scaled` | 3.17 | 3.20 | 0.99× |
| `x_m_einsum` | 3.04 | 3.07 | 0.99× |
| `x_m_wide` | 7.04 | 7.14 | 0.99× |
| `m01_matmul_512` | 15.04 | 15.57 | 0.97× |
| `m02_matmul_rect` | 4.72 | 4.94 | 0.95× |
| `w_m_skinny` | 45.70 | 60.03 | 0.76× |
| `x_m_linear_bias` | 12.90 | 37.50 | 0.34× |

## Fused (22 problems, 0.88×)

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `f03_l2norm` | 3.01 | 2.98 | 1.01× |
| `x_f_div_absmax` | 3.01 | 2.98 | 1.01× |
| `x_f_softmax_100` | 3.01 | 2.98 | 1.01× |
| `v_f_center` | 3.01 | 3.01 | 1.00× |
| `w_f_softmax_dim0` | 10.91 | 10.91 | 1.00× |
| `w_f_swiglu` | 8.83 | 8.83 | 1.00× |
| `x_f_colnorm` | 6.78 | 6.78 | 1.00× |
| `x_f_layernorm_200` | 4.74 | 4.74 | 1.00× |
| `x_f_rmsnorm_1000` | 4.74 | 4.74 | 1.00× |
| `x_f_silu_dot` | 3.26 | 3.26 | 1.00× |
| `z_f_l2_pow` | 3.04 | 3.04 | 1.00× |
| `z_f_rmsnorm_pow` | 3.10 | 3.10 | 1.00× |
| `f02_row_softmax` | 2.98 | 3.01 | 0.99× |
| `x_f_div_sum` | 2.98 | 3.01 | 0.99× |
| `x_f_softmax_3d` | 2.98 | 3.01 | 0.99× |
| `v_f_layernorm` | 4.74 | 6.78 | 0.70× |
| `w_f_layernorm_affine` | 4.74 | 6.82 | 0.69× |
| `f01_rmsnorm` | 3.26 | 4.75 | 0.69× |
| `v_f_rmsnorm_w` | 3.17 | 4.75 | 0.67× |
| `w_f_softmax_temp` | 3.10 | 4.74 | 0.66× |
| `w_f_gelu_mlp` | 10.88 | 17.02 | 0.64× |
| `v_f_softmax_wide` | 3.01 | 4.74 | 0.64× |

## Attention (32 problems, 0.54×)

The `y_a_*` names give q rows × d × keys × v columns; the `y_s_*` names give SDPA's batch × heads × sequence × d. Emmy keeps a running max and sum for every output column a thread owns, though one row's copies always hold the same value; ptxc proves that and computes each once, half the exps in most of these kernels (`v_a_rect`: 58.0 → 33.4 µs, `y_s_1x1x256x64`: 111.3 → 78.5 µs). Kernels with few blocks (`v_a_rect` runs 4 blocks of 256 threads) are bound by the latency of each key's trip, so ptxas's schedule of that trip moves them by tens of percent.

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `y_a_64x32x16x32` | 4.70 | 4.70 | 1.00× |
| `a02_attention_128` | 29.31 | 29.33 | 1.00× |
| `v_a_256x16` | 8.80 | 8.83 | 1.00× |
| `y_s_2x2x64x32` | 64.13 | 80.51 | 0.80× |
| `x_a_causal` | 21.09 | 27.26 | 0.77× |
| `x_a_48x80x40` | 6.82 | 8.83 | 0.77× |
| `x_a_long_kv` | 982.32 | 1279.34 | 0.77× |
| `y_s_1x2x48x64` | 19.07 | 25.22 | 0.76× |
| `x_a_sdpa_b2h4` | 10.88 | 14.96 | 0.73× |
| `y_a_32x128x256x64` | 19.07 | 27.26 | 0.70× |
| `y_s_1x8x64x32` | 19.07 | 27.26 | 0.70× |
| `x_a_16` | 4.70 | 6.75 | 0.70× |
| `y_a_96x64x96x48` | 41.30 | 60.03 | 0.69× |
| `y_a_128x64x128x64` | 51.84 | 76.45 | 0.68× |
| `y_a_256x32x256x32` | 57.95 | 86.66 | 0.67× |
| `y_a_64x64x32x64` | 10.85 | 17.02 | 0.64× |
| `a01_attention` | 41.60 | 74.34 | 0.56× |
| `v_a_32x64` | 12.93 | 23.17 | 0.56× |
| `v_a_rect` | 33.38 | 60.77 | 0.55× |
| `w_a_512x64` | 170.62 | 354.88 | 0.48× |
| `y_a_48x32x48x32` | 8.82 | 19.04 | 0.46× |
| `y_a_16x64x512x64` | 193.14 | 434.78 | 0.44× |
| `w_a_sdpa` | 21.12 | 49.79 | 0.42× |
| `y_a_32x64x64x64` | 27.20 | 66.14 | 0.41× |
| `y_a_64x64x256x64` | 96.86 | 246.91 | 0.39× |
| `y_s_2x4x16x64` | 6.82 | 19.07 | 0.36× |
| `y_s_1x4x128x64` | 41.52 | 117.38 | 0.35× |
| `y_s_4x2x32x64` | 12.93 | 39.55 | 0.33× |
| `y_s_2x2x32x64` | 10.88 | 35.46 | 0.31× |
| `y_s_1x1x256x64` | 78.45 | 277.01 | 0.28× |
| `x_a_cross` | 27.26 | 109.18 | 0.25× |
| `y_s_2x2x32x32` | 94.61 | 487.33 | 0.19× |

## Data types other than f32 (19 problems, 1.00×)

f16, f64, i32 and i64 pointwise ops, conversions, reductions, an f16 softmax and layer norm (bf16 and bool problems are in `tests/problems_packed`, which run one launch each, untimed). Three have no reference time, because the reference does not pass: NVRTC rejects its f16 layer norm, and for integer `//` and `%` it uses C's division, which truncates toward zero, while torch rounds toward minus infinity (the inputs include negative integers).

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `d_p_f16_exp` | 2.98 | 2.94 | 1.01× |
| `d_p_f32_to_f16` | 2.98 | 2.94 | 1.01× |
| `d_p_i64_add` | 3.04 | 3.01 | 1.01× |
| `d_r_f64_sum` | 3.04 | 3.01 | 1.01× |
| `d_p_f16_fma` | 3.07 | 3.07 | 1.00× |
| `d_p_f16_sigmoid` | 2.98 | 2.98 | 1.00× |
| `d_p_f16_to_f32` | 3.01 | 3.01 | 1.00× |
| `d_p_f64_sqrt` | 4.77 | 4.77 | 1.00× |
| `d_p_f64_tanh` | 4.77 | 4.77 | 1.00× |
| `d_p_i32_to_f32` | 3.01 | 3.01 | 1.00× |
| `d_r_f16_sum` | 2.98 | 2.98 | 1.00× |
| `d_r_i32_sum` | 2.98 | 2.98 | 1.00× |
| `d_p_f64_mul` | 4.75 | 4.77 | 1.00× |
| `d_p_i32_arith` | 3.06 | 3.07 | 0.99× |
| `d_f_f16_softmax` | 2.98 | 3.01 | 0.99× |
| `d_r_f16_amax` | 2.98 | 3.01 | 0.99× |
| `d_f_f16_layernorm` | 4.70 | — | no reference: NVRTC rejects its reference source |
| `d_p_i32_floordiv` | 3.01 | — | no reference time: the reference fails the check (it truncates toward zero) |
| `d_p_i32_mod` | 3.01 | — | no reference time: the reference fails the check (C's % keeps the sign of x) |

## f16 matmul on tensor cores (9 problems, 1.00×)

`mma.sync` m16n8k16 with `ldmatrix` from swizzled shared tiles, or fragments read straight from global memory for the small and odd shapes. Both compilers issue the same tensor-core instructions here, so the times match.

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `t_m_f16_512x512x512` | 3.04 | 3.01 | 1.01× |
| `t_m_f16_128x128x128` | 2.98 | 2.98 | 1.00× |
| `t_m_f16_96x80x112` | 2.98 | 2.98 | 1.00× |
| `t_m_f16_bmm` | 3.01 | 3.01 | 1.00× |
| `t_m_f16_linear` | 4.74 | 4.74 | 1.00× |
| `t_m_f16_relu` | 3.04 | 3.04 | 1.00× |
| `t_m_f16_256x256x256` | 6.78 | 6.82 | 1.00× |
| `t_m_f16_100x70x50` | 6.75 | 6.78 | 1.00× |
| `t_m_f16_64x64x64` | 2.94 | 2.96 | 0.99× |

## f16 attention on tensor cores (9 problems, 0.79×)

q @ k.T and p @ v on `mma.sync`, the online softmax between them on the C fragments. `t_s_f16_mask_*` (an additive `attn_mask`) is scheduled by emmy on the scalar path instead; `t_s_f16_causal_1x1x64x32` gains from keeping its key-chunk loop rolled. The q operand is read from global memory as 32-bit words, two f16 values each, so ptxas issues all its loads before the first wait on the k tile; with 16-bit loads it had put some behind it (`c_f16_x_a_sdpa_b2h4`, among the f16 attention problems below: 4.70 → 3.23 µs).

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `t_s_f16_1x1x256x64` | 8.83 | 8.83 | 1.00× |
| `t_s_f16_causal_1x2x128x64` | 8.80 | 8.80 | 1.00× |
| `t_s_f16_1x2x64x64` | 12.93 | 12.96 | 1.00× |
| `t_s_f16_2x4x128x64` | 10.88 | 10.91 | 1.00× |
| `t_s_f16_1x2x64x32` | 4.72 | 4.74 | 1.00× |
| `t_s_f16_scale_1x2x128x64` | 31.36 | 37.50 | 0.84× |
| `t_s_f16_manual_2x128x64` | 4.80 | 6.72 | 0.71× |
| `t_s_f16_causal_1x1x64x32` | 4.74 | 6.82 | 0.69× |
| `t_s_f16_mask_1x2x64x64` | 98.94 | 341.38 | 0.29× |

## The open problems in f16, f64 and i32 (39 problems, 0.91×)

The 15 open problems again, with every input in f16, f64 or i32 (`tests/expressions_cross.py`; the bf16 ones are in `tests/problems_packed`, untimed). f64 values are computed in f32 here, as the reference's CUDA computes them (the default; see the next table for `--exact-f64`). Three have no reference time: NVRTC rejects their f16 reference sources.

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `c_f64_p01_vector_add` | 3.04 | 2.98 | 1.02× |
| `c_f16_r01_array_sum` | 2.98 | 2.94 | 1.01× |
| `c_f16_r04_row_sum` | 2.98 | 2.94 | 1.01× |
| `c_f16_m01_matmul_512` | 3.01 | 2.98 | 1.01× |
| `c_i32_p02_scale_shift` | 3.04 | 3.01 | 1.01× |
| `c_f64_p03_relu_mul` | 4.78 | 4.77 | 1.00× |
| `c_i32_m01_matmul_512` | 21.41 | 21.34 | 1.00× |
| `c_f16_a01_attention` | 6.78 | 6.78 | 1.00× |
| `c_f16_m02_matmul_rect` | 4.77 | 4.77 | 1.00× |
| `c_f16_p00_scalar_add` | 2.98 | 2.98 | 1.00× |
| `c_f16_p01_vector_add` | 2.98 | 2.98 | 1.00× |
| `c_f64_m02_matmul_rect` | 9.18 | 9.18 | 1.00× |
| `c_f64_p00_scalar_add` | 3.01 | 3.01 | 1.00× |
| `c_f64_r01_array_sum` | 3.01 | 3.01 | 1.00× |
| `c_i32_p00_scalar_add` | 2.98 | 2.98 | 1.00× |
| `c_i32_p01_vector_add` | 2.98 | 2.98 | 1.00× |
| `c_i32_p03_relu_mul` | 2.98 | 2.98 | 1.00× |
| `c_i32_r01_array_sum` | 2.98 | 2.98 | 1.00× |
| `c_i32_r04_row_sum` | 2.94 | 2.94 | 1.00× |
| `c_f16_r03_mean` | 6.78 | 6.82 | 1.00× |
| `c_f16_p02_scale_shift` | 3.01 | 3.02 | 0.99× |
| `c_i32_r02_big_sum` | 8.83 | 8.88 | 0.99× |
| `c_i32_m02_matmul_rect` | 5.15 | 5.18 | 0.99× |
| `c_f64_p02_scale_shift` | 4.74 | 4.77 | 0.99× |
| `c_f16_r02_big_sum` | 8.83 | 8.90 | 0.99× |
| `c_f64_r04_row_sum` | 3.01 | 3.04 | 0.99× |
| `c_f16_f02_row_softmax` | 2.94 | 2.98 | 0.99× |
| `c_f64_r03_mean` | 12.90 | 15.01 | 0.86× |
| `c_f16_a02_attention_128` | 10.88 | 12.93 | 0.84× |
| `c_f64_r02_big_sum` | 21.12 | 25.28 | 0.84× |
| `c_f64_f01_rmsnorm` | 4.75 | 6.82 | 0.70× |
| `c_f64_a02_attention_128` | 620.98 | 907.74 | 0.68× |
| `c_f64_f03_l2norm` | 3.17 | 4.77 | 0.66× |
| `c_f64_f02_row_softmax` | 3.04 | 4.61 | 0.66× |
| `c_f64_a01_attention` | 869.70 | 1673.89 | 0.52× |
| `c_f64_m01_matmul_512` | 135.81 | 320.03 | 0.42× |
| `c_f16_f01_rmsnorm` | 3.26 | — | no reference: NVRTC rejects its reference source |
| `c_f16_f03_l2norm` | 2.93 | — | no reference: NVRTC rejects its reference source |
| `c_f16_p03_relu_mul` | 2.98 | — | no reference: NVRTC rejects its reference source |

## Attention and reductions in f16 and f64 (98 problems, 0.78×)

The 49 attention and reduction problems above, with every input in f16 or f64. In f16, attention takes 0.85× the reference's time and reductions 0.91×. In f64, which ptxc computes in f32 by default, as the reference's CUDA does (it declares every f64 value `float`): attention 0.60×, reductions 0.92×. The reference is bound there by converting each f64 element to f32 whenever it loads it, which the f64 unit does at 1/64 of the f32 rate; ptxc rounds three reads in five in integer instructions instead (to the same bits), so the f64 unit and the integer ALUs convert side by side (DESIGN.md, Data types). With `--exact-f64`, ptxc computes in f64 what the IR declares f64, which the RTX 5090 runs at 1/64 of the f32 rate: attention then takes 1.71×, reductions 0.97×, for the same accuracy in the end, which emmy's f32 accumulators limit (the median error of the 30 attention kernels is 4.2e-7 both ways). Three f16 means have no reference time: NVRTC rejects their reference sources.

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `c_f16_x_r_rows_odd` | 2.98 | 2.94 | 1.01× |
| `c_f64_v_r_colsum` | 3.01 | 2.98 | 1.01× |
| `c_f64_x_r_keepdim` | 3.26 | 3.23 | 1.01× |
| `c_f16_x_r_l1` | 4.77 | 4.74 | 1.01× |
| `c_f64_v_r_mean_rows` | 4.77 | 4.74 | 1.01× |
| `c_f64_v_r_sumsq` | 4.77 | 4.74 | 1.01× |
| `c_f16_y_s_1x8x64x32` | 2.98 | 2.96 | 1.01× |
| `c_f64_y_a_32x128x256x64` | 37393.70 | 37372.24 | 1.00× |
| `c_f16_v_a_32x64` | 4.74 | 4.74 | 1.00× |
| `c_f16_v_r_colsum` | 2.98 | 2.98 | 1.00× |
| `c_f16_w_r_rows_unpinned` | 2.94 | 2.94 | 1.00× |
| `c_f16_w_r_sum_unpinned` | 8.83 | 8.83 | 1.00× |
| `c_f16_x_r_keepdim` | 2.94 | 2.94 | 1.00× |
| `c_f16_x_r_last_3d` | 2.94 | 2.94 | 1.00× |
| `c_f16_x_r_mid_axis` | 2.94 | 2.94 | 1.00× |
| `c_f16_x_r_rowmax_narrow` | 2.94 | 2.94 | 1.00× |
| `c_f16_y_a_64x32x16x32` | 4.70 | 4.70 | 1.00× |
| `c_f16_y_s_1x1x256x64` | 8.83 | 8.83 | 1.00× |
| `c_f16_y_s_1x2x48x64` | 6.78 | 6.78 | 1.00× |
| `c_f16_y_s_2x2x32x64` | 4.74 | 4.74 | 1.00× |
| `c_f16_y_s_2x2x64x32` | 8.83 | 8.83 | 1.00× |
| `c_f16_y_s_4x2x32x64` | 10.94 | 10.94 | 1.00× |
| `c_f64_v_r_rowmax_wide` | 4.77 | 4.77 | 1.00× |
| `c_f64_v_r_sum_odd` | 4.77 | 4.77 | 1.00× |
| `c_f64_x_r_mid_axis` | 3.01 | 3.01 | 1.00× |
| `c_f64_x_r_rowmax_narrow` | 3.01 | 3.01 | 1.00× |
| `c_f16_y_s_1x4x128x64` | 6.78 | 6.80 | 1.00× |
| `c_f16_w_a_sdpa` | 4.72 | 4.74 | 1.00× |
| `c_f16_y_a_128x64x128x64` | 8.83 | 8.86 | 1.00× |
| `c_f16_x_a_causal` | 6.78 | 6.82 | 1.00× |
| `c_f64_x_r_rows_odd` | 3.01 | 3.02 | 0.99× |
| `c_f16_v_r_sum_odd` | 4.70 | 4.74 | 0.99× |
| `c_f16_v_a_rect` | 8.77 | 8.83 | 0.99× |
| `c_f64_w_r_amax_rows` | 3.01 | 3.04 | 0.99× |
| `c_f64_x_r_last_3d` | 3.01 | 3.04 | 0.99× |
| `c_f16_v_r_sumsq` | 2.94 | 2.98 | 0.99× |
| `c_f16_w_r_amax_rows` | 2.94 | 2.98 | 0.99× |
| `c_f64_x_r_sumexp` | 4.74 | 4.83 | 0.98× |
| `c_f64_x_r_mean_cols` | 3.30 | 3.42 | 0.96× |
| `c_f16_y_s_2x2x32x32` | 4.50 | 4.74 | 0.95× |
| `c_f64_y_a_128x64x128x64` | 4784.38 | 5161.57 | 0.93× |
| `c_f64_z_r_pow_mean` | 4.30 | 4.77 | 0.90× |
| `c_f16_y_a_256x32x256x32` | 19.04 | 21.12 | 0.90× |
| `c_f64_w_a_sdpa` | 178.82 | 201.34 | 0.89× |
| `c_f64_y_a_64x64x256x64` | 6160.67 | 7087.36 | 0.87× |
| `c_f16_x_a_cross` | 7.62 | 8.80 | 0.87× |
| `c_f16_y_a_32x128x256x64` | 25.22 | 29.31 | 0.86× |
| `c_f64_w_r_sum_unpinned` | 21.15 | 25.28 | 0.84× |
| `c_f64_x_r_big_unpinned` | 295.46 | 356.86 | 0.83× |
| `c_f16_y_a_96x64x96x48` | 8.83 | 10.88 | 0.81× |
| `c_f16_x_r_big_unpinned` | 76.45 | 94.85 | 0.81× |
| `c_f16_w_a_512x64` | 58.21 | 72.32 | 0.80× |
| `c_f64_x_a_sdpa_b2h4` | 191.36 | 240.54 | 0.80× |
| `c_f64_y_s_2x2x32x32` | 156.50 | 201.60 | 0.78× |
| `c_f16_y_a_32x64x64x64` | 6.82 | 8.80 | 0.77× |
| `c_f64_x_r_l1` | 5.12 | 6.78 | 0.75× |
| `c_f64_x_a_causal` | 1037.68 | 1394.08 | 0.74× |
| `c_f64_x_a_cross` | 1406.24 | 1918.26 | 0.73× |
| `c_f64_y_s_1x1x256x64` | 14760.94 | 20299.09 | 0.73× |
| `c_f16_x_a_48x80x40` | 10.88 | 14.98 | 0.73× |
| `c_f16_y_a_16x64x512x64` | 27.23 | 37.50 | 0.73× |
| `c_f16_x_a_16` | 4.74 | 6.62 | 0.71× |
| `c_f16_x_a_sdpa_b2h4` | 3.23 | 4.56 | 0.71× |
| `c_f16_x_r_cumsum` | 4.80 | 6.82 | 0.70× |
| `c_f16_y_a_48x32x48x32` | 4.74 | 6.75 | 0.70× |
| `c_f64_y_s_2x2x64x32` | 516.83 | 740.06 | 0.70× |
| `c_f16_y_s_2x4x16x64` | 4.74 | 6.78 | 0.70× |
| `c_f16_y_a_64x64x32x64` | 8.83 | 12.90 | 0.68× |
| `c_f64_x_a_48x80x40` | 601.36 | 881.12 | 0.68× |
| `c_f64_w_r_rows_unpinned` | 3.17 | 4.75 | 0.67× |
| `c_f16_v_a_256x16` | 68.26 | 103.17 | 0.66× |
| `c_f64_y_a_256x32x256x32` | 1936.18 | 2928.40 | 0.66× |
| `c_f64_y_a_64x32x16x32` | 27.62 | 41.92 | 0.66× |
| `c_f64_y_a_16x64x512x64` | 18901.54 | 29606.34 | 0.64× |
| `c_f64_x_r_cumsum` | 6.82 | 10.91 | 0.62× |
| `c_f64_x_a_long_kv` | 5384.50 | 8700.29 | 0.62× |
| `c_f16_v_r_rowmax_wide` | 2.94 | 4.77 | 0.62× |
| `c_f16_x_r_sumexp` | 2.94 | 4.77 | 0.62× |
| `c_f64_v_a_256x16` | 1498.38 | 2434.51 | 0.62× |
| `c_f64_x_a_16` | 15.55 | 25.76 | 0.60× |
| `c_f64_y_s_1x8x64x32` | 557.74 | 954.82 | 0.58× |
| `c_f64_y_a_96x64x96x48` | 822.32 | 1416.56 | 0.58× |
| `c_f16_x_a_long_kv` | 184.93 | 330.43 | 0.56× |
| `c_f16_y_a_64x64x256x64` | 60.10 | 109.02 | 0.55× |
| `c_f64_y_s_1x2x48x64` | 759.39 | 1407.49 | 0.54× |
| `c_f64_w_a_512x64` | 3779.57 | 7132.45 | 0.53× |
| `c_f64_y_a_48x32x48x32` | 94.88 | 180.86 | 0.52× |
| `c_f64_v_a_rect` | 492.03 | 981.47 | 0.50× |
| `c_f64_y_s_4x2x32x64` | 1240.29 | 2628.86 | 0.47× |
| `c_f64_y_a_64x64x32x64` | 293.76 | 635.92 | 0.46× |
| `c_f64_y_s_2x4x16x64` | 250.82 | 543.52 | 0.46× |
| `c_f64_v_a_32x64` | 189.06 | 453.15 | 0.42× |
| `c_f64_y_a_32x64x64x64` | 246.43 | 612.86 | 0.40× |
| `c_f64_y_s_1x4x128x64` | 5745.04 | 17055.26 | 0.34× |
| `c_f64_y_s_2x2x32x64` | 570.66 | 1784.54 | 0.32× |
| `c_f16_v_r_mean_rows` | 4.77 | — | no reference: NVRTC rejects its reference source |
| `c_f16_x_r_mean_cols` | 4.77 | — | no reference: NVRTC rejects its reference source |
| `c_f16_z_r_pow_mean` | 4.42 | — | no reference: NVRTC rejects its reference source |

## Ops no other problem uses (24 problems, 1.00×)

Data movement (cat, stack, roll, a 3-d transpose, expand, repeat_interleave, tril and triu, chunk), indexing by computed integers (index_select, embedding, gather), conv1d, and the comparisons and bool combines the other problems leave out (>=, <, ==, !=, &, |), in f32, f16, f64 and i32 and inside reductions (`tests/expressions_ops.py`). They are a few thousand elements each, so every kernel takes about 3 µs, nearly all of it launch: the batch is there to show these ops compile and come out right, and none is slower than the reference.

| problem | ours (µs) | reference (µs) | ours / reference |
|---|---|---|---|
| `u_p_expand` | 3.04 | 2.98 | 1.02× |
| `u_p_lt_scalar` | 3.07 | 3.01 | 1.02× |
| `u_p_chunk_glu` | 2.94 | 2.91 | 1.01× |
| `u_p_roll_2d` | 2.98 | 2.94 | 1.01× |
| `u_p_tril` | 3.04 | 3.01 | 1.01× |
| `u_r_f64_masked_sum` | 3.04 | 3.01 | 1.01× |
| `u_p_embedding` | 3.10 | 3.07 | 1.01× |
| `u_p_ge_where` | 2.99 | 2.98 | 1.01× |
| `u_p_cat_cols` | 3.02 | 3.01 | 1.01× |
| `u_p_and_mask` | 3.10 | 3.10 | 1.00× |
| `u_p_i32_eq` | 3.01 | 3.01 | 1.00× |
| `u_p_or_mask` | 3.10 | 3.10 | 1.00× |
| `u_p_repeat_interleave` | 2.98 | 2.98 | 1.00× |
| `u_p_roll` | 2.91 | 2.91 | 1.00× |
| `u_r_masked_sum` | 3.26 | 3.26 | 1.00× |
| `u_p_transpose_3d` | 2.96 | 2.98 | 0.99× |
| `u_m_conv1d` | 3.26 | 3.30 | 0.99× |
| `u_p_stack` | 3.04 | 3.07 | 0.99× |
| `u_p_f16_leaky` | 3.01 | 3.04 | 0.99× |
| `u_p_i32_ne` | 3.01 | 3.04 | 0.99× |
| `u_p_index_select` | 3.01 | 3.04 | 0.99× |
| `u_p_triu_diag` | 3.01 | 3.04 | 0.99× |
| `u_p_gather` | 2.98 | 3.01 | 0.99× |
| `u_f_masked_softmax` | 3.02 | 3.07 | 0.98× |

## Compile time

`compile_file` in one Python process, on the 4-core container this repo was developed in:

| kernels | median | 90% under | longest |
|---|---|---|---|
| the 15 open problems | 3 to 4 ms | 0.05 s | 0.08 s |
| all 382 (`problems`, `tests/problems`, `tests/problems_packed`) | 4 ms | 0.15 s | about 2 s |

The longest are f64 attentions whose fully unrolled module, with integer f64 reads, is over the runner's 2,000,000-character limit, so they are compiled again with `cvt` reads (`c_f64_y_s_1x1x256x64`: 4.0 M characters in 1.1 s, then 0.64 M in 0.7 s). A run of `compile.py` adds 0.3 to 0.4 s for starting Python and importing emmy. NVRTC took 0.1 to 0.4 s a run to turn the same kernels' reference CUDA into PTX.

## How these were measured

* **Time:** the median of 100 launches after a warmup (`tools/run_problems.py --iters 100`). Each compiler ran twice, the two alternating; each figure is the lower of its two runs.
* **Ours:** `compile.py` as submitted, f64 computed in f32 (the default). Every change since commit f5af27d, where the f32, dtype and tensor-core rows were measured, leaves the PTX of those problems as it was; the f64 rows were measured again after the default changed, and the 46 whose PTX the f64 reads (integer rounding, fma) changed were measured again with that compiler. The ops batch (`u_*`) was measured next; the integer casts it needed in index expressions leave the PTX of every other problem as it was. Last, the 38 timed kernels whose PTX the 32-bit fragment loads changed (f16 attention and matmul on tensor cores) were measured again with the compiler as submitted, in three alternating runs. All problems pass.
* **Steps of 2 µs:** kernels under about 15 µs read in steps of about 2 µs (2.98, 4.74, 6.78, 8.83, 10.88 µs), and one can land a step higher or lower in one session than in another. `c_f16_x_a_16` reads 0.71× because the reference took 6.7 µs in the session that measured it, against 4.8 µs before; ptxc's 4.74 µs did not change. The other way, `t_s_f16_manual_2x128x64` (6.61, 6.56, 4.80 µs) and `c_f16_x_a_cross` (8.80, 8.64, 7.62 µs) took the lower step in one run of three, so their rows owe it to that run, not to the compiler.
* **Reference:** Emmy's own CUDA for each problem (`*.ref.cu`), compiled to PTX with NVRTC and run under the same launch. That build is a yardstick only; it is not part of the compiler or of this repo.
* **One session:** the runner's speed drifts by about 20% between sessions, so figures from different sessions do not compare; each row's two columns come from one session.
* **Cold times differ:** without `--iters`, `run_problems.py` shows one cold launch per problem, which reads higher (a02: about 41 µs cold against 29.3 µs here).
