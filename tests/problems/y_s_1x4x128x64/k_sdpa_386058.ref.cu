extern "C" __global__
__launch_bounds__(256) void k_sdpa_386058(const float* scaled_dot_product_attention_scale, const float* q, const float* k, const float* v, float* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 8192;
        int a1 = (_gid / 64) % 128;
        int a6 = (_gid) % 64;
        float in0 = scaled_dot_product_attention_scale[0];
        float acc3 = 0.0f;
        float acc5__sum = 0.0f;
        float acc1 = -1e+30f;
        for (int a2 = 0; a2 < 128; a2++) {
            float acc0 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2 = q[a0 * 8192 + a1 * 64 + a3];
                float in1 = k[a0 * 8192 + a2 * 64 + a3];
                float v0 = in1 * in2;
                acc0 += v0;
            }
            float v1 = acc0 * in0;
            float one = 1.0f;
            float v1__gn = fmaxf(acc1, v1);
            float v1__alpha = expf((acc1 - v1__gn));
            float v1__beta = expf((v1 - v1__gn));
            float v1__acc3_sa = acc3 * v1__alpha;
            float v1__acc3_sb = one * v1__beta;
            acc3 = v1__acc3_sa + v1__acc3_sb;
            float v1__acc5__sum_sa = acc5__sum * v1__alpha;
            float in7 = v[a0 * 8192 + a2 * 64 + a6];
            float v1__acc5__sum_sb = in7 * v1__beta;
            acc5__sum = v1__acc5__sum_sa + v1__acc5__sum_sb;
            acc1 = fmaxf(acc1, v1);
        }
        float acc5 = acc5__sum * (1.0f / acc3);
        scaled_dot_product_attention[a0 * 8192 + a1 * 64 + a6] = acc5;
}
