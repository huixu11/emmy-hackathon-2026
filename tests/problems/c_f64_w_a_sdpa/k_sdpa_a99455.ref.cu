extern "C" __global__
__launch_bounds__(128) void k_sdpa_a99455(const double* scaled_dot_product_attention_scale, const double* q, const double* k, const double* v, double* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 8192;
        int a1_b = (_gid / 128) % 64;
        int a6_b = (_gid / 128) % 1;
        int a1_u = (_gid / 128) % 1;
        int a6_u = (_gid) % 128;
        float in0__ar0 = scaled_dot_product_attention_scale[0];
        float acc1__ar0 = -1e+30f;
        for (int a2 = 0; a2 < 64; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in2__ar0 = q[a0 * 2048 + (a1_b + a1_u) * 32 + a3];
                float in1__ar0 = k[a0 * 2048 + a2 * 32 + a3];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0 * in0__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v1__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a2 = 0; a2 < 64; a2++) {
            float acc2__ar0 = 0.0f;
            for (int a5 = 0; a5 < 32; a5++) {
                float in4__ar0 = q[a0 * 2048 + (a1_b + a1_u) * 32 + a5];
                float in3__ar0 = k[a0 * 2048 + a2 * 32 + a5];
                float v2__ar0 = in3__ar0 * in4__ar0;
                acc2__ar0 += v2__ar0;
            }
            float v3__ar0 = acc2__ar0 * in0__ar0;
            float v5__ar0 = expf((v3__ar0 - acc1__ar0));
            acc3__ar0 += v5__ar0;
        }
        float v6__ar0 = 1.0f / acc3__ar0;
        float acc5__c0_0 = 0.0f;
        #pragma unroll
        for (int a7 = 0; a7 < 64; a7++) {
            float acc4__ar0 = 0.0f;
            for (int a8 = 0; a8 < 32; a8++) {
                float in6__ar0 = q[a0 * 2048 + (a1_b + a1_u) * 32 + a8];
                float in5__ar0 = k[a0 * 2048 + a7 * 32 + a8];
                float v7__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v7__ar0;
            }
            float v8__ar0 = acc4__ar0 * in0__ar0;
            float v11__ar0 = expf((v8__ar0 - acc1__ar0)) * v6__ar0;
            float in7__bc0 = v[a0 * 2048 + a7 * 32 + (a6_b * 128 + a6_u) % 32];
            float v12__c0_0 = in7__bc0 * v11__ar0;
            acc5__c0_0 += v12__c0_0;
        }
        if (a6_b * 128 + a6_u * 1 + 0 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b + a1_u) * 32 + (a6_b * 128 + a6_u) % 32] = acc5__c0_0;
        }
}
