extern "C" __global__
__launch_bounds__(256) void k_scaled_dot_product_attention(const double* scaled_dot_product_attention_scale, const double* q, const double* k, const double* v, double* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a7_ks = _gid / 704;
        int a0 = (_gid / 176) % 4;
        int a1_b = (_gid / 11) % 16;
        int a6_b = (_gid) % 11;
        int a1_u = (_gid) % 1;
        int a6_u = (_gid) % 1;
        float in0__ar0 = scaled_dot_product_attention_scale[0];
        float acc1__ar0 = -1e+30f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in2__ar0 = q[a0 * 1024 + (a1_b * 2 + a1_u * 2) * 32 + a3];
                float in1__ar0 = k[a0 * 1024 + a2 * 32 + a3];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0 * in0__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v1__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar0 = 0.0f;
            for (int a5 = 0; a5 < 32; a5++) {
                float in4__ar0 = q[a0 * 1024 + (a1_b * 2 + a1_u * 2) * 32 + a5];
                float in3__ar0 = k[a0 * 1024 + a2 * 32 + a5];
                float v2__ar0 = in3__ar0 * in4__ar0;
                acc2__ar0 += v2__ar0;
            }
            float v3__ar0 = acc2__ar0 * in0__ar0;
            float v5__ar0 = expf((v3__ar0 - acc1__ar0));
            acc3__ar0 += v5__ar0;
        }
        float v6__ar0 = 1.0f / acc3__ar0;
        float in0__ar1 = scaled_dot_product_attention_scale[0];
        float acc1__ar1 = -1e+30f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar1 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in2__ar1 = q[a0 * 1024 + (a1_b * 2 + a1_u * 2 + 1) * 32 + a3];
                float in1__ar1 = k[a0 * 1024 + a2 * 32 + a3];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__ar1 = acc0__ar1 * in0__ar1;
            acc1__ar1 = fmaxf(acc1__ar1, v1__ar1);
        }
        float acc3__ar1 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar1 = 0.0f;
            for (int a5 = 0; a5 < 32; a5++) {
                float in4__ar1 = q[a0 * 1024 + (a1_b * 2 + a1_u * 2 + 1) * 32 + a5];
                float in3__ar1 = k[a0 * 1024 + a2 * 32 + a5];
                float v2__ar1 = in3__ar1 * in4__ar1;
                acc2__ar1 += v2__ar1;
            }
            float v3__ar1 = acc2__ar1 * in0__ar1;
            float v5__ar1 = expf((v3__ar1 - acc1__ar1));
            acc3__ar1 += v5__ar1;
        }
        float v6__ar1 = 1.0f / acc3__ar1;
        float acc5__c0_0 = 0.0f;
        float acc5__c0_1 = 0.0f;
        float acc5__c0_2 = 0.0f;
        float acc5__c1_0 = 0.0f;
        float acc5__c1_1 = 0.0f;
        float acc5__c1_2 = 0.0f;
        #pragma unroll
        for (int a7 = 0; a7 < 8; a7++) {
            float acc4__ar0 = 0.0f;
            for (int a8 = 0; a8 < 32; a8++) {
                float in5__ar0 = k[a0 * 1024 + (_a7_ks * 8 + a7) * 32 + a8];
                float in6__ar0 = q[a0 * 1024 + (a1_b * 2 + a1_u * 2) * 32 + a8];
                float v7__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v7__ar0;
            }
            float v8__ar0 = acc4__ar0 * in0__ar0;
            float v11__ar0 = expf((v8__ar0 - acc1__ar0)) * v6__ar0;
            float acc4__ar1 = 0.0f;
            for (int a8 = 0; a8 < 32; a8++) {
                float in5__ar1 = k[a0 * 1024 + (_a7_ks * 8 + a7) * 32 + a8];
                float in6__ar1 = q[a0 * 1024 + (a1_b * 2 + a1_u * 2 + 1) * 32 + a8];
                float v7__ar1 = in5__ar1 * in6__ar1;
                acc4__ar1 += v7__ar1;
            }
            float v8__ar1 = acc4__ar1 * in0__ar1;
            float v11__ar1 = expf((v8__ar1 - acc1__ar1)) * v6__ar1;
            float in7__bc0 = v[a0 * 1024 + (_a7_ks * 8 + a7) * 32 + (a6_b * 3 + a6_u * 3) % 32];
            float v12__c0_0 = in7__bc0 * v11__ar0;
            acc5__c0_0 += v12__c0_0;
            float in7__bc1 = v[a0 * 1024 + (_a7_ks * 8 + a7) * 32 + (a6_b * 3 + a6_u * 3 + 1) % 32];
            float v12__c0_1 = in7__bc1 * v11__ar0;
            acc5__c0_1 += v12__c0_1;
            float in7__bc2 = v[a0 * 1024 + (_a7_ks * 8 + a7) * 32 + (a6_b * 3 + a6_u * 3 + 2) % 32];
            float v12__c0_2 = in7__bc2 * v11__ar0;
            acc5__c0_2 += v12__c0_2;
            float v12__c1_0 = in7__bc0 * v11__ar1;
            acc5__c1_0 += v12__c1_0;
            float v12__c1_1 = in7__bc1 * v11__ar1;
            acc5__c1_1 += v12__c1_1;
            float v12__c1_2 = in7__bc2 * v11__ar1;
            acc5__c1_2 += v12__c1_2;
        }
        if (a6_b * 3 + a6_u * 3 + 0 < 32) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 2 + a1_u * 2) * 32 + (a6_b * 3 + a6_u * 3) % 32], acc5__c0_0);
        }
        if (a6_b * 3 + a6_u * 3 + 1 < 32) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 2 + a1_u * 2) * 32 + (a6_b * 3 + a6_u * 3 + 1) % 32], acc5__c0_1);
        }
        if (a6_b * 3 + a6_u * 3 + 2 < 32) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 2 + a1_u * 2) * 32 + (a6_b * 3 + a6_u * 3 + 2) % 32], acc5__c0_2);
        }
        if (a6_b * 3 + a6_u * 3 + 0 < 32) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 2 + a1_u * 2 + 1) * 32 + (a6_b * 3 + a6_u * 3) % 32], acc5__c1_0);
        }
        if (a6_b * 3 + a6_u * 3 + 1 < 32) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 2 + a1_u * 2 + 1) * 32 + (a6_b * 3 + a6_u * 3 + 1) % 32], acc5__c1_1);
        }
        if (a6_b * 3 + a6_u * 3 + 2 < 32) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 2 + a1_u * 2 + 1) * 32 + (a6_b * 3 + a6_u * 3 + 2) % 32], acc5__c1_2);
        }
}
