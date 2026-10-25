extern "C" __global__
__launch_bounds__(128) void k_scaled_dot_product_attention(const double* scaled_dot_product_attention_scale, const double* q, const double* k, const double* v, double* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a7_ks = _gid / 4096;
        int a0 = (_gid / 512) % 8;
        int a1_b = (_gid / 512) % 1;
        int a6_b = (_gid / 128) % 4;
        int a1_u = (_gid / 16) % 8;
        int a6_u = (_gid) % 16;
        float in0__ar0 = scaled_dot_product_attention_scale[0];
        float acc1__ar0 = -1e+30f;
        for (int a2 = 0; a2 < 16; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar0 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4) % 16 * 64 + a3];
                float in1__ar0 = k[a0 * 1024 + a2 * 64 + a3];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0 * in0__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v1__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a2 = 0; a2 < 16; a2++) {
            float acc2__ar0 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar0 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4) % 16 * 64 + a5];
                float in3__ar0 = k[a0 * 1024 + a2 * 64 + a5];
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
        for (int a2 = 0; a2 < 16; a2++) {
            float acc0__ar1 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar1 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 1) % 16 * 64 + a3];
                float in1__ar1 = k[a0 * 1024 + a2 * 64 + a3];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__ar1 = acc0__ar1 * in0__ar1;
            acc1__ar1 = fmaxf(acc1__ar1, v1__ar1);
        }
        float acc3__ar1 = 0.0f;
        for (int a2 = 0; a2 < 16; a2++) {
            float acc2__ar1 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar1 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 1) % 16 * 64 + a5];
                float in3__ar1 = k[a0 * 1024 + a2 * 64 + a5];
                float v2__ar1 = in3__ar1 * in4__ar1;
                acc2__ar1 += v2__ar1;
            }
            float v3__ar1 = acc2__ar1 * in0__ar1;
            float v5__ar1 = expf((v3__ar1 - acc1__ar1));
            acc3__ar1 += v5__ar1;
        }
        float v6__ar1 = 1.0f / acc3__ar1;
        float in0__ar2 = scaled_dot_product_attention_scale[0];
        float acc1__ar2 = -1e+30f;
        for (int a2 = 0; a2 < 16; a2++) {
            float acc0__ar2 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar2 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 2) % 16 * 64 + a3];
                float in1__ar2 = k[a0 * 1024 + a2 * 64 + a3];
                float v0__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v0__ar2;
            }
            float v1__ar2 = acc0__ar2 * in0__ar2;
            acc1__ar2 = fmaxf(acc1__ar2, v1__ar2);
        }
        float acc3__ar2 = 0.0f;
        for (int a2 = 0; a2 < 16; a2++) {
            float acc2__ar2 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar2 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 2) % 16 * 64 + a5];
                float in3__ar2 = k[a0 * 1024 + a2 * 64 + a5];
                float v2__ar2 = in3__ar2 * in4__ar2;
                acc2__ar2 += v2__ar2;
            }
            float v3__ar2 = acc2__ar2 * in0__ar2;
            float v5__ar2 = expf((v3__ar2 - acc1__ar2));
            acc3__ar2 += v5__ar2;
        }
        float v6__ar2 = 1.0f / acc3__ar2;
        float in0__ar3 = scaled_dot_product_attention_scale[0];
        float acc1__ar3 = -1e+30f;
        for (int a2 = 0; a2 < 16; a2++) {
            float acc0__ar3 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar3 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 3) % 16 * 64 + a3];
                float in1__ar3 = k[a0 * 1024 + a2 * 64 + a3];
                float v0__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v0__ar3;
            }
            float v1__ar3 = acc0__ar3 * in0__ar3;
            acc1__ar3 = fmaxf(acc1__ar3, v1__ar3);
        }
        float acc3__ar3 = 0.0f;
        for (int a2 = 0; a2 < 16; a2++) {
            float acc2__ar3 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar3 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 3) % 16 * 64 + a5];
                float in3__ar3 = k[a0 * 1024 + a2 * 64 + a5];
                float v2__ar3 = in3__ar3 * in4__ar3;
                acc2__ar3 += v2__ar3;
            }
            float v3__ar3 = acc2__ar3 * in0__ar3;
            float v5__ar3 = expf((v3__ar3 - acc1__ar3));
            acc3__ar3 += v5__ar3;
        }
        float v6__ar3 = 1.0f / acc3__ar3;
        float acc5__c0_0 = 0.0f;
        float acc5__c1_0 = 0.0f;
        float acc5__c2_0 = 0.0f;
        float acc5__c3_0 = 0.0f;
        #pragma unroll
        for (int a7 = 0; a7 < 8; a7++) {
            float acc4__ar0 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in5__ar0 = k[a0 * 1024 + (_a7_ks * 8 + a7) * 64 + a8];
                float in6__ar0 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4) % 16 * 64 + a8];
                float v7__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v7__ar0;
            }
            float v8__ar0 = acc4__ar0 * in0__ar0;
            float v11__ar0 = expf((v8__ar0 - acc1__ar0)) * v6__ar0;
            float acc4__ar1 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in5__ar1 = k[a0 * 1024 + (_a7_ks * 8 + a7) * 64 + a8];
                float in6__ar1 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 1) % 16 * 64 + a8];
                float v7__ar1 = in5__ar1 * in6__ar1;
                acc4__ar1 += v7__ar1;
            }
            float v8__ar1 = acc4__ar1 * in0__ar1;
            float v11__ar1 = expf((v8__ar1 - acc1__ar1)) * v6__ar1;
            float acc4__ar2 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in5__ar2 = k[a0 * 1024 + (_a7_ks * 8 + a7) * 64 + a8];
                float in6__ar2 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 2) % 16 * 64 + a8];
                float v7__ar2 = in5__ar2 * in6__ar2;
                acc4__ar2 += v7__ar2;
            }
            float v8__ar2 = acc4__ar2 * in0__ar2;
            float v11__ar2 = expf((v8__ar2 - acc1__ar2)) * v6__ar2;
            float acc4__ar3 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in5__ar3 = k[a0 * 1024 + (_a7_ks * 8 + a7) * 64 + a8];
                float in6__ar3 = q[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 3) % 16 * 64 + a8];
                float v7__ar3 = in5__ar3 * in6__ar3;
                acc4__ar3 += v7__ar3;
            }
            float v8__ar3 = acc4__ar3 * in0__ar3;
            float v11__ar3 = expf((v8__ar3 - acc1__ar3)) * v6__ar3;
            float in7__bc0 = v[a0 * 1024 + (_a7_ks * 8 + a7) * 64 + (a6_b * 16 + a6_u)];
            float v12__c0_0 = in7__bc0 * v11__ar0;
            acc5__c0_0 += v12__c0_0;
            float v12__c1_0 = in7__bc0 * v11__ar1;
            acc5__c1_0 += v12__c1_0;
            float v12__c2_0 = in7__bc0 * v11__ar2;
            acc5__c2_0 += v12__c2_0;
            float v12__c3_0 = in7__bc0 * v11__ar3;
            acc5__c3_0 += v12__c3_0;
        }
        if (a1_b * 32 + a1_u * 4 + 0 < 16) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 32 + a1_u * 4) % 16 * 64 + (a6_b * 16 + a6_u)], acc5__c0_0);
        }
        if (a1_b * 32 + a1_u * 4 + 1 < 16) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 1) % 16 * 64 + (a6_b * 16 + a6_u)], acc5__c1_0);
        }
        if (a1_b * 32 + a1_u * 4 + 2 < 16) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 2) % 16 * 64 + (a6_b * 16 + a6_u)], acc5__c2_0);
        }
        if (a1_b * 32 + a1_u * 4 + 3 < 16) {
            atomicAdd(&scaled_dot_product_attention[a0 * 1024 + (a1_b * 32 + a1_u * 4 + 3) % 16 * 64 + (a6_b * 16 + a6_u)], acc5__c3_0);
        }
}
