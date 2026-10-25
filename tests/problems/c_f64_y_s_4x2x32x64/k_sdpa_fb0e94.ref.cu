extern "C" __global__
__launch_bounds__(128) void k_sdpa_fb0e94(const double* scaled_dot_product_attention_scale, const double* q, const double* k, const double* v, double* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 512;
        int a1_b = (_gid / 512) % 1;
        int a6_b = (_gid / 128) % 4;
        int a1_u = (_gid / 16) % 8;
        int a6_u = (_gid) % 16;
        float in0__ar0 = scaled_dot_product_attention_scale[0];
        float acc1__ar0 = -1e+30f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar0 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8) % 32 * 64 + a3];
                float in1__ar0 = k[a0 * 2048 + a2 * 64 + a3];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0 * in0__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v1__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar0 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar0 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8) % 32 * 64 + a5];
                float in3__ar0 = k[a0 * 2048 + a2 * 64 + a5];
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
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar1 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 1) % 32 * 64 + a3];
                float in1__ar1 = k[a0 * 2048 + a2 * 64 + a3];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__ar1 = acc0__ar1 * in0__ar1;
            acc1__ar1 = fmaxf(acc1__ar1, v1__ar1);
        }
        float acc3__ar1 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar1 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar1 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 1) % 32 * 64 + a5];
                float in3__ar1 = k[a0 * 2048 + a2 * 64 + a5];
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
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar2 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar2 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 2) % 32 * 64 + a3];
                float in1__ar2 = k[a0 * 2048 + a2 * 64 + a3];
                float v0__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v0__ar2;
            }
            float v1__ar2 = acc0__ar2 * in0__ar2;
            acc1__ar2 = fmaxf(acc1__ar2, v1__ar2);
        }
        float acc3__ar2 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar2 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar2 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 2) % 32 * 64 + a5];
                float in3__ar2 = k[a0 * 2048 + a2 * 64 + a5];
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
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar3 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar3 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 3) % 32 * 64 + a3];
                float in1__ar3 = k[a0 * 2048 + a2 * 64 + a3];
                float v0__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v0__ar3;
            }
            float v1__ar3 = acc0__ar3 * in0__ar3;
            acc1__ar3 = fmaxf(acc1__ar3, v1__ar3);
        }
        float acc3__ar3 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar3 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar3 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 3) % 32 * 64 + a5];
                float in3__ar3 = k[a0 * 2048 + a2 * 64 + a5];
                float v2__ar3 = in3__ar3 * in4__ar3;
                acc2__ar3 += v2__ar3;
            }
            float v3__ar3 = acc2__ar3 * in0__ar3;
            float v5__ar3 = expf((v3__ar3 - acc1__ar3));
            acc3__ar3 += v5__ar3;
        }
        float v6__ar3 = 1.0f / acc3__ar3;
        float in0__ar4 = scaled_dot_product_attention_scale[0];
        float acc1__ar4 = -1e+30f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar4 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar4 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 4) % 32 * 64 + a3];
                float in1__ar4 = k[a0 * 2048 + a2 * 64 + a3];
                float v0__ar4 = in1__ar4 * in2__ar4;
                acc0__ar4 += v0__ar4;
            }
            float v1__ar4 = acc0__ar4 * in0__ar4;
            acc1__ar4 = fmaxf(acc1__ar4, v1__ar4);
        }
        float acc3__ar4 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar4 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar4 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 4) % 32 * 64 + a5];
                float in3__ar4 = k[a0 * 2048 + a2 * 64 + a5];
                float v2__ar4 = in3__ar4 * in4__ar4;
                acc2__ar4 += v2__ar4;
            }
            float v3__ar4 = acc2__ar4 * in0__ar4;
            float v5__ar4 = expf((v3__ar4 - acc1__ar4));
            acc3__ar4 += v5__ar4;
        }
        float v6__ar4 = 1.0f / acc3__ar4;
        float in0__ar5 = scaled_dot_product_attention_scale[0];
        float acc1__ar5 = -1e+30f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar5 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar5 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 5) % 32 * 64 + a3];
                float in1__ar5 = k[a0 * 2048 + a2 * 64 + a3];
                float v0__ar5 = in1__ar5 * in2__ar5;
                acc0__ar5 += v0__ar5;
            }
            float v1__ar5 = acc0__ar5 * in0__ar5;
            acc1__ar5 = fmaxf(acc1__ar5, v1__ar5);
        }
        float acc3__ar5 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar5 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar5 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 5) % 32 * 64 + a5];
                float in3__ar5 = k[a0 * 2048 + a2 * 64 + a5];
                float v2__ar5 = in3__ar5 * in4__ar5;
                acc2__ar5 += v2__ar5;
            }
            float v3__ar5 = acc2__ar5 * in0__ar5;
            float v5__ar5 = expf((v3__ar5 - acc1__ar5));
            acc3__ar5 += v5__ar5;
        }
        float v6__ar5 = 1.0f / acc3__ar5;
        float in0__ar6 = scaled_dot_product_attention_scale[0];
        float acc1__ar6 = -1e+30f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar6 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar6 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 6) % 32 * 64 + a3];
                float in1__ar6 = k[a0 * 2048 + a2 * 64 + a3];
                float v0__ar6 = in1__ar6 * in2__ar6;
                acc0__ar6 += v0__ar6;
            }
            float v1__ar6 = acc0__ar6 * in0__ar6;
            acc1__ar6 = fmaxf(acc1__ar6, v1__ar6);
        }
        float acc3__ar6 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar6 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar6 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 6) % 32 * 64 + a5];
                float in3__ar6 = k[a0 * 2048 + a2 * 64 + a5];
                float v2__ar6 = in3__ar6 * in4__ar6;
                acc2__ar6 += v2__ar6;
            }
            float v3__ar6 = acc2__ar6 * in0__ar6;
            float v5__ar6 = expf((v3__ar6 - acc1__ar6));
            acc3__ar6 += v5__ar6;
        }
        float v6__ar6 = 1.0f / acc3__ar6;
        float in0__ar7 = scaled_dot_product_attention_scale[0];
        float acc1__ar7 = -1e+30f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar7 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__ar7 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 7) % 32 * 64 + a3];
                float in1__ar7 = k[a0 * 2048 + a2 * 64 + a3];
                float v0__ar7 = in1__ar7 * in2__ar7;
                acc0__ar7 += v0__ar7;
            }
            float v1__ar7 = acc0__ar7 * in0__ar7;
            acc1__ar7 = fmaxf(acc1__ar7, v1__ar7);
        }
        float acc3__ar7 = 0.0f;
        for (int a2 = 0; a2 < 32; a2++) {
            float acc2__ar7 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4__ar7 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 7) % 32 * 64 + a5];
                float in3__ar7 = k[a0 * 2048 + a2 * 64 + a5];
                float v2__ar7 = in3__ar7 * in4__ar7;
                acc2__ar7 += v2__ar7;
            }
            float v3__ar7 = acc2__ar7 * in0__ar7;
            float v5__ar7 = expf((v3__ar7 - acc1__ar7));
            acc3__ar7 += v5__ar7;
        }
        float v6__ar7 = 1.0f / acc3__ar7;
        float acc5__c0_0 = 0.0f;
        float acc5__c1_0 = 0.0f;
        float acc5__c2_0 = 0.0f;
        float acc5__c3_0 = 0.0f;
        float acc5__c4_0 = 0.0f;
        float acc5__c5_0 = 0.0f;
        float acc5__c6_0 = 0.0f;
        float acc5__c7_0 = 0.0f;
        #pragma unroll
        for (int a7 = 0; a7 < 32; a7++) {
            float acc4__ar0 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6__ar0 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8) % 32 * 64 + a8];
                float in5__ar0 = k[a0 * 2048 + a7 * 64 + a8];
                float v7__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v7__ar0;
            }
            float v8__ar0 = acc4__ar0 * in0__ar0;
            float v11__ar0 = expf((v8__ar0 - acc1__ar0)) * v6__ar0;
            float acc4__ar1 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6__ar1 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 1) % 32 * 64 + a8];
                float in5__ar1 = k[a0 * 2048 + a7 * 64 + a8];
                float v7__ar1 = in5__ar1 * in6__ar1;
                acc4__ar1 += v7__ar1;
            }
            float v8__ar1 = acc4__ar1 * in0__ar1;
            float v11__ar1 = expf((v8__ar1 - acc1__ar1)) * v6__ar1;
            float acc4__ar2 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6__ar2 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 2) % 32 * 64 + a8];
                float in5__ar2 = k[a0 * 2048 + a7 * 64 + a8];
                float v7__ar2 = in5__ar2 * in6__ar2;
                acc4__ar2 += v7__ar2;
            }
            float v8__ar2 = acc4__ar2 * in0__ar2;
            float v11__ar2 = expf((v8__ar2 - acc1__ar2)) * v6__ar2;
            float acc4__ar3 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6__ar3 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 3) % 32 * 64 + a8];
                float in5__ar3 = k[a0 * 2048 + a7 * 64 + a8];
                float v7__ar3 = in5__ar3 * in6__ar3;
                acc4__ar3 += v7__ar3;
            }
            float v8__ar3 = acc4__ar3 * in0__ar3;
            float v11__ar3 = expf((v8__ar3 - acc1__ar3)) * v6__ar3;
            float acc4__ar4 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6__ar4 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 4) % 32 * 64 + a8];
                float in5__ar4 = k[a0 * 2048 + a7 * 64 + a8];
                float v7__ar4 = in5__ar4 * in6__ar4;
                acc4__ar4 += v7__ar4;
            }
            float v8__ar4 = acc4__ar4 * in0__ar4;
            float v11__ar4 = expf((v8__ar4 - acc1__ar4)) * v6__ar4;
            float acc4__ar5 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6__ar5 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 5) % 32 * 64 + a8];
                float in5__ar5 = k[a0 * 2048 + a7 * 64 + a8];
                float v7__ar5 = in5__ar5 * in6__ar5;
                acc4__ar5 += v7__ar5;
            }
            float v8__ar5 = acc4__ar5 * in0__ar5;
            float v11__ar5 = expf((v8__ar5 - acc1__ar5)) * v6__ar5;
            float acc4__ar6 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6__ar6 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 6) % 32 * 64 + a8];
                float in5__ar6 = k[a0 * 2048 + a7 * 64 + a8];
                float v7__ar6 = in5__ar6 * in6__ar6;
                acc4__ar6 += v7__ar6;
            }
            float v8__ar6 = acc4__ar6 * in0__ar6;
            float v11__ar6 = expf((v8__ar6 - acc1__ar6)) * v6__ar6;
            float acc4__ar7 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6__ar7 = q[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 7) % 32 * 64 + a8];
                float in5__ar7 = k[a0 * 2048 + a7 * 64 + a8];
                float v7__ar7 = in5__ar7 * in6__ar7;
                acc4__ar7 += v7__ar7;
            }
            float v8__ar7 = acc4__ar7 * in0__ar7;
            float v11__ar7 = expf((v8__ar7 - acc1__ar7)) * v6__ar7;
            float in7__bc0 = v[a0 * 2048 + a7 * 64 + (a6_b * 16 + a6_u)];
            float v12__c0_0 = in7__bc0 * v11__ar0;
            acc5__c0_0 += v12__c0_0;
            float v12__c1_0 = in7__bc0 * v11__ar1;
            acc5__c1_0 += v12__c1_0;
            float v12__c2_0 = in7__bc0 * v11__ar2;
            acc5__c2_0 += v12__c2_0;
            float v12__c3_0 = in7__bc0 * v11__ar3;
            acc5__c3_0 += v12__c3_0;
            float v12__c4_0 = in7__bc0 * v11__ar4;
            acc5__c4_0 += v12__c4_0;
            float v12__c5_0 = in7__bc0 * v11__ar5;
            acc5__c5_0 += v12__c5_0;
            float v12__c6_0 = in7__bc0 * v11__ar6;
            acc5__c6_0 += v12__c6_0;
            float v12__c7_0 = in7__bc0 * v11__ar7;
            acc5__c7_0 += v12__c7_0;
        }
        if (a1_b * 64 + a1_u * 8 + 0 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 64 + a1_u * 8) % 32 * 64 + (a6_b * 16 + a6_u)] = acc5__c0_0;
        }
        if (a1_b * 64 + a1_u * 8 + 1 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 1) % 32 * 64 + (a6_b * 16 + a6_u)] = acc5__c1_0;
        }
        if (a1_b * 64 + a1_u * 8 + 2 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 2) % 32 * 64 + (a6_b * 16 + a6_u)] = acc5__c2_0;
        }
        if (a1_b * 64 + a1_u * 8 + 3 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 3) % 32 * 64 + (a6_b * 16 + a6_u)] = acc5__c3_0;
        }
        if (a1_b * 64 + a1_u * 8 + 4 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 4) % 32 * 64 + (a6_b * 16 + a6_u)] = acc5__c4_0;
        }
        if (a1_b * 64 + a1_u * 8 + 5 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 5) % 32 * 64 + (a6_b * 16 + a6_u)] = acc5__c5_0;
        }
        if (a1_b * 64 + a1_u * 8 + 6 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 6) % 32 * 64 + (a6_b * 16 + a6_u)] = acc5__c6_0;
        }
        if (a1_b * 64 + a1_u * 8 + 7 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 64 + a1_u * 8 + 7) % 32 * 64 + (a6_b * 16 + a6_u)] = acc5__c7_0;
        }
}
