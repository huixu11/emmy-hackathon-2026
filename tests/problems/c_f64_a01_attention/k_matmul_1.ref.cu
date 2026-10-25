extern "C" __global__
__launch_bounds__(128) void k_matmul_1(const double* div_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a6_ks = _gid / 512;
        int a0_b = (_gid / 256) % 2;
        int a5_b = (_gid / 128) % 2;
        int a0_u = (_gid / 16) % 8;
        int a5_u = (_gid) % 16;
        float in0__ar0 = div_c1[0];
        float v0__ar0 = 1.0f / in0__ar0;
        float v1__ar0 = 1.0f / in0__ar0;
        float acc1__ar0 = -1e+30f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar0 = q[(a0_b * 32 + a0_u * 4) * 64 + a2];
                float in1__ar0 = kt[a2 * 64 + a1];
                float v3__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v3__ar0;
            }
            float v4__ar0 = acc0__ar0;
            float v5__ar0 = v1__ar0 * v4__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v5__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc2__ar0 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar0 = q[(a0_b * 32 + a0_u * 4) * 64 + a4];
                float in3__ar0 = kt[a4 * 64 + a1];
                float v6__ar0 = in3__ar0 * in4__ar0;
                acc2__ar0 += v6__ar0;
            }
            float v7__ar0 = acc2__ar0;
            float v8__ar0 = v0__ar0 * v7__ar0;
            float v9__ar0 = v8__ar0 - acc1__ar0;
            float v10__ar0 = expf(v9__ar0);
            acc3__ar0 += v10__ar0;
        }
        float v11__ar0 = 1.0f / acc3__ar0;
        float v2__ar0 = 1.0f / in0__ar0;
        float in0__ar1 = div_c1[0];
        float v0__ar1 = 1.0f / in0__ar1;
        float v1__ar1 = 1.0f / in0__ar1;
        float acc1__ar1 = -1e+30f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar1 = q[(a0_b * 32 + a0_u * 4 + 1) * 64 + a2];
                float in1__ar1 = kt[a2 * 64 + a1];
                float v3__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v3__ar1;
            }
            float v4__ar1 = acc0__ar1;
            float v5__ar1 = v1__ar1 * v4__ar1;
            acc1__ar1 = fmaxf(acc1__ar1, v5__ar1);
        }
        float acc3__ar1 = 0.0f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc2__ar1 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar1 = q[(a0_b * 32 + a0_u * 4 + 1) * 64 + a4];
                float in3__ar1 = kt[a4 * 64 + a1];
                float v6__ar1 = in3__ar1 * in4__ar1;
                acc2__ar1 += v6__ar1;
            }
            float v7__ar1 = acc2__ar1;
            float v8__ar1 = v0__ar1 * v7__ar1;
            float v9__ar1 = v8__ar1 - acc1__ar1;
            float v10__ar1 = expf(v9__ar1);
            acc3__ar1 += v10__ar1;
        }
        float v11__ar1 = 1.0f / acc3__ar1;
        float v2__ar1 = 1.0f / in0__ar1;
        float in0__ar2 = div_c1[0];
        float v0__ar2 = 1.0f / in0__ar2;
        float v1__ar2 = 1.0f / in0__ar2;
        float acc1__ar2 = -1e+30f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__ar2 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar2 = q[(a0_b * 32 + a0_u * 4 + 2) * 64 + a2];
                float in1__ar2 = kt[a2 * 64 + a1];
                float v3__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v3__ar2;
            }
            float v4__ar2 = acc0__ar2;
            float v5__ar2 = v1__ar2 * v4__ar2;
            acc1__ar2 = fmaxf(acc1__ar2, v5__ar2);
        }
        float acc3__ar2 = 0.0f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc2__ar2 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar2 = q[(a0_b * 32 + a0_u * 4 + 2) * 64 + a4];
                float in3__ar2 = kt[a4 * 64 + a1];
                float v6__ar2 = in3__ar2 * in4__ar2;
                acc2__ar2 += v6__ar2;
            }
            float v7__ar2 = acc2__ar2;
            float v8__ar2 = v0__ar2 * v7__ar2;
            float v9__ar2 = v8__ar2 - acc1__ar2;
            float v10__ar2 = expf(v9__ar2);
            acc3__ar2 += v10__ar2;
        }
        float v11__ar2 = 1.0f / acc3__ar2;
        float v2__ar2 = 1.0f / in0__ar2;
        float in0__ar3 = div_c1[0];
        float v0__ar3 = 1.0f / in0__ar3;
        float v1__ar3 = 1.0f / in0__ar3;
        float acc1__ar3 = -1e+30f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__ar3 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar3 = q[(a0_b * 32 + a0_u * 4 + 3) * 64 + a2];
                float in1__ar3 = kt[a2 * 64 + a1];
                float v3__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v3__ar3;
            }
            float v4__ar3 = acc0__ar3;
            float v5__ar3 = v1__ar3 * v4__ar3;
            acc1__ar3 = fmaxf(acc1__ar3, v5__ar3);
        }
        float acc3__ar3 = 0.0f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc2__ar3 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar3 = q[(a0_b * 32 + a0_u * 4 + 3) * 64 + a4];
                float in3__ar3 = kt[a4 * 64 + a1];
                float v6__ar3 = in3__ar3 * in4__ar3;
                acc2__ar3 += v6__ar3;
            }
            float v7__ar3 = acc2__ar3;
            float v8__ar3 = v0__ar3 * v7__ar3;
            float v9__ar3 = v8__ar3 - acc1__ar3;
            float v10__ar3 = expf(v9__ar3);
            acc3__ar3 += v10__ar3;
        }
        float v11__ar3 = 1.0f / acc3__ar3;
        float v2__ar3 = 1.0f / in0__ar3;
        float acc5__c0_0 = 0.0f;
        float acc5__c0_1 = 0.0f;
        float acc5__c1_0 = 0.0f;
        float acc5__c1_1 = 0.0f;
        float acc5__c2_0 = 0.0f;
        float acc5__c2_1 = 0.0f;
        float acc5__c3_0 = 0.0f;
        float acc5__c3_1 = 0.0f;
        #pragma unroll
        for (int a6 = 0; a6 < 32; a6++) {
            float acc4__ar0 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar0 = q[(a0_b * 32 + a0_u * 4) * 64 + a7];
                float in5__ar0 = kt[a7 * 64 + (_a6_ks * 32 + a6)];
                float v12__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v12__ar0;
            }
            float v13__ar0 = acc4__ar0;
            float v14__ar0 = v13__ar0 * v2__ar0;
            float v15__ar0 = v14__ar0 - acc1__ar0;
            float v17__ar0 = v11__ar0 * expf(v15__ar0);
            float acc4__ar1 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar1 = q[(a0_b * 32 + a0_u * 4 + 1) * 64 + a7];
                float in5__ar1 = kt[a7 * 64 + (_a6_ks * 32 + a6)];
                float v12__ar1 = in5__ar1 * in6__ar1;
                acc4__ar1 += v12__ar1;
            }
            float v13__ar1 = acc4__ar1;
            float v14__ar1 = v13__ar1 * v2__ar1;
            float v15__ar1 = v14__ar1 - acc1__ar1;
            float v17__ar1 = v11__ar1 * expf(v15__ar1);
            float acc4__ar2 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar2 = q[(a0_b * 32 + a0_u * 4 + 2) * 64 + a7];
                float in5__ar2 = kt[a7 * 64 + (_a6_ks * 32 + a6)];
                float v12__ar2 = in5__ar2 * in6__ar2;
                acc4__ar2 += v12__ar2;
            }
            float v13__ar2 = acc4__ar2;
            float v14__ar2 = v13__ar2 * v2__ar2;
            float v15__ar2 = v14__ar2 - acc1__ar2;
            float v17__ar2 = v11__ar2 * expf(v15__ar2);
            float acc4__ar3 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar3 = q[(a0_b * 32 + a0_u * 4 + 3) * 64 + a7];
                float in5__ar3 = kt[a7 * 64 + (_a6_ks * 32 + a6)];
                float v12__ar3 = in5__ar3 * in6__ar3;
                acc4__ar3 += v12__ar3;
            }
            float v13__ar3 = acc4__ar3;
            float v14__ar3 = v13__ar3 * v2__ar3;
            float v15__ar3 = v14__ar3 - acc1__ar3;
            float v17__ar3 = v11__ar3 * expf(v15__ar3);
            float in7__bc0 = v[(_a6_ks * 32 + a6) * 64 + (a5_b * 32 + a5_u * 2)];
            float v18__c0_0 = in7__bc0 * v17__ar0;
            acc5__c0_0 += v18__c0_0;
            float in7__bc1 = v[(_a6_ks * 32 + a6) * 64 + (a5_b * 32 + a5_u * 2 + 1)];
            float v18__c0_1 = in7__bc1 * v17__ar0;
            acc5__c0_1 += v18__c0_1;
            float v18__c1_0 = in7__bc0 * v17__ar1;
            acc5__c1_0 += v18__c1_0;
            float v18__c1_1 = in7__bc1 * v17__ar1;
            acc5__c1_1 += v18__c1_1;
            float v18__c2_0 = in7__bc0 * v17__ar2;
            acc5__c2_0 += v18__c2_0;
            float v18__c2_1 = in7__bc1 * v17__ar2;
            acc5__c2_1 += v18__c2_1;
            float v18__c3_0 = in7__bc0 * v17__ar3;
            acc5__c3_0 += v18__c3_0;
            float v18__c3_1 = in7__bc1 * v17__ar3;
            acc5__c3_1 += v18__c3_1;
        }
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 4) * 64 + (a5_b * 32 + a5_u * 2)], acc5__c0_0);
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 4) * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c0_1);
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 4 + 1) * 64 + (a5_b * 32 + a5_u * 2)], acc5__c1_0);
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 4 + 1) * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c1_1);
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 4 + 2) * 64 + (a5_b * 32 + a5_u * 2)], acc5__c2_0);
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 4 + 2) * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c2_1);
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 4 + 3) * 64 + (a5_b * 32 + a5_u * 2)], acc5__c3_0);
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 4 + 3) * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c3_1);
}
