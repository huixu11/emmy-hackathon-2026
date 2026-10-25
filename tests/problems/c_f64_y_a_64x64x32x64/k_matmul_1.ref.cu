extern "C" __global__
__launch_bounds__(256) void k_matmul_1(const double* mul_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a6_ks = _gid / 2048;
        int _rsub = (_gid / 256) % 8;
        int a0_b = (_rsub / 8) * 2 + _rsub % 2;
        int a5_b = (_rsub % 8) / 2;
        int a0_u = (_gid / 16) % 16;
        int a5_u = (_gid) % 16;
        float in0__ar0 = mul_c1[0];
        float acc1__ar0 = -1e+30f;
        for (int a1 = 0; a1 < 32; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar0 = q[(a0_b * 32 + a0_u * 2) * 64 + a2];
                float in1__ar0 = kt[a2 * 32 + a1];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0;
            float v2__ar0 = in0__ar0 * v1__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v2__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a1 = 0; a1 < 32; a1++) {
            float acc2__ar0 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar0 = q[(a0_b * 32 + a0_u * 2) * 64 + a4];
                float in3__ar0 = kt[a4 * 32 + a1];
                float v3__ar0 = in3__ar0 * in4__ar0;
                acc2__ar0 += v3__ar0;
            }
            float v4__ar0 = acc2__ar0;
            float v7__ar0 = expf(((in0__ar0 * v4__ar0) - acc1__ar0));
            acc3__ar0 += v7__ar0;
        }
        float v8__ar0 = 1.0f / acc3__ar0;
        float in0__ar1 = mul_c1[0];
        float acc1__ar1 = -1e+30f;
        for (int a1 = 0; a1 < 32; a1++) {
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar1 = q[(a0_b * 32 + a0_u * 2 + 1) * 64 + a2];
                float in1__ar1 = kt[a2 * 32 + a1];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__ar1 = acc0__ar1;
            float v2__ar1 = in0__ar1 * v1__ar1;
            acc1__ar1 = fmaxf(acc1__ar1, v2__ar1);
        }
        float acc3__ar1 = 0.0f;
        for (int a1 = 0; a1 < 32; a1++) {
            float acc2__ar1 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar1 = q[(a0_b * 32 + a0_u * 2 + 1) * 64 + a4];
                float in3__ar1 = kt[a4 * 32 + a1];
                float v3__ar1 = in3__ar1 * in4__ar1;
                acc2__ar1 += v3__ar1;
            }
            float v4__ar1 = acc2__ar1;
            float v7__ar1 = expf(((in0__ar1 * v4__ar1) - acc1__ar1));
            acc3__ar1 += v7__ar1;
        }
        float v8__ar1 = 1.0f / acc3__ar1;
        float acc5__c0_0 = 0.0f;
        float acc5__c1_0 = 0.0f;
        #pragma unroll
        for (int a6 = 0; a6 < 16; a6++) {
            float acc4__ar0 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar0 = q[(a0_b * 32 + a0_u * 2) * 64 + a7];
                float in5__ar0 = kt[a7 * 32 + (_a6_ks * 16 + a6)];
                float v9__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v9__ar0;
            }
            float v10__ar0 = acc4__ar0;
            float v14__ar0 = expf(((in0__ar0 * v10__ar0) - acc1__ar0)) * v8__ar0;
            float acc4__ar1 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar1 = q[(a0_b * 32 + a0_u * 2 + 1) * 64 + a7];
                float in5__ar1 = kt[a7 * 32 + (_a6_ks * 16 + a6)];
                float v9__ar1 = in5__ar1 * in6__ar1;
                acc4__ar1 += v9__ar1;
            }
            float v10__ar1 = acc4__ar1;
            float v14__ar1 = expf(((in0__ar1 * v10__ar1) - acc1__ar1)) * v8__ar1;
            float in7__bc0 = v[(_a6_ks * 16 + a6) * 64 + (a5_b * 16 + a5_u)];
            float v15__c0_0 = in7__bc0 * v14__ar0;
            acc5__c0_0 += v15__c0_0;
            float v15__c1_0 = in7__bc0 * v14__ar1;
            acc5__c1_0 += v15__c1_0;
        }
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 2) * 64 + (a5_b * 16 + a5_u)], acc5__c0_0);
        atomicAdd(&matmul_1[(a0_b * 32 + a0_u * 2 + 1) * 64 + (a5_b * 16 + a5_u)], acc5__c1_0);
}
