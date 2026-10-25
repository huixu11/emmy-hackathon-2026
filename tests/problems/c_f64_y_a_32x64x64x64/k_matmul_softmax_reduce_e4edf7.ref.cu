extern "C" __global__
__launch_bounds__(32) void k_matmul_softmax_reduce_e4edf7(const double* mul_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 64;
        int a5_b = (_gid / 32) % 2;
        int a0_u = (_gid / 32) % 1;
        int a5_u = (_gid) % 32;
        float in0__ar0 = mul_c1[0];
        float acc1__ar0 = -1e+30f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar0 = q[(a0_b + a0_u) * 64 + a2];
                float in1__ar0 = kt[a2 * 64 + a1];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0;
            float v2__ar0 = in0__ar0 * v1__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v2__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a1 = 0; a1 < 64; a1++) {
            float acc2__ar0 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar0 = q[(a0_b + a0_u) * 64 + a4];
                float in3__ar0 = kt[a4 * 64 + a1];
                float v3__ar0 = in3__ar0 * in4__ar0;
                acc2__ar0 += v3__ar0;
            }
            float v4__ar0 = acc2__ar0;
            float v7__ar0 = expf(((in0__ar0 * v4__ar0) - acc1__ar0));
            acc3__ar0 += v7__ar0;
        }
        float v8__ar0 = 1.0f / acc3__ar0;
        float acc5__c0_0 = 0.0f;
        #pragma unroll
        for (int a6 = 0; a6 < 64; a6++) {
            float acc4__ar0 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar0 = q[(a0_b + a0_u) * 64 + a7];
                float in5__ar0 = kt[a7 * 64 + a6];
                float v9__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v9__ar0;
            }
            float v10__ar0 = acc4__ar0;
            float v14__ar0 = expf(((in0__ar0 * v10__ar0) - acc1__ar0)) * v8__ar0;
            float in7__bc0 = v[a6 * 64 + (a5_b * 32 + a5_u)];
            float v15__c0_0 = in7__bc0 * v14__ar0;
            acc5__c0_0 += v15__c0_0;
        }
        matmul_1[(a0_b + a0_u) * 64 + (a5_b * 32 + a5_u)] = acc5__c0_0;
}
