extern "C" __global__
__launch_bounds__(256) void k_matmul_1(const double* div_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1408) {
        int _a6_ks = _gid / 704;
        int a0_b = (_gid / 22) % 32;
        int a5_b = (_gid) % 22;
        int a0_u = (_gid) % 1;
        int a5_u = (_gid) % 1;
        float in0__ar0 = div_c1[0];
        float v0__ar0 = 1.0f / in0__ar0;
        float v1__ar0 = 1.0f / in0__ar0;
        float acc1__ar0 = -1e+30f;
        for (int a1 = 0; a1 < 1024; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar0 = q[(a0_b + a0_u) * 64 + a2];
                float in1__ar0 = kt[a2 * 1024 + a1];
                float v3__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v3__ar0;
            }
            float v4__ar0 = acc0__ar0;
            float v5__ar0 = v1__ar0 * v4__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v5__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a1 = 0; a1 < 1024; a1++) {
            float acc2__ar0 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar0 = q[(a0_b + a0_u) * 64 + a4];
                float in3__ar0 = kt[a4 * 1024 + a1];
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
        float acc5__c0_0 = 0.0f;
        float acc5__c0_1 = 0.0f;
        float acc5__c0_2 = 0.0f;
        for (int a6 = 0; a6 < 512; a6++) {
            float acc4__ar0 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar0 = q[(a0_b + a0_u) * 64 + a7];
                float in5__ar0 = kt[a7 * 1024 + (_a6_ks * 512 + a6)];
                float v12__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v12__ar0;
            }
            float v13__ar0 = acc4__ar0;
            float v14__ar0 = v13__ar0 * v2__ar0;
            float v15__ar0 = v14__ar0 - acc1__ar0;
            float v17__ar0 = v11__ar0 * expf(v15__ar0);
            float in7__bc0 = v[(_a6_ks * 512 + a6) * 64 + (a5_b * 3 + a5_u * 3) % 64];
            float v18__c0_0 = in7__bc0 * v17__ar0;
            acc5__c0_0 += v18__c0_0;
            float in7__bc1 = v[(_a6_ks * 512 + a6) * 64 + (a5_b * 3 + a5_u * 3 + 1) % 64];
            float v18__c0_1 = in7__bc1 * v17__ar0;
            acc5__c0_1 += v18__c0_1;
            float in7__bc2 = v[(_a6_ks * 512 + a6) * 64 + (a5_b * 3 + a5_u * 3 + 2) % 64];
            float v18__c0_2 = in7__bc2 * v17__ar0;
            acc5__c0_2 += v18__c0_2;
        }
        if (a5_b * 3 + a5_u * 3 + 0 < 64) {
            atomicAdd(&matmul_1[(a0_b + a0_u) * 64 + (a5_b * 3 + a5_u * 3) % 64], acc5__c0_0);
        }
        if (a5_b * 3 + a5_u * 3 + 1 < 64) {
            atomicAdd(&matmul_1[(a0_b + a0_u) * 64 + (a5_b * 3 + a5_u * 3 + 1) % 64], acc5__c0_1);
        }
        if (a5_b * 3 + a5_u * 3 + 2 < 64) {
            atomicAdd(&matmul_1[(a0_b + a0_u) * 64 + (a5_b * 3 + a5_u * 3 + 2) % 64], acc5__c0_2);
        }
    }
}
