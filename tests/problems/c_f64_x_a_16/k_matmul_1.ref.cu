extern "C" __global__
__launch_bounds__(32) void k_matmul_1(const double* div_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a6_ks = _gid / 8192;
        int a0 = (_gid / 512) % 16;
        int a5 = (_gid / 32) % 16;
        int a6_co = (_gid) % 32;
        float in0 = div_c1[0];
        float v0 = 1.0f / in0;
        float v1 = 1.0f / in0;
        float acc1 = -1e+30f;
        float acc1__r1 = -1e+30f;
        for (int a1 = 0; a1 < 16; a1 += 2) {
            float acc0 = 0.0f;
            for (int a2 = 0; a2 < 16; a2++) {
                float in2 = q[a0 * 16 + a2];
                float in1 = kt[a2 * 16 + a1];
                float v3 = in1 * in2;
                acc0 += v3;
            }
            float v4 = acc0;
            float v5 = v1 * v4;
            acc1 = fmaxf(acc1, v5);
            float acc0__r1 = 0.0f;
            for (int a2 = 0; a2 < 16; a2++) {
                float in2__r1 = q[a0 * 16 + a2];
                float in1__r1 = kt[a2 * 16 + (a1 + 1)];
                float v3__r1 = in1__r1 * in2__r1;
                acc0__r1 += v3__r1;
            }
            float v4__r1 = acc0__r1;
            float v5__r1 = v1 * v4__r1;
            acc1__r1 = fmaxf(acc1__r1, v5__r1);
        }
        acc1 = fmaxf(acc1, acc1__r1);
        float acc3 = 0.0f;
        for (int a1 = 0; a1 < 16; a1++) {
            float acc2 = 0.0f;
            for (int a4 = 0; a4 < 16; a4++) {
                float in4 = q[a0 * 16 + a4];
                float in3 = kt[a4 * 16 + a1];
                float v6 = in3 * in4;
                acc2 += v6;
            }
            float v7 = acc2;
            float v8 = v0 * v7;
            float v9 = v8 - acc1;
            float v10 = expf(v9);
            acc3 += v10;
        }
        float v11 = 1.0f / acc3;
        float v2 = 1.0f / in0;
        float acc5 = 0.0f;
        for (int a6 = a6_co; a6 < 4; a6 += 32) {
            float acc4 = 0.0f;
            for (int a7 = 0; a7 < 16; a7++) {
                float in6 = q[a0 * 16 + a7];
                float in5 = kt[a7 * 16 + (_a6_ks * 4 + a6)];
                float v12 = in5 * in6;
                acc4 += v12;
            }
            float v13 = acc4;
            float v14 = v13 * v2;
            float v15 = v14 - acc1;
            float v17 = v11 * expf(v15);
            float in7 = v[(_a6_ks * 4 + a6) * 16 + a5];
            float v18 = in7 * v17;
            acc5 += v18;
        }
        {
            float acc5__o = __shfl_xor_sync(__activemask(), acc5, 16);
            acc5 = acc5 + acc5__o;
        }
        {
            float acc5__o = __shfl_xor_sync(__activemask(), acc5, 8);
            acc5 = acc5 + acc5__o;
        }
        {
            float acc5__o = __shfl_xor_sync(__activemask(), acc5, 4);
            acc5 = acc5 + acc5__o;
        }
        {
            float acc5__o = __shfl_xor_sync(__activemask(), acc5, 2);
            acc5 = acc5 + acc5__o;
        }
        {
            float acc5__o = __shfl_xor_sync(__activemask(), acc5, 1);
            acc5 = acc5 + acc5__o;
        }
        if (a6_co == 0) {
            atomicAdd(&matmul_1[a0 * 16 + a5], acc5);
        }
}
