extern "C" __global__
__launch_bounds__(32) void k_matmul_softmax_reduce_f03c9a(const double* mul_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 1024;
        int a5 = (_gid / 32) % 32;
        int a6_co = (_gid) % 32;
        float in0 = mul_c1[0];
        float acc1 = -1e+30f;
        float acc1__r1 = -1e+30f;
        float acc1__r2 = -1e+30f;
        float acc1__r3 = -1e+30f;
        for (int a1 = 0; a1 < 48; a1 += 4) {
            float acc0 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2 = q[a0 * 32 + a2];
                float in1 = kt[a2 * 48 + a1];
                float v0 = in1 * in2;
                acc0 += v0;
            }
            float v1 = acc0;
            float v2 = in0 * v1;
            acc1 = fmaxf(acc1, v2);
            float acc0__r1 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__r1 = q[a0 * 32 + a2];
                float in1__r1 = kt[a2 * 48 + (a1 + 1)];
                float v0__r1 = in1__r1 * in2__r1;
                acc0__r1 += v0__r1;
            }
            float v1__r1 = acc0__r1;
            float v2__r1 = in0 * v1__r1;
            acc1__r1 = fmaxf(acc1__r1, v2__r1);
            float acc0__r2 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__r2 = q[a0 * 32 + a2];
                float in1__r2 = kt[a2 * 48 + (a1 + 2)];
                float v0__r2 = in1__r2 * in2__r2;
                acc0__r2 += v0__r2;
            }
            float v1__r2 = acc0__r2;
            float v2__r2 = in0 * v1__r2;
            acc1__r2 = fmaxf(acc1__r2, v2__r2);
            float acc0__r3 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__r3 = q[a0 * 32 + a2];
                float in1__r3 = kt[a2 * 48 + (a1 + 3)];
                float v0__r3 = in1__r3 * in2__r3;
                acc0__r3 += v0__r3;
            }
            float v1__r3 = acc0__r3;
            float v2__r3 = in0 * v1__r3;
            acc1__r3 = fmaxf(acc1__r3, v2__r3);
        }
        acc1 = fmaxf(acc1, acc1__r1);
        acc1 = fmaxf(acc1, acc1__r2);
        acc1 = fmaxf(acc1, acc1__r3);
        float acc3 = 0.0f;
        float acc3__r1 = 0.0f;
        float acc3__r2 = 0.0f;
        float acc3__r3 = 0.0f;
        for (int a1 = 0; a1 < 48; a1 += 4) {
            float acc2 = 0.0f;
            for (int a4 = 0; a4 < 32; a4++) {
                float in4 = q[a0 * 32 + a4];
                float in3 = kt[a4 * 48 + a1];
                float v3 = in3 * in4;
                acc2 += v3;
            }
            float v4 = acc2;
            float v6 = (in0 * v4) - acc1;
            float v7 = expf(v6);
            acc3 += v7;
            float acc2__r1 = 0.0f;
            for (int a4 = 0; a4 < 32; a4++) {
                float in4__r1 = q[a0 * 32 + a4];
                float in3__r1 = kt[a4 * 48 + (a1 + 1)];
                float v3__r1 = in3__r1 * in4__r1;
                acc2__r1 += v3__r1;
            }
            float v4__r1 = acc2__r1;
            float v6__r1 = (in0 * v4__r1) - acc1;
            float v7__r1 = expf(v6__r1);
            acc3__r1 += v7__r1;
            float acc2__r2 = 0.0f;
            for (int a4 = 0; a4 < 32; a4++) {
                float in4__r2 = q[a0 * 32 + a4];
                float in3__r2 = kt[a4 * 48 + (a1 + 2)];
                float v3__r2 = in3__r2 * in4__r2;
                acc2__r2 += v3__r2;
            }
            float v4__r2 = acc2__r2;
            float v6__r2 = (in0 * v4__r2) - acc1;
            float v7__r2 = expf(v6__r2);
            acc3__r2 += v7__r2;
            float acc2__r3 = 0.0f;
            for (int a4 = 0; a4 < 32; a4++) {
                float in4__r3 = q[a0 * 32 + a4];
                float in3__r3 = kt[a4 * 48 + (a1 + 3)];
                float v3__r3 = in3__r3 * in4__r3;
                acc2__r3 += v3__r3;
            }
            float v4__r3 = acc2__r3;
            float v6__r3 = (in0 * v4__r3) - acc1;
            float v7__r3 = expf(v6__r3);
            acc3__r3 += v7__r3;
        }
        acc3 += acc3__r1;
        acc3 += acc3__r2;
        acc3 += acc3__r3;
        float v8 = 1.0f / acc3;
        float acc5 = 0.0f;
        float acc5__r1 = 0.0f;
        float acc5__r2 = 0.0f;
        float acc5__r3 = 0.0f;
        for (int a6 = a6_co; a6 < 48; a6 += 128) {
            float acc4 = 0.0f;
            for (int a7 = 0; a7 < 32; a7++) {
                float in6 = q[a0 * 32 + a7];
                float in5 = kt[a7 * 48 + a6];
                float v9 = in5 * in6;
                acc4 += v9;
            }
            float v10 = acc4;
            float v12 = (in0 * v10) - acc1;
            float v14 = expf(v12) * v8;
            float in7 = v[a6 * 32 + a5];
            float v15 = in7 * v14;
            acc5 += v15;
            float acc4__r1 = 0.0f;
            for (int a7 = 0; a7 < 32; a7++) {
                float in6__r1 = q[a0 * 32 + a7];
                float in5__r1 = kt[a7 * 48 + (a6 + 32) % 48];
                float v9__r1 = in5__r1 * in6__r1;
                acc4__r1 += v9__r1;
            }
            float v10__r1 = acc4__r1;
            float v12__r1 = (in0 * v10__r1) - acc1;
            float v14__r1 = expf(v12__r1) * v8;
            float in7__r1 = v[(a6 + 32) % 48 * 32 + a5];
            float v15__r1 = in7__r1 * v14__r1;
            float v15__r1__id = 0.0f;
            float v15__r1__m = ((a6 + 32 < 48) ? (((float)(v15__r1))) : (((float)(v15__r1__id))));
            acc5__r1 += v15__r1__m;
            float acc4__r2 = 0.0f;
            for (int a7 = 0; a7 < 32; a7++) {
                float in6__r2 = q[a0 * 32 + a7];
                float in5__r2 = kt[a7 * 48 + (a6 + 64) % 48];
                float v9__r2 = in5__r2 * in6__r2;
                acc4__r2 += v9__r2;
            }
            float v10__r2 = acc4__r2;
            float v12__r2 = (in0 * v10__r2) - acc1;
            float v14__r2 = expf(v12__r2) * v8;
            float in7__r2 = v[(a6 + 64) % 48 * 32 + a5];
            float v15__r2 = in7__r2 * v14__r2;
            float v15__r2__id = 0.0f;
            float v15__r2__m = ((a6 + 64 < 48) ? (((float)(v15__r2))) : (((float)(v15__r2__id))));
            acc5__r2 += v15__r2__m;
            float acc4__r3 = 0.0f;
            for (int a7 = 0; a7 < 32; a7++) {
                float in6__r3 = q[a0 * 32 + a7];
                float in5__r3 = kt[a7 * 48 + (a6 + 96) % 48];
                float v9__r3 = in5__r3 * in6__r3;
                acc4__r3 += v9__r3;
            }
            float v10__r3 = acc4__r3;
            float v12__r3 = (in0 * v10__r3) - acc1;
            float v14__r3 = expf(v12__r3) * v8;
            float in7__r3 = v[(a6 + 96) % 48 * 32 + a5];
            float v15__r3 = in7__r3 * v14__r3;
            float v15__r3__id = 0.0f;
            float v15__r3__m = ((a6 + 96 < 48) ? (((float)(v15__r3))) : (((float)(v15__r3__id))));
            acc5__r3 += v15__r3__m;
        }
        acc5 += acc5__r1;
        acc5 += acc5__r2;
        acc5 += acc5__r3;
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
            matmul_1[a0 * 32 + a5] = acc5;
        }
}
