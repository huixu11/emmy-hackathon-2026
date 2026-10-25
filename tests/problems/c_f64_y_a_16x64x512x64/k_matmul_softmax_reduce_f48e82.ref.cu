extern "C" __global__
__launch_bounds__(128) void k_matmul_softmax_reduce_f48e82(const double* mul_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 512;
        int a5_b = (_gid / 128) % 4;
        int a0_u = (_gid / 16) % 8;
        int a5_u = (_gid) % 16;
        float in0__ar0 = mul_c1[0];
        float acc1__ar0 = -1e+30f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar0 = q[(a0_b * 64 + a0_u * 8) % 16 * 64 + a2];
                float in1__ar0 = kt[a2 * 512 + a1];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0;
            float v2__ar0 = in0__ar0 * v1__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v2__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc2__ar0 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar0 = q[(a0_b * 64 + a0_u * 8) % 16 * 64 + a4];
                float in3__ar0 = kt[a4 * 512 + a1];
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
        for (int a1 = 0; a1 < 512; a1++) {
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar1 = q[(a0_b * 64 + a0_u * 8 + 1) % 16 * 64 + a2];
                float in1__ar1 = kt[a2 * 512 + a1];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__ar1 = acc0__ar1;
            float v2__ar1 = in0__ar1 * v1__ar1;
            acc1__ar1 = fmaxf(acc1__ar1, v2__ar1);
        }
        float acc3__ar1 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc2__ar1 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar1 = q[(a0_b * 64 + a0_u * 8 + 1) % 16 * 64 + a4];
                float in3__ar1 = kt[a4 * 512 + a1];
                float v3__ar1 = in3__ar1 * in4__ar1;
                acc2__ar1 += v3__ar1;
            }
            float v4__ar1 = acc2__ar1;
            float v7__ar1 = expf(((in0__ar1 * v4__ar1) - acc1__ar1));
            acc3__ar1 += v7__ar1;
        }
        float v8__ar1 = 1.0f / acc3__ar1;
        float in0__ar2 = mul_c1[0];
        float acc1__ar2 = -1e+30f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc0__ar2 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar2 = q[(a0_b * 64 + a0_u * 8 + 2) % 16 * 64 + a2];
                float in1__ar2 = kt[a2 * 512 + a1];
                float v0__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v0__ar2;
            }
            float v1__ar2 = acc0__ar2;
            float v2__ar2 = in0__ar2 * v1__ar2;
            acc1__ar2 = fmaxf(acc1__ar2, v2__ar2);
        }
        float acc3__ar2 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc2__ar2 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar2 = q[(a0_b * 64 + a0_u * 8 + 2) % 16 * 64 + a4];
                float in3__ar2 = kt[a4 * 512 + a1];
                float v3__ar2 = in3__ar2 * in4__ar2;
                acc2__ar2 += v3__ar2;
            }
            float v4__ar2 = acc2__ar2;
            float v7__ar2 = expf(((in0__ar2 * v4__ar2) - acc1__ar2));
            acc3__ar2 += v7__ar2;
        }
        float v8__ar2 = 1.0f / acc3__ar2;
        float in0__ar3 = mul_c1[0];
        float acc1__ar3 = -1e+30f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc0__ar3 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar3 = q[(a0_b * 64 + a0_u * 8 + 3) % 16 * 64 + a2];
                float in1__ar3 = kt[a2 * 512 + a1];
                float v0__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v0__ar3;
            }
            float v1__ar3 = acc0__ar3;
            float v2__ar3 = in0__ar3 * v1__ar3;
            acc1__ar3 = fmaxf(acc1__ar3, v2__ar3);
        }
        float acc3__ar3 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc2__ar3 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar3 = q[(a0_b * 64 + a0_u * 8 + 3) % 16 * 64 + a4];
                float in3__ar3 = kt[a4 * 512 + a1];
                float v3__ar3 = in3__ar3 * in4__ar3;
                acc2__ar3 += v3__ar3;
            }
            float v4__ar3 = acc2__ar3;
            float v7__ar3 = expf(((in0__ar3 * v4__ar3) - acc1__ar3));
            acc3__ar3 += v7__ar3;
        }
        float v8__ar3 = 1.0f / acc3__ar3;
        float in0__ar4 = mul_c1[0];
        float acc1__ar4 = -1e+30f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc0__ar4 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar4 = q[(a0_b * 64 + a0_u * 8 + 4) % 16 * 64 + a2];
                float in1__ar4 = kt[a2 * 512 + a1];
                float v0__ar4 = in1__ar4 * in2__ar4;
                acc0__ar4 += v0__ar4;
            }
            float v1__ar4 = acc0__ar4;
            float v2__ar4 = in0__ar4 * v1__ar4;
            acc1__ar4 = fmaxf(acc1__ar4, v2__ar4);
        }
        float acc3__ar4 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc2__ar4 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar4 = q[(a0_b * 64 + a0_u * 8 + 4) % 16 * 64 + a4];
                float in3__ar4 = kt[a4 * 512 + a1];
                float v3__ar4 = in3__ar4 * in4__ar4;
                acc2__ar4 += v3__ar4;
            }
            float v4__ar4 = acc2__ar4;
            float v7__ar4 = expf(((in0__ar4 * v4__ar4) - acc1__ar4));
            acc3__ar4 += v7__ar4;
        }
        float v8__ar4 = 1.0f / acc3__ar4;
        float in0__ar5 = mul_c1[0];
        float acc1__ar5 = -1e+30f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc0__ar5 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar5 = q[(a0_b * 64 + a0_u * 8 + 5) % 16 * 64 + a2];
                float in1__ar5 = kt[a2 * 512 + a1];
                float v0__ar5 = in1__ar5 * in2__ar5;
                acc0__ar5 += v0__ar5;
            }
            float v1__ar5 = acc0__ar5;
            float v2__ar5 = in0__ar5 * v1__ar5;
            acc1__ar5 = fmaxf(acc1__ar5, v2__ar5);
        }
        float acc3__ar5 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc2__ar5 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar5 = q[(a0_b * 64 + a0_u * 8 + 5) % 16 * 64 + a4];
                float in3__ar5 = kt[a4 * 512 + a1];
                float v3__ar5 = in3__ar5 * in4__ar5;
                acc2__ar5 += v3__ar5;
            }
            float v4__ar5 = acc2__ar5;
            float v7__ar5 = expf(((in0__ar5 * v4__ar5) - acc1__ar5));
            acc3__ar5 += v7__ar5;
        }
        float v8__ar5 = 1.0f / acc3__ar5;
        float in0__ar6 = mul_c1[0];
        float acc1__ar6 = -1e+30f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc0__ar6 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar6 = q[(a0_b * 64 + a0_u * 8 + 6) % 16 * 64 + a2];
                float in1__ar6 = kt[a2 * 512 + a1];
                float v0__ar6 = in1__ar6 * in2__ar6;
                acc0__ar6 += v0__ar6;
            }
            float v1__ar6 = acc0__ar6;
            float v2__ar6 = in0__ar6 * v1__ar6;
            acc1__ar6 = fmaxf(acc1__ar6, v2__ar6);
        }
        float acc3__ar6 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc2__ar6 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar6 = q[(a0_b * 64 + a0_u * 8 + 6) % 16 * 64 + a4];
                float in3__ar6 = kt[a4 * 512 + a1];
                float v3__ar6 = in3__ar6 * in4__ar6;
                acc2__ar6 += v3__ar6;
            }
            float v4__ar6 = acc2__ar6;
            float v7__ar6 = expf(((in0__ar6 * v4__ar6) - acc1__ar6));
            acc3__ar6 += v7__ar6;
        }
        float v8__ar6 = 1.0f / acc3__ar6;
        float in0__ar7 = mul_c1[0];
        float acc1__ar7 = -1e+30f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc0__ar7 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar7 = q[(a0_b * 64 + a0_u * 8 + 7) % 16 * 64 + a2];
                float in1__ar7 = kt[a2 * 512 + a1];
                float v0__ar7 = in1__ar7 * in2__ar7;
                acc0__ar7 += v0__ar7;
            }
            float v1__ar7 = acc0__ar7;
            float v2__ar7 = in0__ar7 * v1__ar7;
            acc1__ar7 = fmaxf(acc1__ar7, v2__ar7);
        }
        float acc3__ar7 = 0.0f;
        for (int a1 = 0; a1 < 512; a1++) {
            float acc2__ar7 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar7 = q[(a0_b * 64 + a0_u * 8 + 7) % 16 * 64 + a4];
                float in3__ar7 = kt[a4 * 512 + a1];
                float v3__ar7 = in3__ar7 * in4__ar7;
                acc2__ar7 += v3__ar7;
            }
            float v4__ar7 = acc2__ar7;
            float v7__ar7 = expf(((in0__ar7 * v4__ar7) - acc1__ar7));
            acc3__ar7 += v7__ar7;
        }
        float v8__ar7 = 1.0f / acc3__ar7;
        float acc5__c0_0 = 0.0f;
        float acc5__c1_0 = 0.0f;
        float acc5__c2_0 = 0.0f;
        float acc5__c3_0 = 0.0f;
        float acc5__c4_0 = 0.0f;
        float acc5__c5_0 = 0.0f;
        float acc5__c6_0 = 0.0f;
        float acc5__c7_0 = 0.0f;
        for (int a6 = 0; a6 < 512; a6++) {
            float acc4__ar0 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar0 = q[(a0_b * 64 + a0_u * 8) % 16 * 64 + a7];
                float in5__ar0 = kt[a7 * 512 + a6];
                float v9__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v9__ar0;
            }
            float v10__ar0 = acc4__ar0;
            float v14__ar0 = expf(((in0__ar0 * v10__ar0) - acc1__ar0)) * v8__ar0;
            float acc4__ar1 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar1 = q[(a0_b * 64 + a0_u * 8 + 1) % 16 * 64 + a7];
                float in5__ar1 = kt[a7 * 512 + a6];
                float v9__ar1 = in5__ar1 * in6__ar1;
                acc4__ar1 += v9__ar1;
            }
            float v10__ar1 = acc4__ar1;
            float v14__ar1 = expf(((in0__ar1 * v10__ar1) - acc1__ar1)) * v8__ar1;
            float acc4__ar2 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar2 = q[(a0_b * 64 + a0_u * 8 + 2) % 16 * 64 + a7];
                float in5__ar2 = kt[a7 * 512 + a6];
                float v9__ar2 = in5__ar2 * in6__ar2;
                acc4__ar2 += v9__ar2;
            }
            float v10__ar2 = acc4__ar2;
            float v14__ar2 = expf(((in0__ar2 * v10__ar2) - acc1__ar2)) * v8__ar2;
            float acc4__ar3 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar3 = q[(a0_b * 64 + a0_u * 8 + 3) % 16 * 64 + a7];
                float in5__ar3 = kt[a7 * 512 + a6];
                float v9__ar3 = in5__ar3 * in6__ar3;
                acc4__ar3 += v9__ar3;
            }
            float v10__ar3 = acc4__ar3;
            float v14__ar3 = expf(((in0__ar3 * v10__ar3) - acc1__ar3)) * v8__ar3;
            float acc4__ar4 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar4 = q[(a0_b * 64 + a0_u * 8 + 4) % 16 * 64 + a7];
                float in5__ar4 = kt[a7 * 512 + a6];
                float v9__ar4 = in5__ar4 * in6__ar4;
                acc4__ar4 += v9__ar4;
            }
            float v10__ar4 = acc4__ar4;
            float v14__ar4 = expf(((in0__ar4 * v10__ar4) - acc1__ar4)) * v8__ar4;
            float acc4__ar5 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar5 = q[(a0_b * 64 + a0_u * 8 + 5) % 16 * 64 + a7];
                float in5__ar5 = kt[a7 * 512 + a6];
                float v9__ar5 = in5__ar5 * in6__ar5;
                acc4__ar5 += v9__ar5;
            }
            float v10__ar5 = acc4__ar5;
            float v14__ar5 = expf(((in0__ar5 * v10__ar5) - acc1__ar5)) * v8__ar5;
            float acc4__ar6 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar6 = q[(a0_b * 64 + a0_u * 8 + 6) % 16 * 64 + a7];
                float in5__ar6 = kt[a7 * 512 + a6];
                float v9__ar6 = in5__ar6 * in6__ar6;
                acc4__ar6 += v9__ar6;
            }
            float v10__ar6 = acc4__ar6;
            float v14__ar6 = expf(((in0__ar6 * v10__ar6) - acc1__ar6)) * v8__ar6;
            float acc4__ar7 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in6__ar7 = q[(a0_b * 64 + a0_u * 8 + 7) % 16 * 64 + a7];
                float in5__ar7 = kt[a7 * 512 + a6];
                float v9__ar7 = in5__ar7 * in6__ar7;
                acc4__ar7 += v9__ar7;
            }
            float v10__ar7 = acc4__ar7;
            float v14__ar7 = expf(((in0__ar7 * v10__ar7) - acc1__ar7)) * v8__ar7;
            float in7__bc0 = v[a6 * 64 + (a5_b * 16 + a5_u)];
            float v15__c0_0 = in7__bc0 * v14__ar0;
            acc5__c0_0 += v15__c0_0;
            float v15__c1_0 = in7__bc0 * v14__ar1;
            acc5__c1_0 += v15__c1_0;
            float v15__c2_0 = in7__bc0 * v14__ar2;
            acc5__c2_0 += v15__c2_0;
            float v15__c3_0 = in7__bc0 * v14__ar3;
            acc5__c3_0 += v15__c3_0;
            float v15__c4_0 = in7__bc0 * v14__ar4;
            acc5__c4_0 += v15__c4_0;
            float v15__c5_0 = in7__bc0 * v14__ar5;
            acc5__c5_0 += v15__c5_0;
            float v15__c6_0 = in7__bc0 * v14__ar6;
            acc5__c6_0 += v15__c6_0;
            float v15__c7_0 = in7__bc0 * v14__ar7;
            acc5__c7_0 += v15__c7_0;
        }
        if (a0_b * 64 + a0_u * 8 + 0 < 16) {
            matmul_1[(a0_b * 64 + a0_u * 8) % 16 * 64 + (a5_b * 16 + a5_u)] = acc5__c0_0;
        }
        if (a0_b * 64 + a0_u * 8 + 1 < 16) {
            matmul_1[(a0_b * 64 + a0_u * 8 + 1) % 16 * 64 + (a5_b * 16 + a5_u)] = acc5__c1_0;
        }
        if (a0_b * 64 + a0_u * 8 + 2 < 16) {
            matmul_1[(a0_b * 64 + a0_u * 8 + 2) % 16 * 64 + (a5_b * 16 + a5_u)] = acc5__c2_0;
        }
        if (a0_b * 64 + a0_u * 8 + 3 < 16) {
            matmul_1[(a0_b * 64 + a0_u * 8 + 3) % 16 * 64 + (a5_b * 16 + a5_u)] = acc5__c3_0;
        }
        if (a0_b * 64 + a0_u * 8 + 4 < 16) {
            matmul_1[(a0_b * 64 + a0_u * 8 + 4) % 16 * 64 + (a5_b * 16 + a5_u)] = acc5__c4_0;
        }
        if (a0_b * 64 + a0_u * 8 + 5 < 16) {
            matmul_1[(a0_b * 64 + a0_u * 8 + 5) % 16 * 64 + (a5_b * 16 + a5_u)] = acc5__c5_0;
        }
        if (a0_b * 64 + a0_u * 8 + 6 < 16) {
            matmul_1[(a0_b * 64 + a0_u * 8 + 6) % 16 * 64 + (a5_b * 16 + a5_u)] = acc5__c6_0;
        }
        if (a0_b * 64 + a0_u * 8 + 7 < 16) {
            matmul_1[(a0_b * 64 + a0_u * 8 + 7) % 16 * 64 + (a5_b * 16 + a5_u)] = acc5__c7_0;
        }
}
