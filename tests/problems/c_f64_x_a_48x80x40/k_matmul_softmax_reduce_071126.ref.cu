extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_071126(const double* mul_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 168) {
        int _rsub = _gid;
        int _rgrp = _rsub / 112;
        int _rsz = min(8, 12 - _rgrp * 8);
        int a0_b = _rgrp * 8 + (_rsub % 112) % _rsz;
        int a5_b = (_rsub % 112) / _rsz;
        int a0_u = (_gid) % 1;
        int a5_u = (_gid) % 1;
        float in0__ar0 = mul_c1[0];
        float acc1__ar0 = -1e+30f;
        for (int a1 = 0; a1 < 80; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__ar0 = q[(a0_b * 4 + a0_u * 4) * 32 + a2];
                float in1__ar0 = kt[a2 * 80 + a1];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0;
            float v2__ar0 = in0__ar0 * v1__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v2__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a1 = 0; a1 < 80; a1++) {
            float acc2__ar0 = 0.0f;
            for (int a4 = 0; a4 < 32; a4++) {
                float in4__ar0 = q[(a0_b * 4 + a0_u * 4) * 32 + a4];
                float in3__ar0 = kt[a4 * 80 + a1];
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
        for (int a1 = 0; a1 < 80; a1++) {
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__ar1 = q[(a0_b * 4 + a0_u * 4 + 1) * 32 + a2];
                float in1__ar1 = kt[a2 * 80 + a1];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__ar1 = acc0__ar1;
            float v2__ar1 = in0__ar1 * v1__ar1;
            acc1__ar1 = fmaxf(acc1__ar1, v2__ar1);
        }
        float acc3__ar1 = 0.0f;
        for (int a1 = 0; a1 < 80; a1++) {
            float acc2__ar1 = 0.0f;
            for (int a4 = 0; a4 < 32; a4++) {
                float in4__ar1 = q[(a0_b * 4 + a0_u * 4 + 1) * 32 + a4];
                float in3__ar1 = kt[a4 * 80 + a1];
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
        for (int a1 = 0; a1 < 80; a1++) {
            float acc0__ar2 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__ar2 = q[(a0_b * 4 + a0_u * 4 + 2) * 32 + a2];
                float in1__ar2 = kt[a2 * 80 + a1];
                float v0__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v0__ar2;
            }
            float v1__ar2 = acc0__ar2;
            float v2__ar2 = in0__ar2 * v1__ar2;
            acc1__ar2 = fmaxf(acc1__ar2, v2__ar2);
        }
        float acc3__ar2 = 0.0f;
        for (int a1 = 0; a1 < 80; a1++) {
            float acc2__ar2 = 0.0f;
            for (int a4 = 0; a4 < 32; a4++) {
                float in4__ar2 = q[(a0_b * 4 + a0_u * 4 + 2) * 32 + a4];
                float in3__ar2 = kt[a4 * 80 + a1];
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
        for (int a1 = 0; a1 < 80; a1++) {
            float acc0__ar3 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__ar3 = q[(a0_b * 4 + a0_u * 4 + 3) * 32 + a2];
                float in1__ar3 = kt[a2 * 80 + a1];
                float v0__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v0__ar3;
            }
            float v1__ar3 = acc0__ar3;
            float v2__ar3 = in0__ar3 * v1__ar3;
            acc1__ar3 = fmaxf(acc1__ar3, v2__ar3);
        }
        float acc3__ar3 = 0.0f;
        for (int a1 = 0; a1 < 80; a1++) {
            float acc2__ar3 = 0.0f;
            for (int a4 = 0; a4 < 32; a4++) {
                float in4__ar3 = q[(a0_b * 4 + a0_u * 4 + 3) * 32 + a4];
                float in3__ar3 = kt[a4 * 80 + a1];
                float v3__ar3 = in3__ar3 * in4__ar3;
                acc2__ar3 += v3__ar3;
            }
            float v4__ar3 = acc2__ar3;
            float v7__ar3 = expf(((in0__ar3 * v4__ar3) - acc1__ar3));
            acc3__ar3 += v7__ar3;
        }
        float v8__ar3 = 1.0f / acc3__ar3;
        float acc5__c0_0 = 0.0f;
        float acc5__c0_1 = 0.0f;
        float acc5__c0_2 = 0.0f;
        float acc5__c1_0 = 0.0f;
        float acc5__c1_1 = 0.0f;
        float acc5__c1_2 = 0.0f;
        float acc5__c2_0 = 0.0f;
        float acc5__c2_1 = 0.0f;
        float acc5__c2_2 = 0.0f;
        float acc5__c3_0 = 0.0f;
        float acc5__c3_1 = 0.0f;
        float acc5__c3_2 = 0.0f;
        for (int a6 = 0; a6 < 80; a6++) {
            float acc4__ar0 = 0.0f;
            for (int a7 = 0; a7 < 32; a7++) {
                float in6__ar0 = q[(a0_b * 4 + a0_u * 4) * 32 + a7];
                float in5__ar0 = kt[a7 * 80 + a6];
                float v9__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v9__ar0;
            }
            float v10__ar0 = acc4__ar0;
            float v14__ar0 = expf(((in0__ar0 * v10__ar0) - acc1__ar0)) * v8__ar0;
            float acc4__ar1 = 0.0f;
            for (int a7 = 0; a7 < 32; a7++) {
                float in6__ar1 = q[(a0_b * 4 + a0_u * 4 + 1) * 32 + a7];
                float in5__ar1 = kt[a7 * 80 + a6];
                float v9__ar1 = in5__ar1 * in6__ar1;
                acc4__ar1 += v9__ar1;
            }
            float v10__ar1 = acc4__ar1;
            float v14__ar1 = expf(((in0__ar1 * v10__ar1) - acc1__ar1)) * v8__ar1;
            float acc4__ar2 = 0.0f;
            for (int a7 = 0; a7 < 32; a7++) {
                float in6__ar2 = q[(a0_b * 4 + a0_u * 4 + 2) * 32 + a7];
                float in5__ar2 = kt[a7 * 80 + a6];
                float v9__ar2 = in5__ar2 * in6__ar2;
                acc4__ar2 += v9__ar2;
            }
            float v10__ar2 = acc4__ar2;
            float v14__ar2 = expf(((in0__ar2 * v10__ar2) - acc1__ar2)) * v8__ar2;
            float acc4__ar3 = 0.0f;
            for (int a7 = 0; a7 < 32; a7++) {
                float in6__ar3 = q[(a0_b * 4 + a0_u * 4 + 3) * 32 + a7];
                float in5__ar3 = kt[a7 * 80 + a6];
                float v9__ar3 = in5__ar3 * in6__ar3;
                acc4__ar3 += v9__ar3;
            }
            float v10__ar3 = acc4__ar3;
            float v14__ar3 = expf(((in0__ar3 * v10__ar3) - acc1__ar3)) * v8__ar3;
            float in7__bc0 = v[a6 * 40 + (a5_b * 3 + a5_u * 3) % 40];
            float v15__c0_0 = in7__bc0 * v14__ar0;
            acc5__c0_0 += v15__c0_0;
            float in7__bc1 = v[a6 * 40 + (a5_b * 3 + a5_u * 3 + 1) % 40];
            float v15__c0_1 = in7__bc1 * v14__ar0;
            acc5__c0_1 += v15__c0_1;
            float in7__bc2 = v[a6 * 40 + (a5_b * 3 + a5_u * 3 + 2) % 40];
            float v15__c0_2 = in7__bc2 * v14__ar0;
            acc5__c0_2 += v15__c0_2;
            float v15__c1_0 = in7__bc0 * v14__ar1;
            acc5__c1_0 += v15__c1_0;
            float v15__c1_1 = in7__bc1 * v14__ar1;
            acc5__c1_1 += v15__c1_1;
            float v15__c1_2 = in7__bc2 * v14__ar1;
            acc5__c1_2 += v15__c1_2;
            float v15__c2_0 = in7__bc0 * v14__ar2;
            acc5__c2_0 += v15__c2_0;
            float v15__c2_1 = in7__bc1 * v14__ar2;
            acc5__c2_1 += v15__c2_1;
            float v15__c2_2 = in7__bc2 * v14__ar2;
            acc5__c2_2 += v15__c2_2;
            float v15__c3_0 = in7__bc0 * v14__ar3;
            acc5__c3_0 += v15__c3_0;
            float v15__c3_1 = in7__bc1 * v14__ar3;
            acc5__c3_1 += v15__c3_1;
            float v15__c3_2 = in7__bc2 * v14__ar3;
            acc5__c3_2 += v15__c3_2;
        }
        if (a5_b * 3 + a5_u * 3 + 0 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4) * 40 + (a5_b * 3 + a5_u * 3) % 40] = acc5__c0_0;
        }
        if (a5_b * 3 + a5_u * 3 + 1 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4) * 40 + (a5_b * 3 + a5_u * 3 + 1) % 40] = acc5__c0_1;
        }
        if (a5_b * 3 + a5_u * 3 + 2 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4) * 40 + (a5_b * 3 + a5_u * 3 + 2) % 40] = acc5__c0_2;
        }
        if (a5_b * 3 + a5_u * 3 + 0 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) * 40 + (a5_b * 3 + a5_u * 3) % 40] = acc5__c1_0;
        }
        if (a5_b * 3 + a5_u * 3 + 1 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) * 40 + (a5_b * 3 + a5_u * 3 + 1) % 40] = acc5__c1_1;
        }
        if (a5_b * 3 + a5_u * 3 + 2 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) * 40 + (a5_b * 3 + a5_u * 3 + 2) % 40] = acc5__c1_2;
        }
        if (a5_b * 3 + a5_u * 3 + 0 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) * 40 + (a5_b * 3 + a5_u * 3) % 40] = acc5__c2_0;
        }
        if (a5_b * 3 + a5_u * 3 + 1 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) * 40 + (a5_b * 3 + a5_u * 3 + 1) % 40] = acc5__c2_1;
        }
        if (a5_b * 3 + a5_u * 3 + 2 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) * 40 + (a5_b * 3 + a5_u * 3 + 2) % 40] = acc5__c2_2;
        }
        if (a5_b * 3 + a5_u * 3 + 0 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) * 40 + (a5_b * 3 + a5_u * 3) % 40] = acc5__c3_0;
        }
        if (a5_b * 3 + a5_u * 3 + 1 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) * 40 + (a5_b * 3 + a5_u * 3 + 1) % 40] = acc5__c3_1;
        }
        if (a5_b * 3 + a5_u * 3 + 2 < 40) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) * 40 + (a5_b * 3 + a5_u * 3 + 2) % 40] = acc5__c3_2;
        }
    }
}
