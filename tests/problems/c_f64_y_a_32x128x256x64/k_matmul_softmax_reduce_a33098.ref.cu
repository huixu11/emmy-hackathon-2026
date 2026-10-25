extern "C" __global__
__launch_bounds__(128) void k_matmul_softmax_reduce_a33098(const double* mul_c1, const double* q, const double* kt, const double* v, double* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 8192;
        int a5 = (_gid / 128) % 64;
        int a6_co = (_gid) % 128;
        float in0 = mul_c1[0];
        float acc1 = -1e+30f;
        float acc1__r1 = -1e+30f;
        float acc1__r2 = -1e+30f;
        float acc1__r3 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1 += 4) {
            float acc0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2 = q[a0 * 128 + a2];
                float in1 = kt[a2 * 256 + a1];
                float v0 = in1 * in2;
                acc0 += v0;
            }
            float v1 = acc0;
            float v2 = in0 * v1;
            acc1 = fmaxf(acc1, v2);
            float acc0__r1 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__r1 = q[a0 * 128 + a2];
                float in1__r1 = kt[a2 * 256 + (a1 + 1)];
                float v0__r1 = in1__r1 * in2__r1;
                acc0__r1 += v0__r1;
            }
            float v1__r1 = acc0__r1;
            float v2__r1 = in0 * v1__r1;
            acc1__r1 = fmaxf(acc1__r1, v2__r1);
            float acc0__r2 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__r2 = q[a0 * 128 + a2];
                float in1__r2 = kt[a2 * 256 + (a1 + 2)];
                float v0__r2 = in1__r2 * in2__r2;
                acc0__r2 += v0__r2;
            }
            float v1__r2 = acc0__r2;
            float v2__r2 = in0 * v1__r2;
            acc1__r2 = fmaxf(acc1__r2, v2__r2);
            float acc0__r3 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__r3 = q[a0 * 128 + a2];
                float in1__r3 = kt[a2 * 256 + (a1 + 3)];
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
        for (int a1 = 0; a1 < 256; a1 += 2) {
            float acc2 = 0.0f;
            for (int a4 = 0; a4 < 128; a4++) {
                float in4 = q[a0 * 128 + a4];
                float in3 = kt[a4 * 256 + a1];
                float v3 = in3 * in4;
                acc2 += v3;
            }
            float v4 = acc2;
            float v6 = (in0 * v4) - acc1;
            float v7 = expf(v6);
            acc3 += v7;
            float acc2__r1 = 0.0f;
            for (int a4 = 0; a4 < 128; a4++) {
                float in4__r1 = q[a0 * 128 + a4];
                float in3__r1 = kt[a4 * 256 + (a1 + 1)];
                float v3__r1 = in3__r1 * in4__r1;
                acc2__r1 += v3__r1;
            }
            float v4__r1 = acc2__r1;
            float v6__r1 = (in0 * v4__r1) - acc1;
            float v7__r1 = expf(v6__r1);
            acc3__r1 += v7__r1;
        }
        acc3 += acc3__r1;
        float v8 = 1.0f / acc3;
        float acc5 = 0.0f;
        for (int a6 = a6_co; a6 < 256; a6 += 128) {
            float acc4 = 0.0f;
            for (int a7 = 0; a7 < 128; a7++) {
                float in6 = q[a0 * 128 + a7];
                float in5 = kt[a7 * 256 + a6];
                float v9 = in5 * in6;
                acc4 += v9;
            }
            float v10 = acc4;
            float v12 = (in0 * v10) - acc1;
            float v14 = expf(v12) * v8;
            float in7 = v[a6 * 64 + a5];
            float v15 = in7 * v14;
            acc5 += v15;
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
        __shared__ float acc5_smem[4];
        if (lane == 0) {
            acc5_smem[warp] = acc5;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc5 = acc5_smem[warp];
                float acc5__o = acc5_smem[warp + s];
                acc5 = acc5 + acc5__o;
                acc5_smem[warp] = acc5;
            }
            __syncthreads();
        }
        acc5 = acc5_smem[0];
        if (a6_co == 0) {
            matmul_1[a0 * 64 + a5] = acc5;
        }
}
