extern "C" __global__
__launch_bounds__(64) void k_matmul_softmax_reduce_9b9b28(const float* mul_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 4096;
        int a5 = (_gid / 64) % 64;
        int a1_co = (_gid) % 64;
        float in0 = mul_c1[0];
        float acc3 = 0.0f;
        float acc5__sum = 0.0f;
        float acc1 = -1e+30f;
        float acc3__r1 = 0.0f;
        float acc5__sum__r1 = 0.0f;
        float acc1__r1 = -1e+30f;
        for (int a1 = a1_co; a1 < 256; a1 += 128) {
            float acc0 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2 = q[a0 * 128 + a2];
                float in1 = kt[a2 * 256 + a1];
                float v0 = in1 * in2;
                acc0 += v0;
            }
            float v1 = acc0 * in0;
            float one = 1.0f;
            float v1__gn = fmaxf(acc1, v1);
            float v1__alpha = expf((acc1 - v1__gn));
            float v1__beta = expf((v1 - v1__gn));
            float v1__acc3_sa = acc3 * v1__alpha;
            float v1__acc3_sb = one * v1__beta;
            acc3 = v1__acc3_sa + v1__acc3_sb;
            float v1__acc5__sum_sa = acc5__sum * v1__alpha;
            float in7 = v[a1 * 64 + a5];
            float v1__acc5__sum_sb = in7 * v1__beta;
            acc5__sum = v1__acc5__sum_sa + v1__acc5__sum_sb;
            acc1 = fmaxf(acc1, v1);
            float acc0__r1 = 0.0f;
            for (int a2 = 0; a2 < 128; a2++) {
                float in2__r1 = q[a0 * 128 + a2];
                float in1__r1 = kt[a2 * 256 + (a1 + 64)];
                float v0__r1 = in1__r1 * in2__r1;
                acc0__r1 += v0__r1;
            }
            float v1__r1 = acc0__r1 * in0;
            float one__r1 = 1.0f;
            float v1__gn__r1 = fmaxf(acc1__r1, v1__r1);
            float v1__alpha__r1 = expf((acc1__r1 - v1__gn__r1));
            float v1__beta__r1 = expf((v1__r1 - v1__gn__r1));
            float v1__acc3_sa__r1 = acc3__r1 * v1__alpha__r1;
            float v1__acc3_sb__r1 = one__r1 * v1__beta__r1;
            acc3__r1 = v1__acc3_sa__r1 + v1__acc3_sb__r1;
            float v1__acc5__sum_sa__r1 = acc5__sum__r1 * v1__alpha__r1;
            float in7__r1 = v[(a1 + 64) * 64 + a5];
            float v1__acc5__sum_sb__r1 = in7__r1 * v1__beta__r1;
            acc5__sum__r1 = v1__acc5__sum_sa__r1 + v1__acc5__sum_sb__r1;
            acc1__r1 = fmaxf(acc1__r1, v1__r1);
        }
        float acc1__r1__gn = fmaxf(acc1, acc1__r1);
        float acc1__r1__alpha = expf((acc1 - acc1__r1__gn));
        float acc1__r1__beta = expf((acc1__r1 - acc1__r1__gn));
        float acc1__r1__acc3_sa = acc1__r1__alpha * acc3;
        float acc1__r1__acc3_sb = acc1__r1__beta * acc3__r1;
        acc3 = acc1__r1__acc3_sa + acc1__r1__acc3_sb;
        float acc1__r1__acc5__sum_sa = acc1__r1__alpha * acc5__sum;
        float acc1__r1__acc5__sum_sb = acc1__r1__beta * acc5__sum__r1;
        acc5__sum = acc1__r1__acc5__sum_sa + acc1__r1__acc5__sum_sb;
        acc1 = fmaxf(acc1, acc1__r1);
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 16);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 16);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 8);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 8);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 4);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 4);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 2);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 2);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 1);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 1);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        __shared__ float acc1_smem[2];
        __shared__ float acc3_smem[2];
        __shared__ float acc5__sum_smem[2];
        if (lane == 0) {
            acc1_smem[warp] = acc1;
            acc3_smem[warp] = acc3;
            acc5__sum_smem[warp] = acc5__sum;
        }
        __syncthreads();
        for (int s = 1; s > 0; s >>= 1) {
            if (warp < s) {
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                float acc3 = acc3_smem[warp];
                float acc3__o = acc3_smem[warp + s];
                float acc5__sum = acc5__sum_smem[warp];
                float acc5__sum__o = acc5__sum_smem[warp + s];
                float acc1__o__gn = fmaxf(acc1, acc1__o);
                float acc1__o__dg = acc1 - acc1__o__gn;
                float acc1__o__alpha = expf(acc1__o__dg);
                float acc1__o__dg_o = acc1__o - acc1__o__gn;
                float acc1__o__beta = expf(acc1__o__dg_o);
                float acc1__o__acc3_sa = acc1__o__alpha * acc3;
                float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
                acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
                float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
                float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
                acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
                acc1 = acc1__o__gn;
                acc1_smem[warp] = acc1;
                acc3_smem[warp] = acc3;
                acc5__sum_smem[warp] = acc5__sum;
            }
            __syncthreads();
        }
        acc1 = acc1_smem[0];
        acc3 = acc3_smem[0];
        acc5__sum = acc5__sum_smem[0];
        if (a1_co == 0) {
            float acc5 = acc5__sum * (1.0f / acc3);
            matmul_1[a0 * 64 + a5] = acc5;
        }
}
