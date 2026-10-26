extern "C" __global__
__launch_bounds__(128) void k_softmax_5c6897(const float* masked_fill_c2, const float* lt_c1, const float* x, float* softmax) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float in0 = lt_c1[0];
        float in1 = masked_fill_c2[0];
        float acc1 = 0.0f;
        float acc0 = -1e+30f;
        for (int a1 = a1_co; a1 < 128; a1 += 128) {
            float in2 = x[a0 * 128 + a1];
            bool v0 = in2 < in0;
            float v1 = ((v0) ? (in1) : (in2));
            float one = 1.0f;
            float v1__gn = fmaxf(acc0, v1);
            float v1__acc1_sa = acc1 * expf((acc0 - v1__gn));
            float v1__acc1_sb = one * expf((v1 - v1__gn));
            acc1 = v1__acc1_sa + v1__acc1_sb;
            acc0 = fmaxf(acc0, v1);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            float acc0__o__gn = fmaxf(acc0, acc0__o);
            float acc0__o__dg = acc0 - acc0__o__gn;
            float acc0__o__alpha = expf(acc0__o__dg);
            float acc0__o__dg_o = acc0__o - acc0__o__gn;
            float acc0__o__beta = expf(acc0__o__dg_o);
            float acc0__o__acc1_sa = acc0__o__alpha * acc1;
            float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
            acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
            acc0 = acc0__o__gn;
        }
        __shared__ float acc0_smem[4];
        __shared__ float acc1_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
            acc1_smem[warp] = acc1;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                float acc0__o__gn = fmaxf(acc0, acc0__o);
                float acc0__o__dg = acc0 - acc0__o__gn;
                float acc0__o__alpha = expf(acc0__o__dg);
                float acc0__o__dg_o = acc0__o - acc0__o__gn;
                float acc0__o__beta = expf(acc0__o__dg_o);
                float acc0__o__acc1_sa = acc0__o__alpha * acc1;
                float acc0__o__acc1_sb = acc0__o__beta * acc1__o;
                acc1 = acc0__o__acc1_sa + acc0__o__acc1_sb;
                acc0 = acc0__o__gn;
                acc0_smem[warp] = acc0;
                acc1_smem[warp] = acc1;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        acc1 = acc1_smem[0];
        float v6 = 1.0f / acc1;
        for (int a3 = a1_co; a3 < 128; a3 += 128) {
            float in4 = x[a0 * 128 + a3];
            bool v7 = in4 < in0;
            float v8 = ((v7) ? (in1) : (in4));
            float v11 = expf((v8 - acc0)) * v6;
            softmax[a0 * 128 + a3] = v11;
        }
}
