extern "C" __global__
__launch_bounds__(32) void k_sdpa_4dac4b(const double* scaled_dot_product_attention_scale, const double* q, const double* k, const double* v, double* scaled_dot_product_attention) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 98304;
        int a1 = (_gid / 2048) % 48;
        int a6 = (_gid / 32) % 64;
        int a2_co = (_gid) % 32;
        float in0 = scaled_dot_product_attention_scale[0];
        float acc1 = -1e+30f;
        float acc1__r1 = -1e+30f;
        for (int a2 = 0; a2 < 48; a2 += 2) {
            float acc0 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2 = q[a0 * 3072 + a1 * 64 + a3];
                float in1 = k[a0 * 3072 + a2 * 64 + a3];
                float v0 = in1 * in2;
                acc0 += v0;
            }
            float v1 = acc0 * in0;
            acc1 = fmaxf(acc1, v1);
            float acc0__r1 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                float in2__r1 = q[a0 * 3072 + a1 * 64 + a3];
                float in1__r1 = k[a0 * 3072 + (a2 + 1) * 64 + a3];
                float v0__r1 = in1__r1 * in2__r1;
                acc0__r1 += v0__r1;
            }
            float v1__r1 = acc0__r1 * in0;
            acc1__r1 = fmaxf(acc1__r1, v1__r1);
        }
        acc1 = fmaxf(acc1, acc1__r1);
        float acc3 = 0.0f;
        for (int a2 = a2_co; a2 < 48; a2 += 32) {
            float acc2 = 0.0f;
            for (int a5 = 0; a5 < 64; a5++) {
                float in4 = q[a0 * 3072 + a1 * 64 + a5];
                float in3 = k[a0 * 3072 + a2 * 64 + a5];
                float v2 = in3 * in4;
                acc2 += v2;
            }
            float v3 = acc2 * in0;
            float v4 = v3 - acc1;
            float v5 = expf(v4);
            acc3 += v5;
        }
        {
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 16);
            acc3 = acc3 + acc3__o;
        }
        {
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 8);
            acc3 = acc3 + acc3__o;
        }
        {
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 4);
            acc3 = acc3 + acc3__o;
        }
        {
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 2);
            acc3 = acc3 + acc3__o;
        }
        {
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 1);
            acc3 = acc3 + acc3__o;
        }
        float v6 = 1.0f / acc3;
        float acc5 = 0.0f;
        for (int a7 = a2_co; a7 < 48; a7 += 32) {
            float acc4 = 0.0f;
            for (int a8 = 0; a8 < 64; a8++) {
                float in6 = q[a0 * 3072 + a1 * 64 + a8];
                float in5 = k[a0 * 3072 + a7 * 64 + a8];
                float v7 = in5 * in6;
                acc4 += v7;
            }
            float v8 = acc4 * in0;
            float v9 = v8 - acc1;
            float v11 = expf(v9) * v6;
            float in7 = v[a0 * 3072 + a7 * 64 + a6];
            float v12 = in7 * v11;
            acc5 += v12;
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
        if (a2_co == 0) {
            scaled_dot_product_attention[a0 * 3072 + a1 * 64 + a6] = acc5;
        }
}
