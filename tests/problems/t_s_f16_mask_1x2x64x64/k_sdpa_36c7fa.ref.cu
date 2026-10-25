#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(512) void k_sdpa_36c7fa(const __half* scaled_dot_product_attention_scale, const __half* q, const __half* k, const __half* m, const __half* v, __half* scaled_dot_product_attention) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 2097152;
        int a1 = (_gid / 32768) % 64;
        int a6 = (_gid / 512) % 64;
        int a2_co = (_gid) % 512;
        __half in0 = scaled_dot_product_attention_scale[0];
        float acc3 = 0.0f;
        float acc5__sum = 0.0f;
        float acc1 = -1e+30f;
        float acc3__r1 = 0.0f;
        float acc5__sum__r1 = 0.0f;
        float acc1__r1 = -1e+30f;
        float acc3__r2 = 0.0f;
        float acc5__sum__r2 = 0.0f;
        float acc1__r2 = -1e+30f;
        float acc3__r3 = 0.0f;
        float acc5__sum__r3 = 0.0f;
        float acc1__r3 = -1e+30f;
        for (int a2 = a2_co; a2 < 64; a2 += 2048) {
            float acc0 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                __half in2 = q[a0 * 4096 + a1 * 64 + a3];
                __half in1 = k[a0 * 4096 + a2 * 64 + a3];
                __half v0 = in1 * in2;
                acc0 += __half2float(v0);
            }
            float v1 = acc0 * __half2float(in0);
            __half in3 = m[a1 * 64 + a2];
            float v2 = __half2float(in3) + v1;
            float one = 1.0f;
            float v2__gn = fmaxf(acc1, v2);
            float v2__alpha = expf((acc1 - v2__gn));
            float v2__beta = expf((v2 - v2__gn));
            float v2__acc3_sa = acc3 * v2__alpha;
            float v2__acc3_sb = one * v2__beta;
            acc3 = v2__acc3_sa + v2__acc3_sb;
            float v2__acc5__sum_sa = acc5__sum * v2__alpha;
            __half in9 = v[a0 * 4096 + a2 * 64 + a6];
            float v2__acc5__sum_sb = __half2float(in9) * v2__beta;
            acc5__sum = v2__acc5__sum_sa + v2__acc5__sum_sb;
            acc1 = fmaxf(acc1, v2);
            float acc0__r1 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                __half in2__r1 = q[a0 * 4096 + a1 * 64 + a3];
                __half in1__r1 = k[a0 * 4096 + (a2 + 512) % 64 * 64 + a3];
                __half v0__r1 = in1__r1 * in2__r1;
                acc0__r1 += __half2float(v0__r1);
            }
            float v1__r1 = acc0__r1 * __half2float(in0);
            __half in3__r1 = m[a1 * 64 + (a2 + 512) % 64];
            float v2__r1 = __half2float(in3__r1) + v1__r1;
            float v2__r1__id = -1e+30f;
            float v2__r1__mv = ((a2 + 512 < 64) ? (((float)(v2__r1))) : (((float)(v2__r1__id))));
            float one__r1 = 1.0f;
            float v2__gn__r1 = fmaxf(acc1__r1, v2__r1__mv);
            float v2__alpha__r1 = expf((acc1__r1 - v2__gn__r1));
            float v2__beta__r1 = expf((v2__r1__mv - v2__gn__r1));
            float v2__acc3_sa__r1 = acc3__r1 * v2__alpha__r1;
            float v2__acc3_sb__r1 = one__r1 * v2__beta__r1;
            acc3__r1 = v2__acc3_sa__r1 + v2__acc3_sb__r1;
            float v2__acc5__sum_sa__r1 = acc5__sum__r1 * v2__alpha__r1;
            __half in9__r1 = v[a0 * 4096 + (a2 + 512) % 64 * 64 + a6];
            float v2__acc5__sum_sb__r1 = __half2float(in9__r1) * v2__beta__r1;
            acc5__sum__r1 = v2__acc5__sum_sa__r1 + v2__acc5__sum_sb__r1;
            acc1__r1 = fmaxf(acc1__r1, v2__r1__mv);
            float acc0__r2 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                __half in2__r2 = q[a0 * 4096 + a1 * 64 + a3];
                __half in1__r2 = k[a0 * 4096 + (a2 + 1024) % 64 * 64 + a3];
                __half v0__r2 = in1__r2 * in2__r2;
                acc0__r2 += __half2float(v0__r2);
            }
            float v1__r2 = acc0__r2 * __half2float(in0);
            __half in3__r2 = m[a1 * 64 + (a2 + 1024) % 64];
            float v2__r2 = __half2float(in3__r2) + v1__r2;
            float v2__r2__id = -1e+30f;
            float v2__r2__mv = ((a2 + 1024 < 64) ? (((float)(v2__r2))) : (((float)(v2__r2__id))));
            float one__r2 = 1.0f;
            float v2__gn__r2 = fmaxf(acc1__r2, v2__r2__mv);
            float v2__alpha__r2 = expf((acc1__r2 - v2__gn__r2));
            float v2__beta__r2 = expf((v2__r2__mv - v2__gn__r2));
            float v2__acc3_sa__r2 = acc3__r2 * v2__alpha__r2;
            float v2__acc3_sb__r2 = one__r2 * v2__beta__r2;
            acc3__r2 = v2__acc3_sa__r2 + v2__acc3_sb__r2;
            float v2__acc5__sum_sa__r2 = acc5__sum__r2 * v2__alpha__r2;
            __half in9__r2 = v[a0 * 4096 + (a2 + 1024) % 64 * 64 + a6];
            float v2__acc5__sum_sb__r2 = __half2float(in9__r2) * v2__beta__r2;
            acc5__sum__r2 = v2__acc5__sum_sa__r2 + v2__acc5__sum_sb__r2;
            acc1__r2 = fmaxf(acc1__r2, v2__r2__mv);
            float acc0__r3 = 0.0f;
            for (int a3 = 0; a3 < 64; a3++) {
                __half in2__r3 = q[a0 * 4096 + a1 * 64 + a3];
                __half in1__r3 = k[a0 * 4096 + (a2 + 1536) % 64 * 64 + a3];
                __half v0__r3 = in1__r3 * in2__r3;
                acc0__r3 += __half2float(v0__r3);
            }
            float v1__r3 = acc0__r3 * __half2float(in0);
            __half in3__r3 = m[a1 * 64 + (a2 + 1536) % 64];
            float v2__r3 = __half2float(in3__r3) + v1__r3;
            float v2__r3__id = -1e+30f;
            float v2__r3__mv = ((a2 + 1536 < 64) ? (((float)(v2__r3))) : (((float)(v2__r3__id))));
            float one__r3 = 1.0f;
            float v2__gn__r3 = fmaxf(acc1__r3, v2__r3__mv);
            float v2__alpha__r3 = expf((acc1__r3 - v2__gn__r3));
            float v2__beta__r3 = expf((v2__r3__mv - v2__gn__r3));
            float v2__acc3_sa__r3 = acc3__r3 * v2__alpha__r3;
            float v2__acc3_sb__r3 = one__r3 * v2__beta__r3;
            acc3__r3 = v2__acc3_sa__r3 + v2__acc3_sb__r3;
            float v2__acc5__sum_sa__r3 = acc5__sum__r3 * v2__alpha__r3;
            __half in9__r3 = v[a0 * 4096 + (a2 + 1536) % 64 * 64 + a6];
            float v2__acc5__sum_sb__r3 = __half2float(in9__r3) * v2__beta__r3;
            acc5__sum__r3 = v2__acc5__sum_sa__r3 + v2__acc5__sum_sb__r3;
            acc1__r3 = fmaxf(acc1__r3, v2__r3__mv);
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
        float acc1__r2__gn = fmaxf(acc1, acc1__r2);
        float acc1__r2__alpha = expf((acc1 - acc1__r2__gn));
        float acc1__r2__beta = expf((acc1__r2 - acc1__r2__gn));
        float acc1__r2__acc3_sa = acc1__r2__alpha * acc3;
        float acc1__r2__acc3_sb = acc1__r2__beta * acc3__r2;
        acc3 = acc1__r2__acc3_sa + acc1__r2__acc3_sb;
        float acc1__r2__acc5__sum_sa = acc1__r2__alpha * acc5__sum;
        float acc1__r2__acc5__sum_sb = acc1__r2__beta * acc5__sum__r2;
        acc5__sum = acc1__r2__acc5__sum_sa + acc1__r2__acc5__sum_sb;
        acc1 = fmaxf(acc1, acc1__r2);
        float acc1__r3__gn = fmaxf(acc1, acc1__r3);
        float acc1__r3__alpha = expf((acc1 - acc1__r3__gn));
        float acc1__r3__beta = expf((acc1__r3 - acc1__r3__gn));
        float acc1__r3__acc3_sa = acc1__r3__alpha * acc3;
        float acc1__r3__acc3_sb = acc1__r3__beta * acc3__r3;
        acc3 = acc1__r3__acc3_sa + acc1__r3__acc3_sb;
        float acc1__r3__acc5__sum_sa = acc1__r3__alpha * acc5__sum;
        float acc1__r3__acc5__sum_sb = acc1__r3__beta * acc5__sum__r3;
        acc5__sum = acc1__r3__acc5__sum_sa + acc1__r3__acc5__sum_sb;
        acc1 = fmaxf(acc1, acc1__r3);
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
        __shared__ float acc1_smem[16];
        __shared__ float acc3_smem[16];
        __shared__ float acc5__sum_smem[16];
        if (lane == 0) {
            acc1_smem[warp] = acc1;
            acc3_smem[warp] = acc3;
            acc5__sum_smem[warp] = acc5__sum;
        }
        __syncthreads();
        for (int s = 8; s > 0; s >>= 1) {
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
        if (a2_co == 0) {
            float acc5 = acc5__sum * (1.0f / acc3);
            scaled_dot_product_attention[a0 * 4096 + a1 * 64 + a6] = __float2half(acc5);
        }
}
