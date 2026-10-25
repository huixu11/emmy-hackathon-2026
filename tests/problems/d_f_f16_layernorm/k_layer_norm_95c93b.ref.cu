#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(128) void k_layer_norm_95c93b(const __half* layer_norm_var_count, const __half* layer_norm_mean_count, const __half* layer_norm_eps, const __half* x, __half* layer_norm) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 256; a1 += 128) {
            __half in3 = x[a0 * 256 + a1];
            acc0 += __half2float(in3);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }
        __shared__ float acc0_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        __half in1 = layer_norm_mean_count[0];
        float v2 = acc0 * __half2float((1.0f / in1));
        float acc1 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            __half in4 = x[a0 * 256 + a1];
            float v3 = __half2float(in4) - v2;
            float v4 = v3 * v3;
            acc1 += v4;
        }
        __half in2 = layer_norm_var_count[0];
        float v5 = acc1 * __half2float((1.0f / in2));
        __half in0 = layer_norm_eps[0];
        float v6 = __half2float(in0) + v5;
        float v7 = rsqrtf(v6);
        for (int a3 = a1_co; a3 < 256; a3 += 128) {
            __half in5 = x[a0 * 256 + a3];
            float v8 = __half2float(in5) - v2;
            float v9 = v7 * v8;
            layer_norm[a0 * 256 + a3] = __float2half(v9);
        }
}
