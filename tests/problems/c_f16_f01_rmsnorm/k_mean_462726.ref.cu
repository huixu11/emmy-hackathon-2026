#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(128) void k_mean_462726(const __half* mean_count, const __half* add_c1, const __half* x, const __half* w, __half* mul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 1024; a1 += 128) {
            __half in2 = x[a0 * 1024 + a1];
            float v1 = __half2float(in2) * __half2float(in2);
            acc0 += v1;
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
        __half in1 = mean_count[0];
        float v2 = acc0 * __half2float((1.0f / in1));
        __half in0 = add_c1[0];
        float v3 = __half2float(in0) + v2;
        float v5 = 1.0f / sqrtf(v3);
        for (int a2 = a1_co; a2 < 1024; a2 += 128) {
            __half in3 = x[a0 * 1024 + a2];
            float v6 = __half2float(in3) * v5;
            __half in4 = w[a2];
            float v7 = __half2float(in4) * v6;
            mul_1[a0 * 1024 + a2] = __float2half(v7);
        }
}
