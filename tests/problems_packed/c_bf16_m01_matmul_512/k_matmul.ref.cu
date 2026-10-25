#include <cuda_bf16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* matmul__partial, __nv_bfloat16* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 512;
        int a1 = (_gid) % 512;
        float acc0 = 0.0f;
        for (int _a2_ks = 0; _a2_ks < 8; _a2_ks++) {
            float acc0__p = matmul__partial[_a2_ks * 262144LL + a0 * 512 + a1];
            acc0 += acc0__p;
        }
        matmul[a0 * 512 + a1] = acc0;
}
