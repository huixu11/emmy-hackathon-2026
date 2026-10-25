#include <cuda_bf16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const __nv_bfloat16* mul_c1, const __nv_bfloat16* x, __nv_bfloat16* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = mul_c1[0];
        float in1__u0 = x[a0 * 4];
        float v0__u0 = in0__u0 * in1__u0;
        float in0__u1 = mul_c1[0];
        float in1__u1 = x[a0 * 4 + 1];
        float v0__u1 = in0__u1 * in1__u1;
        float in0__u2 = mul_c1[0];
        float in1__u2 = x[a0 * 4 + 2];
        float v0__u2 = in0__u2 * in1__u2;
        float in0__u3 = mul_c1[0];
        float in1__u3 = x[a0 * 4 + 3];
        float v0__u3 = in0__u3 * in1__u3;
        mul[a0 * 4] = v0__u0;
        mul[a0 * 4 + 1] = v0__u1;
        mul[a0 * 4 + 2] = v0__u2;
        mul[a0 * 4 + 3] = v0__u3;
}
