#include <cuda_bf16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const __nv_bfloat16* x, const __nv_bfloat16* y, __nv_bfloat16* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x[a0 * 4];
        float in1__u0 = y[a0 * 4];
        float v1__u0 = in1__u0 * fmaxf(0.0f, in0__u0);
        float in0__u1 = x[a0 * 4 + 1];
        float in1__u1 = y[a0 * 4 + 1];
        float v1__u1 = in1__u1 * fmaxf(0.0f, in0__u1);
        float in0__u2 = x[a0 * 4 + 2];
        float in1__u2 = y[a0 * 4 + 2];
        float v1__u2 = in1__u2 * fmaxf(0.0f, in0__u2);
        float in0__u3 = x[a0 * 4 + 3];
        float in1__u3 = y[a0 * 4 + 3];
        float v1__u3 = in1__u3 * fmaxf(0.0f, in0__u3);
        mul[a0 * 4] = v1__u0;
        mul[a0 * 4 + 1] = v1__u1;
        mul[a0 * 4 + 2] = v1__u2;
        mul[a0 * 4 + 3] = v1__u3;
}
