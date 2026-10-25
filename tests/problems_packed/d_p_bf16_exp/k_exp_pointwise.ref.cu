#include <cuda_bf16.h>
extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const __nv_bfloat16* x, __nv_bfloat16* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x[a0 * 4];
        float v0__u0 = expf(in0__u0);
        float in0__u1 = x[a0 * 4 + 1];
        float v0__u1 = expf(in0__u1);
        float in0__u2 = x[a0 * 4 + 2];
        float v0__u2 = expf(in0__u2);
        float in0__u3 = x[a0 * 4 + 3];
        float v0__u3 = expf(in0__u3);
        exp[a0 * 4] = v0__u0;
        exp[a0 * 4 + 1] = v0__u1;
        exp[a0 * 4 + 2] = v0__u2;
        exp[a0 * 4 + 3] = v0__u3;
}
