#include <cuda_bf16.h>
extern "C" __global__
__launch_bounds__(256) void k_to_pointwise(const float* x, __nv_bfloat16* to) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        float v0__u0 = in0__u0;
        float v0__u1 = in0__u1;
        float v0__u2 = in0__u2;
        float v0__u3 = in0__u3;
        to[a0 * 4] = v0__u0;
        to[a0 * 4 + 1] = v0__u1;
        to[a0 * 4 + 2] = v0__u2;
        to[a0 * 4 + 3] = v0__u3;
}
