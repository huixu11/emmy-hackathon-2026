#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_to_pointwise(const float* x, __half* to) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        __half v0__u0 = __float2half(in0__u0);
        __half v0__u1 = __float2half(in0__u1);
        __half v0__u2 = __float2half(in0__u2);
        __half v0__u3 = __float2half(in0__u3);
        __half _vs_v0__u0_33029231[4] = { v0__u0, v0__u1, v0__u2, v0__u3 };
        *reinterpret_cast<uint2*>(&to[a0 * 4]) = *reinterpret_cast<const uint2*>(_vs_v0__u0_33029231);
}
