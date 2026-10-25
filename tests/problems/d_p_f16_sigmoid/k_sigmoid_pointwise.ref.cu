#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_sigmoid_pointwise(const __half* x, __half* sigmoid) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        uint2 _v_in0__u0 = *reinterpret_cast<const uint2*>(&x[a0 * 4]);
        const __half* _v_in0__u0_h = reinterpret_cast<const __half*>(&_v_in0__u0);
        __half in0__u0 = _v_in0__u0_h[0];
        __half in0__u1 = _v_in0__u0_h[1];
        __half in0__u2 = _v_in0__u0_h[2];
        __half in0__u3 = _v_in0__u0_h[3];
        __half v0__u0 = __float2half(1.0f / (1.0f + expf(0.0f - __half2float(in0__u0))));
        __half v0__u1 = __float2half(1.0f / (1.0f + expf(0.0f - __half2float(in0__u1))));
        __half v0__u2 = __float2half(1.0f / (1.0f + expf(0.0f - __half2float(in0__u2))));
        __half v0__u3 = __float2half(1.0f / (1.0f + expf(0.0f - __half2float(in0__u3))));
        __half _vs_v0__u0_13811a1e[4] = { v0__u0, v0__u1, v0__u2, v0__u3 };
        *reinterpret_cast<uint2*>(&sigmoid[a0 * 4]) = *reinterpret_cast<const uint2*>(_vs_v0__u0_13811a1e);
}
