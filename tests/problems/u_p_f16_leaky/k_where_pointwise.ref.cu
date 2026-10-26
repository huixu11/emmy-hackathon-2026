#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const __half* mul_c1, const float* ge_c1, const __half* x, __half* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = ge_c1[0];
        __half in2__u0 = x[a0 * 4];
        bool v0__u0 = __half2float(in2__u0) >= in0__u0;
        __half in1__u0 = mul_c1[0];
        __half v2__u0 = __float2half(((v0__u0) ? (__half2float(in2__u0)) : (__half2float((in1__u0 * in2__u0)))));
        float in0__u1 = ge_c1[0];
        __half in2__u1 = x[a0 * 4 + 1];
        bool v0__u1 = __half2float(in2__u1) >= in0__u1;
        __half in1__u1 = mul_c1[0];
        __half v2__u1 = __float2half(((v0__u1) ? (__half2float(in2__u1)) : (__half2float((in1__u1 * in2__u1)))));
        float in0__u2 = ge_c1[0];
        __half in2__u2 = x[a0 * 4 + 2];
        bool v0__u2 = __half2float(in2__u2) >= in0__u2;
        __half in1__u2 = mul_c1[0];
        __half v2__u2 = __float2half(((v0__u2) ? (__half2float(in2__u2)) : (__half2float((in1__u2 * in2__u2)))));
        float in0__u3 = ge_c1[0];
        __half in2__u3 = x[a0 * 4 + 3];
        bool v0__u3 = __half2float(in2__u3) >= in0__u3;
        __half in1__u3 = mul_c1[0];
        __half v2__u3 = __float2half(((v0__u3) ? (__half2float(in2__u3)) : (__half2float((in1__u3 * in2__u3)))));
        __half _vs_v2__u0_740a266d[4] = { v2__u0, v2__u1, v2__u2, v2__u3 };
        *reinterpret_cast<uint2*>(&where[a0 * 4]) = *reinterpret_cast<const uint2*>(_vs_v2__u0_740a266d);
}
