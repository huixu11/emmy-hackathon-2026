#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const __half* x1, const __half* x0, const __half* x2, __half* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        __half in0__u0 = x0[a0 * 4];
        __half in1__u0 = x1[a0 * 4];
        __half in2__u0 = x2[a0 * 4];
        __half v1__u0 = in2__u0 + (in0__u0 * in1__u0);
        __half in0__u1 = x0[a0 * 4 + 1];
        __half in1__u1 = x1[a0 * 4 + 1];
        __half in2__u1 = x2[a0 * 4 + 1];
        __half v1__u1 = in2__u1 + (in0__u1 * in1__u1);
        __half in0__u2 = x0[a0 * 4 + 2];
        __half in1__u2 = x1[a0 * 4 + 2];
        __half in2__u2 = x2[a0 * 4 + 2];
        __half v1__u2 = in2__u2 + (in0__u2 * in1__u2);
        __half in0__u3 = x0[a0 * 4 + 3];
        __half in1__u3 = x1[a0 * 4 + 3];
        __half in2__u3 = x2[a0 * 4 + 3];
        __half v1__u3 = in2__u3 + (in0__u3 * in1__u3);
        __half _vs_v1__u0_93c81c5a[4] = { v1__u0, v1__u1, v1__u2, v1__u3 };
        *reinterpret_cast<uint2*>(&add[a0 * 4]) = *reinterpret_cast<const uint2*>(_vs_v1__u0_93c81c5a);
}
