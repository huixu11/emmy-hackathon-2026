extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* mul_c1, const float* b, const float* a, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 512;
        int a1 = (_gid) % 512;
        float in1__u0 = a[a1 * 4];
        float in2__u0 = b[a1 * 4];
        float v0__u0 = ((0 == a0) ? (((float)(in1__u0))) : (((float)(in2__u0))));
        float in0__u0 = mul_c1[0];
        float v1__u0 = in0__u0 * v0__u0;
        float in1__u1 = a[a1 * 4 + 1];
        float in2__u1 = b[a1 * 4 + 1];
        float v0__u1 = ((0 == a0) ? (((float)(in1__u1))) : (((float)(in2__u1))));
        float in0__u1 = mul_c1[0];
        float v1__u1 = in0__u1 * v0__u1;
        float in1__u2 = a[a1 * 4 + 2];
        float in2__u2 = b[a1 * 4 + 2];
        float v0__u2 = ((0 == a0) ? (((float)(in1__u2))) : (((float)(in2__u2))));
        float in0__u2 = mul_c1[0];
        float v1__u2 = in0__u2 * v0__u2;
        float in1__u3 = a[a1 * 4 + 3];
        float in2__u3 = b[a1 * 4 + 3];
        float v0__u3 = ((0 == a0) ? (((float)(in1__u3))) : (((float)(in2__u3))));
        float in0__u3 = mul_c1[0];
        float v1__u3 = in0__u3 * v0__u3;
        mul[a1 / 8 * 64 + a0 * 32 + 4 * (a1 % 8)] = v1__u0;
        mul[(a1 * 4 + 1) / 32 * 64 + a0 * 32 + (a1 * 4 + 1) % 32] = v1__u1;
        mul[(a1 * 4 + 2) / 32 * 64 + a0 * 32 + (a1 * 4 + 2) % 32] = v1__u2;
        mul[(a1 * 4 + 3) / 32 * 64 + a0 * 32 + (a1 * 4 + 3) % 32] = v1__u3;
}
