extern "C" __global__
__launch_bounds__(256) void k_remainder_pointwise(const int* remainder_c1, const int* x, int* remainder) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in0__u0 = remainder_c1[0];
        int in1__u0 = x[a0 * 4];
        int v0__u0 = in1__u0 % in0__u0;
        int in0__u1 = remainder_c1[0];
        int in1__u1 = x[a0 * 4 + 1];
        int v0__u1 = in1__u1 % in0__u1;
        int in0__u2 = remainder_c1[0];
        int in1__u2 = x[a0 * 4 + 2];
        int v0__u2 = in1__u2 % in0__u2;
        int in0__u3 = remainder_c1[0];
        int in1__u3 = x[a0 * 4 + 3];
        int v0__u3 = in1__u3 % in0__u3;
        remainder[a0 * 4] = v0__u0;
        remainder[a0 * 4 + 1] = v0__u1;
        remainder[a0 * 4 + 2] = v0__u2;
        remainder[a0 * 4 + 3] = v0__u3;
}
