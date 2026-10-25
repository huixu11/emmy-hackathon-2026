extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const int* mul_c1, const int* x0, const int* x1, int* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in0__u0 = mul_c1[0];
        int in1__u0 = x0[a0 * 4];
        int in2__u0 = x1[a0 * 4];
        int v1__u0 = (in0__u0 * in1__u0) - in2__u0;
        int in0__u1 = mul_c1[0];
        int in1__u1 = x0[a0 * 4 + 1];
        int in2__u1 = x1[a0 * 4 + 1];
        int v1__u1 = (in0__u1 * in1__u1) - in2__u1;
        int in0__u2 = mul_c1[0];
        int in1__u2 = x0[a0 * 4 + 2];
        int in2__u2 = x1[a0 * 4 + 2];
        int v1__u2 = (in0__u2 * in1__u2) - in2__u2;
        int in0__u3 = mul_c1[0];
        int in1__u3 = x0[a0 * 4 + 3];
        int in2__u3 = x1[a0 * 4 + 3];
        int v1__u3 = (in0__u3 * in1__u3) - in2__u3;
        sub[a0 * 4] = v1__u0;
        sub[a0 * 4 + 1] = v1__u1;
        sub[a0 * 4 + 2] = v1__u2;
        sub[a0 * 4 + 3] = v1__u3;
}
