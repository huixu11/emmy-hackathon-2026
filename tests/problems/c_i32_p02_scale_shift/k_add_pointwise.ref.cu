extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const int* mul_c1, const int* add_c1, const int* x, int* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in1__u0 = mul_c1[0];
        int in2__u0 = x[a0 * 4];
        int in0__u0 = add_c1[0];
        int v1__u0 = in0__u0 + (in1__u0 * in2__u0);
        int in1__u1 = mul_c1[0];
        int in2__u1 = x[a0 * 4 + 1];
        int in0__u1 = add_c1[0];
        int v1__u1 = in0__u1 + (in1__u1 * in2__u1);
        int in1__u2 = mul_c1[0];
        int in2__u2 = x[a0 * 4 + 2];
        int in0__u2 = add_c1[0];
        int v1__u2 = in0__u2 + (in1__u2 * in2__u2);
        int in1__u3 = mul_c1[0];
        int in2__u3 = x[a0 * 4 + 3];
        int in0__u3 = add_c1[0];
        int v1__u3 = in0__u3 + (in1__u3 * in2__u3);
        add[a0 * 4] = v1__u0;
        add[a0 * 4 + 1] = v1__u1;
        add[a0 * 4 + 2] = v1__u2;
        add[a0 * 4 + 3] = v1__u3;
}
