extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const float* eq_c1, const int* add_c1, const int* x, int* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in0__u0 = add_c1[0];
        int in2__u0 = x[a0 * 4];
        float in1__u0 = eq_c1[0];
        bool v1__u0 = in2__u0 == in1__u0;
        int v2__u0 = ((v1__u0) ? ((in0__u0 + in2__u0)) : (in2__u0));
        int in0__u1 = add_c1[0];
        int in2__u1 = x[a0 * 4 + 1];
        float in1__u1 = eq_c1[0];
        bool v1__u1 = in2__u1 == in1__u1;
        int v2__u1 = ((v1__u1) ? ((in0__u1 + in2__u1)) : (in2__u1));
        int in0__u2 = add_c1[0];
        int in2__u2 = x[a0 * 4 + 2];
        float in1__u2 = eq_c1[0];
        bool v1__u2 = in2__u2 == in1__u2;
        int v2__u2 = ((v1__u2) ? ((in0__u2 + in2__u2)) : (in2__u2));
        int in0__u3 = add_c1[0];
        int in2__u3 = x[a0 * 4 + 3];
        float in1__u3 = eq_c1[0];
        bool v1__u3 = in2__u3 == in1__u3;
        int v2__u3 = ((v1__u3) ? ((in0__u3 + in2__u3)) : (in2__u3));
        where[a0 * 4] = v2__u0;
        where[a0 * 4 + 1] = v2__u1;
        where[a0 * 4 + 2] = v2__u2;
        where[a0 * 4 + 3] = v2__u3;
}
