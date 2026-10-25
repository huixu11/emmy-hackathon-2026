extern "C" __global__
__launch_bounds__(256) void k_sqrt_pointwise(const double* add_c1, const double* x, double* sqrt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 4];
        float in0__u0 = add_c1[0];
        float v2__u0 = sqrtf((in0__u0 + fabsf(in1__u0)));
        float in1__u1 = x[a0 * 4 + 1];
        float in0__u1 = add_c1[0];
        float v2__u1 = sqrtf((in0__u1 + fabsf(in1__u1)));
        float in1__u2 = x[a0 * 4 + 2];
        float in0__u2 = add_c1[0];
        float v2__u2 = sqrtf((in0__u2 + fabsf(in1__u2)));
        float in1__u3 = x[a0 * 4 + 3];
        float in0__u3 = add_c1[0];
        float v2__u3 = sqrtf((in0__u3 + fabsf(in1__u3)));
        sqrt[a0 * 4] = v2__u0;
        sqrt[a0 * 4 + 1] = v2__u1;
        sqrt[a0 * 4 + 2] = v2__u2;
        sqrt[a0 * 4 + 3] = v2__u3;
}
