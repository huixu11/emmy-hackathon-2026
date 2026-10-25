extern "C" __global__
__launch_bounds__(256) void k_tanh_pointwise(const double* x, double* tanh) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x[a0 * 4];
        float v0__u0 = tanhf(in0__u0);
        float in0__u1 = x[a0 * 4 + 1];
        float v0__u1 = tanhf(in0__u1);
        float in0__u2 = x[a0 * 4 + 2];
        float v0__u2 = tanhf(in0__u2);
        float in0__u3 = x[a0 * 4 + 3];
        float v0__u3 = tanhf(in0__u3);
        tanh[a0 * 4] = v0__u0;
        tanh[a0 * 4 + 1] = v0__u1;
        tanh[a0 * 4 + 2] = v0__u2;
        tanh[a0 * 4 + 3] = v0__u3;
}
