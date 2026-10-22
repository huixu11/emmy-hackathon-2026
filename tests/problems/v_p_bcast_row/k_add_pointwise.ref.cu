extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* add_c1, const float* b, const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 24;
        int a1 = (_gid) % 24;
        float in2__u0 = b[a1 * 4];
        float in1__u0 = x[a0 * 96 + a1 * 4];
        float in0__u0 = add_c1[0];
        float v1__u0 = in0__u0 + (in1__u0 * in2__u0);
        float in2__u1 = b[a1 * 4 + 1];
        float in1__u1 = x[a0 * 96 + (a1 * 4 + 1)];
        float in0__u1 = add_c1[0];
        float v1__u1 = in0__u1 + (in1__u1 * in2__u1);
        float in2__u2 = b[a1 * 4 + 2];
        float in1__u2 = x[a0 * 96 + (a1 * 4 + 2)];
        float in0__u2 = add_c1[0];
        float v1__u2 = in0__u2 + (in1__u2 * in2__u2);
        float in2__u3 = b[a1 * 4 + 3];
        float in1__u3 = x[a0 * 96 + (a1 * 4 + 3)];
        float in0__u3 = add_c1[0];
        float v1__u3 = in0__u3 + (in1__u3 * in2__u3);
        float4 _vs_v1__u0_4c73d4fd = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&add[a0 * 96 + a1 * 4]) = _vs_v1__u0_4c73d4fd;
}
