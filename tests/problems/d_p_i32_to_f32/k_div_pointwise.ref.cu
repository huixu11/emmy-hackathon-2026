extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* div_c1, const int* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = div_c1[0];
        int in1__u0 = x[a0 * 4];
        float v1__u0 = in1__u0 * (1.0f / in0__u0);
        float in0__u1 = div_c1[0];
        int in1__u1 = x[a0 * 4 + 1];
        float v1__u1 = in1__u1 * (1.0f / in0__u1);
        float in0__u2 = div_c1[0];
        int in1__u2 = x[a0 * 4 + 2];
        float v1__u2 = in1__u2 * (1.0f / in0__u2);
        float in0__u3 = div_c1[0];
        int in1__u3 = x[a0 * 4 + 3];
        float v1__u3 = in1__u3 * (1.0f / in0__u3);
        float4 _vs_v1__u0_79d4309a = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&div[a0 * 4]) = _vs_v1__u0_79d4309a;
}
