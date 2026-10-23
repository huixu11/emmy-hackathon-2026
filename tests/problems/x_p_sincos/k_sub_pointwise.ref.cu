extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* t, const float* x, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = t[a0 * 4];
        float in1__u0 = x[a0 * 4];
        float v4__u0 = (in1__u0 * cosf(in0__u0)) - (in1__u0 * sinf(in0__u0));
        float in0__u1 = t[a0 * 4 + 1];
        float in1__u1 = x[a0 * 4 + 1];
        float v4__u1 = (in1__u1 * cosf(in0__u1)) - (in1__u1 * sinf(in0__u1));
        float in0__u2 = t[a0 * 4 + 2];
        float in1__u2 = x[a0 * 4 + 2];
        float v4__u2 = (in1__u2 * cosf(in0__u2)) - (in1__u2 * sinf(in0__u2));
        float in0__u3 = t[a0 * 4 + 3];
        float in1__u3 = x[a0 * 4 + 3];
        float v4__u3 = (in1__u3 * cosf(in0__u3)) - (in1__u3 * sinf(in0__u3));
        float4 _vs_v4__u0_0a07631b = make_float4(v4__u0, v4__u1, v4__u2, v4__u3);
        *reinterpret_cast<float4*>(&sub[a0 * 4]) = _vs_v4__u0_0a07631b;
}
