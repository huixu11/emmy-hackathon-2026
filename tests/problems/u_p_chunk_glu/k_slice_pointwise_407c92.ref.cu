extern "C" __global__
__launch_bounds__(256) void k_slice_pointwise_407c92(const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 16;
        int a1 = (_gid) % 16;
        float in0__u0 = x[a0 * 128 + (a1 * 4 + 64)];
        float in1__u0 = x[a0 * 128 + a1 * 4];
        float v1__u0 = in1__u0 * (1.0f / (1.0f + expf(0.0f - in0__u0)));
        float in0__u1 = x[a0 * 128 + (a1 * 4 + 1 + 64)];
        float in1__u1 = x[a0 * 128 + (a1 * 4 + 1)];
        float v1__u1 = in1__u1 * (1.0f / (1.0f + expf(0.0f - in0__u1)));
        float in0__u2 = x[a0 * 128 + (a1 * 4 + 2 + 64)];
        float in1__u2 = x[a0 * 128 + (a1 * 4 + 2)];
        float v1__u2 = in1__u2 * (1.0f / (1.0f + expf(0.0f - in0__u2)));
        float in0__u3 = x[a0 * 128 + (a1 * 4 + 3 + 64)];
        float in1__u3 = x[a0 * 128 + (a1 * 4 + 3)];
        float v1__u3 = in1__u3 * (1.0f / (1.0f + expf(0.0f - in0__u3)));
        float4 _vs_v1__u0_cb42a068 = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 64 + a1 * 4]) = _vs_v1__u0_cb42a068;
}
