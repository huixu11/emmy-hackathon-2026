extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* add_c1, const float* clamp_bound0, const float* clamp_bound1, const float* div_c1, const float* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in3__u0 = div_c1[0];
        float in0__u0 = add_c1[0];
        float in4__u0 = x[a0 * 4];
        float in1__u0 = clamp_bound0[0];
        float in2__u0 = clamp_bound1[0];
        float v5__u0 = (1.0f / in3__u0) * (in4__u0 * fminf(in2__u0, fmaxf(in1__u0, (in0__u0 + in4__u0))));
        float in3__u1 = div_c1[0];
        float in0__u1 = add_c1[0];
        float in4__u1 = x[a0 * 4 + 1];
        float in1__u1 = clamp_bound0[0];
        float in2__u1 = clamp_bound1[0];
        float v5__u1 = (1.0f / in3__u1) * (in4__u1 * fminf(in2__u1, fmaxf(in1__u1, (in0__u1 + in4__u1))));
        float in3__u2 = div_c1[0];
        float in0__u2 = add_c1[0];
        float in4__u2 = x[a0 * 4 + 2];
        float in1__u2 = clamp_bound0[0];
        float in2__u2 = clamp_bound1[0];
        float v5__u2 = (1.0f / in3__u2) * (in4__u2 * fminf(in2__u2, fmaxf(in1__u2, (in0__u2 + in4__u2))));
        float in3__u3 = div_c1[0];
        float in0__u3 = add_c1[0];
        float in4__u3 = x[a0 * 4 + 3];
        float in1__u3 = clamp_bound0[0];
        float in2__u3 = clamp_bound1[0];
        float v5__u3 = (1.0f / in3__u3) * (in4__u3 * fminf(in2__u3, fmaxf(in1__u3, (in0__u3 + in4__u3))));
        float4 _vs_v5__u0_ae265c4e = make_float4(v5__u0, v5__u1, v5__u2, v5__u3);
        *reinterpret_cast<float4*>(&div[a0 * 4]) = _vs_v5__u0_ae265c4e;
}
