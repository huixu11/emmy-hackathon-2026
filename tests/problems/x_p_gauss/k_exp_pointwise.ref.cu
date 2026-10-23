extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const float* x, float* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        float v2__u0 = expf((in0__u0 * (0.0f - in0__u0)));
        float v2__u1 = expf((in0__u1 * (0.0f - in0__u1)));
        float v2__u2 = expf((in0__u2 * (0.0f - in0__u2)));
        float v2__u3 = expf((in0__u3 * (0.0f - in0__u3)));
        float4 _vs_v2__u0_fee848f3 = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&exp[a0 * 4]) = _vs_v2__u0_fee848f3;
}
