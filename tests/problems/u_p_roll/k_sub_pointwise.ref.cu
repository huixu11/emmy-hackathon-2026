extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* x, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x[a0 * 4 + -7];
        float in1__u0 = x[a0 * 4 + 4089];
        float v0__u0 = ((a0 * 4 + 0 < 7) ? (((float)(in1__u0))) : (((float)(in0__u0))));
        float in2__u0 = x[a0 * 4];
        float v1__u0 = v0__u0 - in2__u0;
        float in0__u1 = x[a0 * 4 + 1 + -7];
        float in1__u1 = x[a0 * 4 + 1 + 4089];
        float v0__u1 = ((a0 * 4 + 1 < 7) ? (((float)(in1__u1))) : (((float)(in0__u1))));
        float in2__u1 = x[a0 * 4 + 1];
        float v1__u1 = v0__u1 - in2__u1;
        float in0__u2 = x[a0 * 4 + 2 + -7];
        float in1__u2 = x[a0 * 4 + 2 + 4089];
        float v0__u2 = ((a0 * 4 + 2 < 7) ? (((float)(in1__u2))) : (((float)(in0__u2))));
        float in2__u2 = x[a0 * 4 + 2];
        float v1__u2 = v0__u2 - in2__u2;
        float in0__u3 = x[a0 * 4 + 3 + -7];
        float in1__u3 = x[a0 * 4 + 3 + 4089];
        float v0__u3 = ((a0 * 4 + 3 < 7) ? (((float)(in1__u3))) : (((float)(in0__u3))));
        float in2__u3 = x[a0 * 4 + 3];
        float v1__u3 = v0__u3 - in2__u3;
        float4 _vs_v1__u0_e89a3caa = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&sub[a0 * 4]) = _vs_v1__u0_e89a3caa;
}
