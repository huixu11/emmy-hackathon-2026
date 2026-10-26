extern "C" __global__
__launch_bounds__(256) void k_cat_f2f1d0(const float* mul_c1, const float* b, const float* a, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 20;
        int a1 = (_gid) % 20;
        float in2__u0 = a[a0 * 32 + ((a1 * 4 < 32) ? (a1 * 4) : (0))];
        float in1__u0 = b[a0 * 48 + ((a1 * 4 < 32) ? (0) : (a1 * 4 + -32))];
        float v0__u0 = ((a1 * 4 + 0 < 32) ? (((float)(in2__u0))) : (((float)(in1__u0))));
        float in0__u0 = mul_c1[0];
        float v1__u0 = in0__u0 * v0__u0;
        float in2__u1 = a[a0 * 32 + ((a1 * 4 + 1 < 32) ? (a1 * 4 + 1) : (0))];
        float in1__u1 = b[a0 * 48 + ((a1 * 4 + 1 < 32) ? (0) : (a1 * 4 + 1 + -32))];
        float v0__u1 = ((a1 * 4 + 1 < 32) ? (((float)(in2__u1))) : (((float)(in1__u1))));
        float in0__u1 = mul_c1[0];
        float v1__u1 = in0__u1 * v0__u1;
        float in2__u2 = a[a0 * 32 + ((a1 * 4 + 2 < 32) ? (a1 * 4 + 2) : (0))];
        float in1__u2 = b[a0 * 48 + ((a1 * 4 + 2 < 32) ? (0) : (a1 * 4 + 2 + -32))];
        float v0__u2 = ((a1 * 4 + 2 < 32) ? (((float)(in2__u2))) : (((float)(in1__u2))));
        float in0__u2 = mul_c1[0];
        float v1__u2 = in0__u2 * v0__u2;
        float in2__u3 = a[a0 * 32 + ((a1 * 4 + 3 < 32) ? (a1 * 4 + 3) : (0))];
        float in1__u3 = b[a0 * 48 + ((a1 * 4 + 3 < 32) ? (0) : (a1 * 4 + 3 + -32))];
        float v0__u3 = ((a1 * 4 + 3 < 32) ? (((float)(in2__u3))) : (((float)(in1__u3))));
        float in0__u3 = mul_c1[0];
        float v1__u3 = in0__u3 * v0__u3;
        float4 _vs_v1__u0_5f3ef616 = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 80 + a1 * 4]) = _vs_v1__u0_5f3ef616;
}
