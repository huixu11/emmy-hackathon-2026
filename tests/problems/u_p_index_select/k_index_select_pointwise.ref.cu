extern "C" __global__
__launch_bounds__(256) void k_index_select_pointwise(const float* gt_c1, const float* x, const float* w, float* index_select) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 8;
        int a1 = (_gid) % 8;
        float in0__u0 = gt_c1[0];
        float in1__u0 = x[a0];
        bool v0__u0 = in1__u0 > in0__u0;
        long long v1__u0 = v0__u0;
        float in0__u1 = gt_c1[0];
        float in1__u1 = x[a0];
        bool v0__u1 = in1__u1 > in0__u1;
        long long v1__u1 = v0__u1;
        float in0__u2 = gt_c1[0];
        float in1__u2 = x[a0];
        bool v0__u2 = in1__u2 > in0__u2;
        long long v1__u2 = v0__u2;
        float in0__u3 = gt_c1[0];
        float in1__u3 = x[a0];
        bool v0__u3 = in1__u3 > in0__u3;
        long long v1__u3 = v0__u3;
        float in2__u0 = w[((int)(v1__u0)) * 32 + a1 * 4];
        float in2__u1 = w[((int)(v1__u1)) * 32 + (a1 * 4 + 1)];
        float in2__u2 = w[((int)(v1__u2)) * 32 + (a1 * 4 + 2)];
        float in2__u3 = w[((int)(v1__u3)) * 32 + (a1 * 4 + 3)];
        float4 _vs_in2__u0_800ede55 = make_float4(in2__u0, in2__u1, in2__u2, in2__u3);
        *reinterpret_cast<float4*>(&index_select[a0 * 32 + a1 * 4]) = _vs_in2__u0_800ede55;
}
