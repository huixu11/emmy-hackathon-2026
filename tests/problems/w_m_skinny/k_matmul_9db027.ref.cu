extern "C" __global__
__launch_bounds__(256) void k_matmul_9db027(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 64) {
        int a0_b = _gid / 8;
        int a1_b = (_gid) % 8;
        int a0_u = (_gid) % 1;
        int a1_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        for (int a2 = 0; a2 < 2048; a2++) {
            float in0__ar0 = x0[(a0_b + a0_u) * 2048 + a2];
            float2 _v_in1__bc0 = *reinterpret_cast<const float2*>(&x1[a2 * 16 + (a1_b * 2 + a1_u * 2)]);
            float in1__bc0 = _v_in1__bc0.x;
            float in1__bc1 = _v_in1__bc0.y;
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
        }
        float2 _vs_acc0__c0_0_7ed3280a = make_float2(acc0__c0_0, acc0__c0_1);
        *reinterpret_cast<float2*>(&matmul[(a0_b + a0_u) * 16 + (a1_b * 2 + a1_u * 2)]) = _vs_acc0__c0_0_7ed3280a;
    }
}
