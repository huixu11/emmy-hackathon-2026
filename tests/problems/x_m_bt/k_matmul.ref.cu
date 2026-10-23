extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* b, const float* a, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 3072;
        int a0_b = (_gid / 768) % 4;
        int a1_b = (_gid / 256) % 3;
        int a0_u = (_gid / 16) % 16;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 3; a2++) {
            float in0__ar0 = a[(a0_b * 16 + a0_u) * 48 + (_a2_ks * 3 + a2)];
            float in1__bc0 = b[(a1_b * 32 + a1_u * 2) % 80 * 48 + (_a2_ks * 3 + a2)];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in1__bc1 = b[(a1_b * 32 + a1_u * 2 + 1) % 80 * 48 + (_a2_ks * 3 + a2)];
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
        }
        if (a1_b * 32 + a1_u * 2 + 0 < 80) {
            atomicAdd(&matmul[(a0_b * 16 + a0_u) * 80 + (a1_b * 32 + a1_u * 2) % 80], acc0__c0_0);
        }
        if (a1_b * 32 + a1_u * 2 + 1 < 80) {
            atomicAdd(&matmul[(a0_b * 16 + a0_u) * 80 + (a1_b * 32 + a1_u * 2 + 1) % 80], acc0__c0_1);
        }
}
