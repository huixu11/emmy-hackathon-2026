extern "C" __global__
__launch_bounds__(256) void k_conv1d(const float* x, const float* w, float* conv1d) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 168) {
        int _a2_ks = _gid / 84;
        int a0_b = (_gid / 21) % 4;
        int a1_b = (_gid) % 21;
        int a0_u = (_gid) % 1;
        int a1_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c1_2 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 6; a2++) {
            float in1__ar0 = w[(a0_b * 2 + a0_u * 2) * 12 + (_a2_ks * 6 + a2) % 4 * 3 + (_a2_ks * 6 + a2) / 4];
            float in0__bc0 = x[(_a2_ks * 6 + a2) % 4 * 64 + ((_a2_ks * 6 + a2) / 4 + (a1_b * 3 + a1_u * 3) % 62)];
            float v0__c0_0 = in0__bc0 * in1__ar0;
            acc0__c0_0 += v0__c0_0;
            float in0__bc1 = x[(_a2_ks * 6 + a2) % 4 * 64 + ((_a2_ks * 6 + a2) / 4 + (a1_b * 3 + a1_u * 3 + 1) % 62)];
            float v0__c0_1 = in0__bc1 * in1__ar0;
            acc0__c0_1 += v0__c0_1;
            float in0__bc2 = x[(_a2_ks * 6 + a2) % 4 * 64 + ((_a2_ks * 6 + a2) / 4 + (a1_b * 3 + a1_u * 3 + 2) % 62)];
            float v0__c0_2 = in0__bc2 * in1__ar0;
            acc0__c0_2 += v0__c0_2;
            float in1__ar1 = w[(a0_b * 2 + a0_u * 2 + 1) * 12 + (_a2_ks * 6 + a2) % 4 * 3 + (_a2_ks * 6 + a2) / 4];
            float v0__c1_0 = in0__bc0 * in1__ar1;
            acc0__c1_0 += v0__c1_0;
            float v0__c1_1 = in0__bc1 * in1__ar1;
            acc0__c1_1 += v0__c1_1;
            float v0__c1_2 = in0__bc2 * in1__ar1;
            acc0__c1_2 += v0__c1_2;
        }
        if (a1_b * 3 + a1_u * 3 + 0 < 62) {
            atomicAdd(&conv1d[(a0_b * 2 + a0_u * 2) * 62 + (a1_b * 3 + a1_u * 3) % 62], acc0__c0_0);
        }
        if (a1_b * 3 + a1_u * 3 + 1 < 62) {
            atomicAdd(&conv1d[(a0_b * 2 + a0_u * 2) * 62 + (a1_b * 3 + a1_u * 3 + 1) % 62], acc0__c0_1);
        }
        if (a1_b * 3 + a1_u * 3 + 2 < 62) {
            atomicAdd(&conv1d[(a0_b * 2 + a0_u * 2) * 62 + (a1_b * 3 + a1_u * 3 + 2) % 62], acc0__c0_2);
        }
        if (a1_b * 3 + a1_u * 3 + 0 < 62) {
            atomicAdd(&conv1d[(a0_b * 2 + a0_u * 2 + 1) * 62 + (a1_b * 3 + a1_u * 3) % 62], acc0__c1_0);
        }
        if (a1_b * 3 + a1_u * 3 + 1 < 62) {
            atomicAdd(&conv1d[(a0_b * 2 + a0_u * 2 + 1) * 62 + (a1_b * 3 + a1_u * 3 + 1) % 62], acc0__c1_1);
        }
        if (a1_b * 3 + a1_u * 3 + 2 < 62) {
            atomicAdd(&conv1d[(a0_b * 2 + a0_u * 2 + 1) * 62 + (a1_b * 3 + a1_u * 3 + 2) % 62], acc0__c1_2);
        }
    }
}
