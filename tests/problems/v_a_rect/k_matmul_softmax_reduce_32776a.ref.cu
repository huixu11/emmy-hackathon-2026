extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_32776a(const float* mul_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 512;
        int a5_b = (_gid / 256) % 2;
        int a0_u = (_gid / 16) % 16;
        int a5_u = (_gid) % 16;
        float in0 = mul_c1[0];
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        float acc1__c1_1 = -1e+30f;
        for (int a1 = 0; a1 < 128; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__ar0 = q[(a0_b * 32 + a0_u * 2) * 32 + a2];
                float in1__ar0 = kt[a2 * 128 + a1];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 32; a2++) {
                float in2__ar1 = q[(a0_b * 32 + a0_u * 2 + 1) * 32 + a2];
                float in1__ar1 = kt[a2 * 128 + a1];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__c0_0 = acc0__ar0 * in0;
            float one__c0_0 = 1.0f;
            float v1__gn__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__alpha__c0_0 = expf((acc1__c0_0 - v1__gn__c0_0));
            float v1__beta__c0_0 = expf((v1__c0_0 - v1__gn__c0_0));
            float v1__acc3_sa__c0_0 = acc3__c0_0 * v1__alpha__c0_0;
            float v1__acc3_sb__c0_0 = one__c0_0 * v1__beta__c0_0;
            acc3__c0_0 = v1__acc3_sa__c0_0 + v1__acc3_sb__c0_0;
            float v1__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v1__alpha__c0_0;
            float in7__bc0 = v[a1 * 48 + (a5_b * 32 + a5_u * 2) % 48];
            float v1__acc5__sum_sb__c0_0 = in7__bc0 * v1__beta__c0_0;
            acc5__sum__c0_0 = v1__acc5__sum_sa__c0_0 + v1__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
            float v1__c0_1 = acc0__ar0 * in0;
            float one__c0_1 = 1.0f;
            float v1__gn__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__alpha__c0_1 = expf((acc1__c0_1 - v1__gn__c0_1));
            float v1__beta__c0_1 = expf((v1__c0_1 - v1__gn__c0_1));
            float v1__acc3_sa__c0_1 = acc3__c0_1 * v1__alpha__c0_1;
            float v1__acc3_sb__c0_1 = one__c0_1 * v1__beta__c0_1;
            acc3__c0_1 = v1__acc3_sa__c0_1 + v1__acc3_sb__c0_1;
            float v1__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v1__alpha__c0_1;
            float in7__bc1 = v[a1 * 48 + (a5_b * 32 + a5_u * 2 + 1) % 48];
            float v1__acc5__sum_sb__c0_1 = in7__bc1 * v1__beta__c0_1;
            acc5__sum__c0_1 = v1__acc5__sum_sa__c0_1 + v1__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__c1_0 = acc0__ar1 * in0;
            float one__c1_0 = 1.0f;
            float v1__gn__c1_0 = fmaxf(acc1__c1_0, v1__c1_0);
            float v1__alpha__c1_0 = expf((acc1__c1_0 - v1__gn__c1_0));
            float v1__beta__c1_0 = expf((v1__c1_0 - v1__gn__c1_0));
            float v1__acc3_sa__c1_0 = acc3__c1_0 * v1__alpha__c1_0;
            float v1__acc3_sb__c1_0 = one__c1_0 * v1__beta__c1_0;
            acc3__c1_0 = v1__acc3_sa__c1_0 + v1__acc3_sb__c1_0;
            float v1__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v1__alpha__c1_0;
            float v1__acc5__sum_sb__c1_0 = in7__bc0 * v1__beta__c1_0;
            acc5__sum__c1_0 = v1__acc5__sum_sa__c1_0 + v1__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v1__c1_0);
            float v1__c1_1 = acc0__ar1 * in0;
            float one__c1_1 = 1.0f;
            float v1__gn__c1_1 = fmaxf(acc1__c1_1, v1__c1_1);
            float v1__alpha__c1_1 = expf((acc1__c1_1 - v1__gn__c1_1));
            float v1__beta__c1_1 = expf((v1__c1_1 - v1__gn__c1_1));
            float v1__acc3_sa__c1_1 = acc3__c1_1 * v1__alpha__c1_1;
            float v1__acc3_sb__c1_1 = one__c1_1 * v1__beta__c1_1;
            acc3__c1_1 = v1__acc3_sa__c1_1 + v1__acc3_sb__c1_1;
            float v1__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v1__alpha__c1_1;
            float v1__acc5__sum_sb__c1_1 = in7__bc1 * v1__beta__c1_1;
            acc5__sum__c1_1 = v1__acc5__sum_sa__c1_1 + v1__acc5__sum_sb__c1_1;
            acc1__c1_1 = fmaxf(acc1__c1_1, v1__c1_1);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a5_b * 32 + a5_u * 2 + 0 < 48) {
            matmul_1[(a0_b * 32 + a0_u * 2) * 48 + (a5_b * 32 + a5_u * 2) % 48] = acc5__c0_0;
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a5_b * 32 + a5_u * 2 + 1 < 48) {
            matmul_1[(a0_b * 32 + a0_u * 2) * 48 + (a5_b * 32 + a5_u * 2 + 1) % 48] = acc5__c0_1;
        }
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        if (a5_b * 32 + a5_u * 2 + 0 < 48) {
            matmul_1[(a0_b * 32 + a0_u * 2 + 1) * 48 + (a5_b * 32 + a5_u * 2) % 48] = acc5__c1_0;
        }
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        if (a5_b * 32 + a5_u * 2 + 1 < 48) {
            matmul_1[(a0_b * 32 + a0_u * 2 + 1) * 48 + (a5_b * 32 + a5_u * 2 + 1) % 48] = acc5__c1_1;
        }
}
