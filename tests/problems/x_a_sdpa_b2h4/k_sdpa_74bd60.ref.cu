extern "C" __global__
__launch_bounds__(256) void k_sdpa_74bd60(const float* scaled_dot_product_attention_scale, const float* q, const float* k, const float* v, float* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 256;
        int a1_b = (_gid / 256) % 1;
        int a6_b = (_gid / 256) % 1;
        int a1_u = (_gid / 16) % 16;
        int a6_u = (_gid) % 16;
        float in0 = scaled_dot_product_attention_scale[0];
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        #pragma unroll
        for (int a2 = 0; a2 < 32; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 16; a3++) {
                float in2__ar0 = q[a0 * 512 + (a1_b * 32 + a1_u * 2) * 16 + a3];
                float in1__ar0 = k[a0 * 512 + a2 * 16 + a3];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a3 = 0; a3 < 16; a3++) {
                float in2__ar1 = q[a0 * 512 + (a1_b * 32 + a1_u * 2 + 1) * 16 + a3];
                float in1__ar1 = k[a0 * 512 + a2 * 16 + a3];
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
            float in7__bc0 = v[a0 * 512 + a2 * 16 + (a6_b * 16 + a6_u)];
            float v1__acc5__sum_sb__c0_0 = in7__bc0 * v1__beta__c0_0;
            acc5__sum__c0_0 = v1__acc5__sum_sa__c0_0 + v1__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v1__c0_0);
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
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        scaled_dot_product_attention[a0 * 512 + (a1_b * 32 + a1_u * 2) * 16 + (a6_b * 16 + a6_u)] = acc5__c0_0;
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        scaled_dot_product_attention[a0 * 512 + (a1_b * 32 + a1_u * 2 + 1) * 16 + (a6_b * 16 + a6_u)] = acc5__c1_0;
}
