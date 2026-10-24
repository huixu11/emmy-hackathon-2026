extern "C" __global__
__launch_bounds__(256) void k_sdpa_232e98(const float* scaled_dot_product_attention_scale, const float* q, const float* k, const float* v, float* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 1024;
        int a1_b = (_gid / 256) % 4;
        int a6_b = (_gid / 256) % 1;
        int a1_u = (_gid / 16) % 16;
        int a6_u = (_gid) % 16;
        float in0 = scaled_dot_product_attention_scale[0];
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        float acc3__c0_2 = 0.0f;
        float acc5__sum__c0_2 = 0.0f;
        float acc1__c0_2 = -1e+30f;
        float acc3__c0_3 = 0.0f;
        float acc5__sum__c0_3 = 0.0f;
        float acc1__c0_3 = -1e+30f;
        #pragma unroll
        for (int a2 = 0; a2 < 64; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in2__ar0 = q[a0 * 2048 + (a1_b * 16 + a1_u) * 32 + a3];
                float in1__ar0 = k[a0 * 2048 + a2 * 32 + a3];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
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
            float in7__bc0 = v[a0 * 2048 + a2 * 32 + (a6_b * 64 + a6_u * 4) % 32];
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
            float in7__bc1 = v[a0 * 2048 + a2 * 32 + (a6_b * 64 + a6_u * 4 + 1) % 32];
            float v1__acc5__sum_sb__c0_1 = in7__bc1 * v1__beta__c0_1;
            acc5__sum__c0_1 = v1__acc5__sum_sa__c0_1 + v1__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v1__c0_1);
            float v1__c0_2 = acc0__ar0 * in0;
            float one__c0_2 = 1.0f;
            float v1__gn__c0_2 = fmaxf(acc1__c0_2, v1__c0_2);
            float v1__alpha__c0_2 = expf((acc1__c0_2 - v1__gn__c0_2));
            float v1__beta__c0_2 = expf((v1__c0_2 - v1__gn__c0_2));
            float v1__acc3_sa__c0_2 = acc3__c0_2 * v1__alpha__c0_2;
            float v1__acc3_sb__c0_2 = one__c0_2 * v1__beta__c0_2;
            acc3__c0_2 = v1__acc3_sa__c0_2 + v1__acc3_sb__c0_2;
            float v1__acc5__sum_sa__c0_2 = acc5__sum__c0_2 * v1__alpha__c0_2;
            float in7__bc2 = v[a0 * 2048 + a2 * 32 + (a6_b * 64 + a6_u * 4 + 2) % 32];
            float v1__acc5__sum_sb__c0_2 = in7__bc2 * v1__beta__c0_2;
            acc5__sum__c0_2 = v1__acc5__sum_sa__c0_2 + v1__acc5__sum_sb__c0_2;
            acc1__c0_2 = fmaxf(acc1__c0_2, v1__c0_2);
            float v1__c0_3 = acc0__ar0 * in0;
            float one__c0_3 = 1.0f;
            float v1__gn__c0_3 = fmaxf(acc1__c0_3, v1__c0_3);
            float v1__alpha__c0_3 = expf((acc1__c0_3 - v1__gn__c0_3));
            float v1__beta__c0_3 = expf((v1__c0_3 - v1__gn__c0_3));
            float v1__acc3_sa__c0_3 = acc3__c0_3 * v1__alpha__c0_3;
            float v1__acc3_sb__c0_3 = one__c0_3 * v1__beta__c0_3;
            acc3__c0_3 = v1__acc3_sa__c0_3 + v1__acc3_sb__c0_3;
            float v1__acc5__sum_sa__c0_3 = acc5__sum__c0_3 * v1__alpha__c0_3;
            float in7__bc3 = v[a0 * 2048 + a2 * 32 + (a6_b * 64 + a6_u * 4 + 3) % 32];
            float v1__acc5__sum_sb__c0_3 = in7__bc3 * v1__beta__c0_3;
            acc5__sum__c0_3 = v1__acc5__sum_sa__c0_3 + v1__acc5__sum_sb__c0_3;
            acc1__c0_3 = fmaxf(acc1__c0_3, v1__c0_3);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a6_b * 64 + a6_u * 4 + 0 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 16 + a1_u) * 32 + (a6_b * 64 + a6_u * 4) % 32] = acc5__c0_0;
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a6_b * 64 + a6_u * 4 + 1 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 16 + a1_u) * 32 + (a6_b * 64 + a6_u * 4 + 1) % 32] = acc5__c0_1;
        }
        float acc5__c0_2 = acc5__sum__c0_2 * (1.0f / acc3__c0_2);
        if (a6_b * 64 + a6_u * 4 + 2 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 16 + a1_u) * 32 + (a6_b * 64 + a6_u * 4 + 2) % 32] = acc5__c0_2;
        }
        float acc5__c0_3 = acc5__sum__c0_3 * (1.0f / acc3__c0_3);
        if (a6_b * 64 + a6_u * 4 + 3 < 32) {
            scaled_dot_product_attention[a0 * 2048 + (a1_b * 16 + a1_u) * 32 + (a6_b * 64 + a6_u * 4 + 3) % 32] = acc5__c0_3;
        }
}
