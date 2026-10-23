extern "C" __global__
__launch_bounds__(256) void k_sdpa_8751ff(const float* scaled_dot_product_attention_mask_fill_c, const float* scaled_dot_product_attention_mask_zero_c, const float* scaled_dot_product_attention_scale, const float* q, const float* k, const float* v, float* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 512;
        int a1_b = (_gid / 256) % 2;
        int a6_b = (_gid / 256) % 1;
        int a1_u = (_gid / 16) % 16;
        int a6_u = (_gid) % 16;
        float in0 = scaled_dot_product_attention_mask_fill_c[0];
        float in1 = scaled_dot_product_attention_mask_zero_c[0];
        float in2 = scaled_dot_product_attention_scale[0];
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
        #pragma unroll
        for (int a2 = 0; a2 < 64; a2++) {
            float acc0__ar0 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in4__ar0 = q[a0 * 2048 + (a1_b * 32 + a1_u * 2) * 32 + a3];
                float in3__ar0 = k[a0 * 2048 + a2 * 32 + a3];
                float v0__ar0 = in3__ar0 * in4__ar0;
                acc0__ar0 += v0__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                float in4__ar1 = q[a0 * 2048 + (a1_b * 32 + a1_u * 2 + 1) * 32 + a3];
                float in3__ar1 = k[a0 * 2048 + a2 * 32 + a3];
                float v0__ar1 = in3__ar1 * in4__ar1;
                acc0__ar1 += v0__ar1;
            }
            float one__c0_0 = 1.0f;
            float v2__c0_0 = ((a2 <= a1_b * 32 + a1_u * 2 + 0) ? (((float)(in1))) : (((float)(in0))));
            float v3__c0_0 = (acc0__ar0 * in2) + v2__c0_0;
            float v3__gn__c0_0 = fmaxf(acc1__c0_0, v3__c0_0);
            float v3__alpha__c0_0 = expf((acc1__c0_0 - v3__gn__c0_0));
            float v3__beta__c0_0 = expf((v3__c0_0 - v3__gn__c0_0));
            float v3__acc3_sa__c0_0 = acc3__c0_0 * v3__alpha__c0_0;
            float v3__acc3_sb__c0_0 = one__c0_0 * v3__beta__c0_0;
            acc3__c0_0 = v3__acc3_sa__c0_0 + v3__acc3_sb__c0_0;
            float v3__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v3__alpha__c0_0;
            float2 _v_in9__bc0 = *reinterpret_cast<const float2*>(&v[a0 * 2048 + a2 * 32 + (a6_b * 32 + a6_u * 2)]);
            float in9__bc0 = _v_in9__bc0.x;
            float in9__bc1 = _v_in9__bc0.y;
            float v3__acc5__sum_sb__c0_0 = in9__bc0 * v3__beta__c0_0;
            acc5__sum__c0_0 = v3__acc5__sum_sa__c0_0 + v3__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v3__c0_0);
            float one__c0_1 = 1.0f;
            float v2__c0_1 = ((a2 <= a1_b * 32 + a1_u * 2 + 0) ? (((float)(in1))) : (((float)(in0))));
            float v3__c0_1 = (acc0__ar0 * in2) + v2__c0_1;
            float v3__gn__c0_1 = fmaxf(acc1__c0_1, v3__c0_1);
            float v3__alpha__c0_1 = expf((acc1__c0_1 - v3__gn__c0_1));
            float v3__beta__c0_1 = expf((v3__c0_1 - v3__gn__c0_1));
            float v3__acc3_sa__c0_1 = acc3__c0_1 * v3__alpha__c0_1;
            float v3__acc3_sb__c0_1 = one__c0_1 * v3__beta__c0_1;
            acc3__c0_1 = v3__acc3_sa__c0_1 + v3__acc3_sb__c0_1;
            float v3__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v3__alpha__c0_1;
            float v3__acc5__sum_sb__c0_1 = in9__bc1 * v3__beta__c0_1;
            acc5__sum__c0_1 = v3__acc5__sum_sa__c0_1 + v3__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v3__c0_1);
            float one__c1_0 = 1.0f;
            float v2__c1_0 = ((a2 <= a1_b * 32 + a1_u * 2 + 1) ? (((float)(in1))) : (((float)(in0))));
            float v3__c1_0 = (acc0__ar1 * in2) + v2__c1_0;
            float v3__gn__c1_0 = fmaxf(acc1__c1_0, v3__c1_0);
            float v3__alpha__c1_0 = expf((acc1__c1_0 - v3__gn__c1_0));
            float v3__beta__c1_0 = expf((v3__c1_0 - v3__gn__c1_0));
            float v3__acc3_sa__c1_0 = acc3__c1_0 * v3__alpha__c1_0;
            float v3__acc3_sb__c1_0 = one__c1_0 * v3__beta__c1_0;
            acc3__c1_0 = v3__acc3_sa__c1_0 + v3__acc3_sb__c1_0;
            float v3__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v3__alpha__c1_0;
            float v3__acc5__sum_sb__c1_0 = in9__bc0 * v3__beta__c1_0;
            acc5__sum__c1_0 = v3__acc5__sum_sa__c1_0 + v3__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v3__c1_0);
            float one__c1_1 = 1.0f;
            float v2__c1_1 = ((a2 <= a1_b * 32 + a1_u * 2 + 1) ? (((float)(in1))) : (((float)(in0))));
            float v3__c1_1 = (acc0__ar1 * in2) + v2__c1_1;
            float v3__gn__c1_1 = fmaxf(acc1__c1_1, v3__c1_1);
            float v3__alpha__c1_1 = expf((acc1__c1_1 - v3__gn__c1_1));
            float v3__beta__c1_1 = expf((v3__c1_1 - v3__gn__c1_1));
            float v3__acc3_sa__c1_1 = acc3__c1_1 * v3__alpha__c1_1;
            float v3__acc3_sb__c1_1 = one__c1_1 * v3__beta__c1_1;
            acc3__c1_1 = v3__acc3_sa__c1_1 + v3__acc3_sb__c1_1;
            float v3__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v3__alpha__c1_1;
            float v3__acc5__sum_sb__c1_1 = in9__bc1 * v3__beta__c1_1;
            acc5__sum__c1_1 = v3__acc5__sum_sa__c1_1 + v3__acc5__sum_sb__c1_1;
            acc1__c1_1 = fmaxf(acc1__c1_1, v3__c1_1);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        scaled_dot_product_attention[a0 * 2048 + (a1_b * 32 + a1_u * 2) * 32 + (a6_b * 32 + a6_u * 2)] = acc5__c0_0;
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        scaled_dot_product_attention[a0 * 2048 + (a1_b * 32 + a1_u * 2) * 32 + (a6_b * 32 + a6_u * 2 + 1)] = acc5__c0_1;
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        scaled_dot_product_attention[a0 * 2048 + (a1_b * 32 + a1_u * 2 + 1) * 32 + (a6_b * 32 + a6_u * 2)] = acc5__c1_0;
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        scaled_dot_product_attention[a0 * 2048 + (a1_b * 32 + a1_u * 2 + 1) * 32 + (a6_b * 32 + a6_u * 2 + 1)] = acc5__c1_1;
}
