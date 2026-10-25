extern "C" __global__
__launch_bounds__(128) void k_scaled_dot_product_attention(const double* scaled_dot_product_attention_scale, const double* q, const double* k, const double* v, double* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a6_ks = _gid / 768;
        int a0_b = (_gid / 256) % 3;
        int a5_b = (_gid / 128) % 2;
        int a0_u = (_gid / 16) % 8;
        int a5_u = (_gid) % 16;
        float in0__ar0 = scaled_dot_product_attention_scale[0];
        float acc1__ar0 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar0 = q[(a0_b * 96 + a0_u * 12) % 256 * 64 + a2];
                float in1__ar0 = k[a1 * 64 + a2];
                float v0__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v0__ar0;
            }
            float v1__ar0 = acc0__ar0 * in0__ar0;
            acc1__ar0 = fmaxf(acc1__ar0, v1__ar0);
        }
        float acc3__ar0 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar0 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar0 = q[(a0_b * 96 + a0_u * 12) % 256 * 64 + a4];
                float in3__ar0 = k[a1 * 64 + a4];
                float v2__ar0 = in3__ar0 * in4__ar0;
                acc2__ar0 += v2__ar0;
            }
            float v3__ar0 = acc2__ar0 * in0__ar0;
            float v5__ar0 = expf((v3__ar0 - acc1__ar0));
            acc3__ar0 += v5__ar0;
        }
        float v6__ar0 = 1.0f / acc3__ar0;
        float in0__ar1 = scaled_dot_product_attention_scale[0];
        float acc1__ar1 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar1 = q[(a0_b * 96 + a0_u * 12 + 1) % 256 * 64 + a2];
                float in1__ar1 = k[a1 * 64 + a2];
                float v0__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v0__ar1;
            }
            float v1__ar1 = acc0__ar1 * in0__ar1;
            acc1__ar1 = fmaxf(acc1__ar1, v1__ar1);
        }
        float acc3__ar1 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar1 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar1 = q[(a0_b * 96 + a0_u * 12 + 1) % 256 * 64 + a4];
                float in3__ar1 = k[a1 * 64 + a4];
                float v2__ar1 = in3__ar1 * in4__ar1;
                acc2__ar1 += v2__ar1;
            }
            float v3__ar1 = acc2__ar1 * in0__ar1;
            float v5__ar1 = expf((v3__ar1 - acc1__ar1));
            acc3__ar1 += v5__ar1;
        }
        float v6__ar1 = 1.0f / acc3__ar1;
        float in0__ar2 = scaled_dot_product_attention_scale[0];
        float acc1__ar2 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar2 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar2 = q[(a0_b * 96 + a0_u * 12 + 2) % 256 * 64 + a2];
                float in1__ar2 = k[a1 * 64 + a2];
                float v0__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v0__ar2;
            }
            float v1__ar2 = acc0__ar2 * in0__ar2;
            acc1__ar2 = fmaxf(acc1__ar2, v1__ar2);
        }
        float acc3__ar2 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar2 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar2 = q[(a0_b * 96 + a0_u * 12 + 2) % 256 * 64 + a4];
                float in3__ar2 = k[a1 * 64 + a4];
                float v2__ar2 = in3__ar2 * in4__ar2;
                acc2__ar2 += v2__ar2;
            }
            float v3__ar2 = acc2__ar2 * in0__ar2;
            float v5__ar2 = expf((v3__ar2 - acc1__ar2));
            acc3__ar2 += v5__ar2;
        }
        float v6__ar2 = 1.0f / acc3__ar2;
        float in0__ar3 = scaled_dot_product_attention_scale[0];
        float acc1__ar3 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar3 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar3 = q[(a0_b * 96 + a0_u * 12 + 3) % 256 * 64 + a2];
                float in1__ar3 = k[a1 * 64 + a2];
                float v0__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v0__ar3;
            }
            float v1__ar3 = acc0__ar3 * in0__ar3;
            acc1__ar3 = fmaxf(acc1__ar3, v1__ar3);
        }
        float acc3__ar3 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar3 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar3 = q[(a0_b * 96 + a0_u * 12 + 3) % 256 * 64 + a4];
                float in3__ar3 = k[a1 * 64 + a4];
                float v2__ar3 = in3__ar3 * in4__ar3;
                acc2__ar3 += v2__ar3;
            }
            float v3__ar3 = acc2__ar3 * in0__ar3;
            float v5__ar3 = expf((v3__ar3 - acc1__ar3));
            acc3__ar3 += v5__ar3;
        }
        float v6__ar3 = 1.0f / acc3__ar3;
        float in0__ar4 = scaled_dot_product_attention_scale[0];
        float acc1__ar4 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar4 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar4 = q[(a0_b * 96 + a0_u * 12 + 4) % 256 * 64 + a2];
                float in1__ar4 = k[a1 * 64 + a2];
                float v0__ar4 = in1__ar4 * in2__ar4;
                acc0__ar4 += v0__ar4;
            }
            float v1__ar4 = acc0__ar4 * in0__ar4;
            acc1__ar4 = fmaxf(acc1__ar4, v1__ar4);
        }
        float acc3__ar4 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar4 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar4 = q[(a0_b * 96 + a0_u * 12 + 4) % 256 * 64 + a4];
                float in3__ar4 = k[a1 * 64 + a4];
                float v2__ar4 = in3__ar4 * in4__ar4;
                acc2__ar4 += v2__ar4;
            }
            float v3__ar4 = acc2__ar4 * in0__ar4;
            float v5__ar4 = expf((v3__ar4 - acc1__ar4));
            acc3__ar4 += v5__ar4;
        }
        float v6__ar4 = 1.0f / acc3__ar4;
        float in0__ar5 = scaled_dot_product_attention_scale[0];
        float acc1__ar5 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar5 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar5 = q[(a0_b * 96 + a0_u * 12 + 5) % 256 * 64 + a2];
                float in1__ar5 = k[a1 * 64 + a2];
                float v0__ar5 = in1__ar5 * in2__ar5;
                acc0__ar5 += v0__ar5;
            }
            float v1__ar5 = acc0__ar5 * in0__ar5;
            acc1__ar5 = fmaxf(acc1__ar5, v1__ar5);
        }
        float acc3__ar5 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar5 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar5 = q[(a0_b * 96 + a0_u * 12 + 5) % 256 * 64 + a4];
                float in3__ar5 = k[a1 * 64 + a4];
                float v2__ar5 = in3__ar5 * in4__ar5;
                acc2__ar5 += v2__ar5;
            }
            float v3__ar5 = acc2__ar5 * in0__ar5;
            float v5__ar5 = expf((v3__ar5 - acc1__ar5));
            acc3__ar5 += v5__ar5;
        }
        float v6__ar5 = 1.0f / acc3__ar5;
        float in0__ar6 = scaled_dot_product_attention_scale[0];
        float acc1__ar6 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar6 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar6 = q[(a0_b * 96 + a0_u * 12 + 6) % 256 * 64 + a2];
                float in1__ar6 = k[a1 * 64 + a2];
                float v0__ar6 = in1__ar6 * in2__ar6;
                acc0__ar6 += v0__ar6;
            }
            float v1__ar6 = acc0__ar6 * in0__ar6;
            acc1__ar6 = fmaxf(acc1__ar6, v1__ar6);
        }
        float acc3__ar6 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar6 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar6 = q[(a0_b * 96 + a0_u * 12 + 6) % 256 * 64 + a4];
                float in3__ar6 = k[a1 * 64 + a4];
                float v2__ar6 = in3__ar6 * in4__ar6;
                acc2__ar6 += v2__ar6;
            }
            float v3__ar6 = acc2__ar6 * in0__ar6;
            float v5__ar6 = expf((v3__ar6 - acc1__ar6));
            acc3__ar6 += v5__ar6;
        }
        float v6__ar6 = 1.0f / acc3__ar6;
        float in0__ar7 = scaled_dot_product_attention_scale[0];
        float acc1__ar7 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar7 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar7 = q[(a0_b * 96 + a0_u * 12 + 7) % 256 * 64 + a2];
                float in1__ar7 = k[a1 * 64 + a2];
                float v0__ar7 = in1__ar7 * in2__ar7;
                acc0__ar7 += v0__ar7;
            }
            float v1__ar7 = acc0__ar7 * in0__ar7;
            acc1__ar7 = fmaxf(acc1__ar7, v1__ar7);
        }
        float acc3__ar7 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar7 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar7 = q[(a0_b * 96 + a0_u * 12 + 7) % 256 * 64 + a4];
                float in3__ar7 = k[a1 * 64 + a4];
                float v2__ar7 = in3__ar7 * in4__ar7;
                acc2__ar7 += v2__ar7;
            }
            float v3__ar7 = acc2__ar7 * in0__ar7;
            float v5__ar7 = expf((v3__ar7 - acc1__ar7));
            acc3__ar7 += v5__ar7;
        }
        float v6__ar7 = 1.0f / acc3__ar7;
        float in0__ar8 = scaled_dot_product_attention_scale[0];
        float acc1__ar8 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar8 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar8 = q[(a0_b * 96 + a0_u * 12 + 8) % 256 * 64 + a2];
                float in1__ar8 = k[a1 * 64 + a2];
                float v0__ar8 = in1__ar8 * in2__ar8;
                acc0__ar8 += v0__ar8;
            }
            float v1__ar8 = acc0__ar8 * in0__ar8;
            acc1__ar8 = fmaxf(acc1__ar8, v1__ar8);
        }
        float acc3__ar8 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar8 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar8 = q[(a0_b * 96 + a0_u * 12 + 8) % 256 * 64 + a4];
                float in3__ar8 = k[a1 * 64 + a4];
                float v2__ar8 = in3__ar8 * in4__ar8;
                acc2__ar8 += v2__ar8;
            }
            float v3__ar8 = acc2__ar8 * in0__ar8;
            float v5__ar8 = expf((v3__ar8 - acc1__ar8));
            acc3__ar8 += v5__ar8;
        }
        float v6__ar8 = 1.0f / acc3__ar8;
        float in0__ar9 = scaled_dot_product_attention_scale[0];
        float acc1__ar9 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar9 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar9 = q[(a0_b * 96 + a0_u * 12 + 9) % 256 * 64 + a2];
                float in1__ar9 = k[a1 * 64 + a2];
                float v0__ar9 = in1__ar9 * in2__ar9;
                acc0__ar9 += v0__ar9;
            }
            float v1__ar9 = acc0__ar9 * in0__ar9;
            acc1__ar9 = fmaxf(acc1__ar9, v1__ar9);
        }
        float acc3__ar9 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar9 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar9 = q[(a0_b * 96 + a0_u * 12 + 9) % 256 * 64 + a4];
                float in3__ar9 = k[a1 * 64 + a4];
                float v2__ar9 = in3__ar9 * in4__ar9;
                acc2__ar9 += v2__ar9;
            }
            float v3__ar9 = acc2__ar9 * in0__ar9;
            float v5__ar9 = expf((v3__ar9 - acc1__ar9));
            acc3__ar9 += v5__ar9;
        }
        float v6__ar9 = 1.0f / acc3__ar9;
        float in0__ar10 = scaled_dot_product_attention_scale[0];
        float acc1__ar10 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar10 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar10 = q[(a0_b * 96 + a0_u * 12 + 10) % 256 * 64 + a2];
                float in1__ar10 = k[a1 * 64 + a2];
                float v0__ar10 = in1__ar10 * in2__ar10;
                acc0__ar10 += v0__ar10;
            }
            float v1__ar10 = acc0__ar10 * in0__ar10;
            acc1__ar10 = fmaxf(acc1__ar10, v1__ar10);
        }
        float acc3__ar10 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar10 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar10 = q[(a0_b * 96 + a0_u * 12 + 10) % 256 * 64 + a4];
                float in3__ar10 = k[a1 * 64 + a4];
                float v2__ar10 = in3__ar10 * in4__ar10;
                acc2__ar10 += v2__ar10;
            }
            float v3__ar10 = acc2__ar10 * in0__ar10;
            float v5__ar10 = expf((v3__ar10 - acc1__ar10));
            acc3__ar10 += v5__ar10;
        }
        float v6__ar10 = 1.0f / acc3__ar10;
        float in0__ar11 = scaled_dot_product_attention_scale[0];
        float acc1__ar11 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar11 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar11 = q[(a0_b * 96 + a0_u * 12 + 11) % 256 * 64 + a2];
                float in1__ar11 = k[a1 * 64 + a2];
                float v0__ar11 = in1__ar11 * in2__ar11;
                acc0__ar11 += v0__ar11;
            }
            float v1__ar11 = acc0__ar11 * in0__ar11;
            acc1__ar11 = fmaxf(acc1__ar11, v1__ar11);
        }
        float acc3__ar11 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc2__ar11 = 0.0f;
            for (int a4 = 0; a4 < 64; a4++) {
                float in4__ar11 = q[(a0_b * 96 + a0_u * 12 + 11) % 256 * 64 + a4];
                float in3__ar11 = k[a1 * 64 + a4];
                float v2__ar11 = in3__ar11 * in4__ar11;
                acc2__ar11 += v2__ar11;
            }
            float v3__ar11 = acc2__ar11 * in0__ar11;
            float v5__ar11 = expf((v3__ar11 - acc1__ar11));
            acc3__ar11 += v5__ar11;
        }
        float v6__ar11 = 1.0f / acc3__ar11;
        float acc5__c0_0 = 0.0f;
        float acc5__c0_1 = 0.0f;
        float acc5__c1_0 = 0.0f;
        float acc5__c1_1 = 0.0f;
        float acc5__c2_0 = 0.0f;
        float acc5__c2_1 = 0.0f;
        float acc5__c3_0 = 0.0f;
        float acc5__c3_1 = 0.0f;
        float acc5__c4_0 = 0.0f;
        float acc5__c4_1 = 0.0f;
        float acc5__c5_0 = 0.0f;
        float acc5__c5_1 = 0.0f;
        float acc5__c6_0 = 0.0f;
        float acc5__c6_1 = 0.0f;
        float acc5__c7_0 = 0.0f;
        float acc5__c7_1 = 0.0f;
        float acc5__c8_0 = 0.0f;
        float acc5__c8_1 = 0.0f;
        float acc5__c9_0 = 0.0f;
        float acc5__c9_1 = 0.0f;
        float acc5__c10_0 = 0.0f;
        float acc5__c10_1 = 0.0f;
        float acc5__c11_0 = 0.0f;
        float acc5__c11_1 = 0.0f;
        for (int a6 = 0; a6 < 128; a6++) {
            float acc4__ar0 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar0 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar0 = q[(a0_b * 96 + a0_u * 12) % 256 * 64 + a7];
                float v7__ar0 = in5__ar0 * in6__ar0;
                acc4__ar0 += v7__ar0;
            }
            float v8__ar0 = acc4__ar0 * in0__ar0;
            float v11__ar0 = expf((v8__ar0 - acc1__ar0)) * v6__ar0;
            float acc4__ar1 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar1 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar1 = q[(a0_b * 96 + a0_u * 12 + 1) % 256 * 64 + a7];
                float v7__ar1 = in5__ar1 * in6__ar1;
                acc4__ar1 += v7__ar1;
            }
            float v8__ar1 = acc4__ar1 * in0__ar1;
            float v11__ar1 = expf((v8__ar1 - acc1__ar1)) * v6__ar1;
            float acc4__ar2 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar2 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar2 = q[(a0_b * 96 + a0_u * 12 + 2) % 256 * 64 + a7];
                float v7__ar2 = in5__ar2 * in6__ar2;
                acc4__ar2 += v7__ar2;
            }
            float v8__ar2 = acc4__ar2 * in0__ar2;
            float v11__ar2 = expf((v8__ar2 - acc1__ar2)) * v6__ar2;
            float acc4__ar3 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar3 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar3 = q[(a0_b * 96 + a0_u * 12 + 3) % 256 * 64 + a7];
                float v7__ar3 = in5__ar3 * in6__ar3;
                acc4__ar3 += v7__ar3;
            }
            float v8__ar3 = acc4__ar3 * in0__ar3;
            float v11__ar3 = expf((v8__ar3 - acc1__ar3)) * v6__ar3;
            float acc4__ar4 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar4 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar4 = q[(a0_b * 96 + a0_u * 12 + 4) % 256 * 64 + a7];
                float v7__ar4 = in5__ar4 * in6__ar4;
                acc4__ar4 += v7__ar4;
            }
            float v8__ar4 = acc4__ar4 * in0__ar4;
            float v11__ar4 = expf((v8__ar4 - acc1__ar4)) * v6__ar4;
            float acc4__ar5 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar5 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar5 = q[(a0_b * 96 + a0_u * 12 + 5) % 256 * 64 + a7];
                float v7__ar5 = in5__ar5 * in6__ar5;
                acc4__ar5 += v7__ar5;
            }
            float v8__ar5 = acc4__ar5 * in0__ar5;
            float v11__ar5 = expf((v8__ar5 - acc1__ar5)) * v6__ar5;
            float acc4__ar6 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar6 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar6 = q[(a0_b * 96 + a0_u * 12 + 6) % 256 * 64 + a7];
                float v7__ar6 = in5__ar6 * in6__ar6;
                acc4__ar6 += v7__ar6;
            }
            float v8__ar6 = acc4__ar6 * in0__ar6;
            float v11__ar6 = expf((v8__ar6 - acc1__ar6)) * v6__ar6;
            float acc4__ar7 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar7 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar7 = q[(a0_b * 96 + a0_u * 12 + 7) % 256 * 64 + a7];
                float v7__ar7 = in5__ar7 * in6__ar7;
                acc4__ar7 += v7__ar7;
            }
            float v8__ar7 = acc4__ar7 * in0__ar7;
            float v11__ar7 = expf((v8__ar7 - acc1__ar7)) * v6__ar7;
            float acc4__ar8 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar8 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar8 = q[(a0_b * 96 + a0_u * 12 + 8) % 256 * 64 + a7];
                float v7__ar8 = in5__ar8 * in6__ar8;
                acc4__ar8 += v7__ar8;
            }
            float v8__ar8 = acc4__ar8 * in0__ar8;
            float v11__ar8 = expf((v8__ar8 - acc1__ar8)) * v6__ar8;
            float acc4__ar9 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar9 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar9 = q[(a0_b * 96 + a0_u * 12 + 9) % 256 * 64 + a7];
                float v7__ar9 = in5__ar9 * in6__ar9;
                acc4__ar9 += v7__ar9;
            }
            float v8__ar9 = acc4__ar9 * in0__ar9;
            float v11__ar9 = expf((v8__ar9 - acc1__ar9)) * v6__ar9;
            float acc4__ar10 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar10 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar10 = q[(a0_b * 96 + a0_u * 12 + 10) % 256 * 64 + a7];
                float v7__ar10 = in5__ar10 * in6__ar10;
                acc4__ar10 += v7__ar10;
            }
            float v8__ar10 = acc4__ar10 * in0__ar10;
            float v11__ar10 = expf((v8__ar10 - acc1__ar10)) * v6__ar10;
            float acc4__ar11 = 0.0f;
            for (int a7 = 0; a7 < 64; a7++) {
                float in5__ar11 = k[(_a6_ks * 128 + a6) * 64 + a7];
                float in6__ar11 = q[(a0_b * 96 + a0_u * 12 + 11) % 256 * 64 + a7];
                float v7__ar11 = in5__ar11 * in6__ar11;
                acc4__ar11 += v7__ar11;
            }
            float v8__ar11 = acc4__ar11 * in0__ar11;
            float v11__ar11 = expf((v8__ar11 - acc1__ar11)) * v6__ar11;
            float in7__bc0 = v[(_a6_ks * 128 + a6) * 64 + (a5_b * 32 + a5_u * 2)];
            float v12__c0_0 = in7__bc0 * v11__ar0;
            acc5__c0_0 += v12__c0_0;
            float in7__bc1 = v[(_a6_ks * 128 + a6) * 64 + (a5_b * 32 + a5_u * 2 + 1)];
            float v12__c0_1 = in7__bc1 * v11__ar0;
            acc5__c0_1 += v12__c0_1;
            float v12__c1_0 = in7__bc0 * v11__ar1;
            acc5__c1_0 += v12__c1_0;
            float v12__c1_1 = in7__bc1 * v11__ar1;
            acc5__c1_1 += v12__c1_1;
            float v12__c2_0 = in7__bc0 * v11__ar2;
            acc5__c2_0 += v12__c2_0;
            float v12__c2_1 = in7__bc1 * v11__ar2;
            acc5__c2_1 += v12__c2_1;
            float v12__c3_0 = in7__bc0 * v11__ar3;
            acc5__c3_0 += v12__c3_0;
            float v12__c3_1 = in7__bc1 * v11__ar3;
            acc5__c3_1 += v12__c3_1;
            float v12__c4_0 = in7__bc0 * v11__ar4;
            acc5__c4_0 += v12__c4_0;
            float v12__c4_1 = in7__bc1 * v11__ar4;
            acc5__c4_1 += v12__c4_1;
            float v12__c5_0 = in7__bc0 * v11__ar5;
            acc5__c5_0 += v12__c5_0;
            float v12__c5_1 = in7__bc1 * v11__ar5;
            acc5__c5_1 += v12__c5_1;
            float v12__c6_0 = in7__bc0 * v11__ar6;
            acc5__c6_0 += v12__c6_0;
            float v12__c6_1 = in7__bc1 * v11__ar6;
            acc5__c6_1 += v12__c6_1;
            float v12__c7_0 = in7__bc0 * v11__ar7;
            acc5__c7_0 += v12__c7_0;
            float v12__c7_1 = in7__bc1 * v11__ar7;
            acc5__c7_1 += v12__c7_1;
            float v12__c8_0 = in7__bc0 * v11__ar8;
            acc5__c8_0 += v12__c8_0;
            float v12__c8_1 = in7__bc1 * v11__ar8;
            acc5__c8_1 += v12__c8_1;
            float v12__c9_0 = in7__bc0 * v11__ar9;
            acc5__c9_0 += v12__c9_0;
            float v12__c9_1 = in7__bc1 * v11__ar9;
            acc5__c9_1 += v12__c9_1;
            float v12__c10_0 = in7__bc0 * v11__ar10;
            acc5__c10_0 += v12__c10_0;
            float v12__c10_1 = in7__bc1 * v11__ar10;
            acc5__c10_1 += v12__c10_1;
            float v12__c11_0 = in7__bc0 * v11__ar11;
            acc5__c11_0 += v12__c11_0;
            float v12__c11_1 = in7__bc1 * v11__ar11;
            acc5__c11_1 += v12__c11_1;
        }
        if (a0_b * 96 + a0_u * 12 + 0 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c0_0);
        }
        if (a0_b * 96 + a0_u * 12 + 0 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c0_1);
        }
        if (a0_b * 96 + a0_u * 12 + 1 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 1) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c1_0);
        }
        if (a0_b * 96 + a0_u * 12 + 1 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 1) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c1_1);
        }
        if (a0_b * 96 + a0_u * 12 + 2 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 2) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c2_0);
        }
        if (a0_b * 96 + a0_u * 12 + 2 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 2) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c2_1);
        }
        if (a0_b * 96 + a0_u * 12 + 3 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 3) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c3_0);
        }
        if (a0_b * 96 + a0_u * 12 + 3 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 3) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c3_1);
        }
        if (a0_b * 96 + a0_u * 12 + 4 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 4) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c4_0);
        }
        if (a0_b * 96 + a0_u * 12 + 4 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 4) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c4_1);
        }
        if (a0_b * 96 + a0_u * 12 + 5 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 5) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c5_0);
        }
        if (a0_b * 96 + a0_u * 12 + 5 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 5) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c5_1);
        }
        if (a0_b * 96 + a0_u * 12 + 6 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 6) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c6_0);
        }
        if (a0_b * 96 + a0_u * 12 + 6 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 6) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c6_1);
        }
        if (a0_b * 96 + a0_u * 12 + 7 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 7) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c7_0);
        }
        if (a0_b * 96 + a0_u * 12 + 7 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 7) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c7_1);
        }
        if (a0_b * 96 + a0_u * 12 + 8 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 8) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c8_0);
        }
        if (a0_b * 96 + a0_u * 12 + 8 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 8) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c8_1);
        }
        if (a0_b * 96 + a0_u * 12 + 9 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 9) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c9_0);
        }
        if (a0_b * 96 + a0_u * 12 + 9 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 9) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c9_1);
        }
        if (a0_b * 96 + a0_u * 12 + 10 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 10) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c10_0);
        }
        if (a0_b * 96 + a0_u * 12 + 10 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 10) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c10_1);
        }
        if (a0_b * 96 + a0_u * 12 + 11 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 11) % 256 * 64 + (a5_b * 32 + a5_u * 2)], acc5__c11_0);
        }
        if (a0_b * 96 + a0_u * 12 + 11 < 256) {
            atomicAdd(&scaled_dot_product_attention[(a0_b * 96 + a0_u * 12 + 11) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)], acc5__c11_1);
        }
}
