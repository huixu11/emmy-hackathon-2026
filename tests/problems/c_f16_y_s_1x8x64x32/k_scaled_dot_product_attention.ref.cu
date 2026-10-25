#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_scaled_dot_product_attention(const float* scaled_dot_product_attention__partial, __half* scaled_dot_product_attention) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 2048;
        int a1 = (_gid / 32) % 64;
        int a6 = (_gid) % 32;
        float acc3 = 0.0f;
        float acc5__sum = 0.0f;
        float acc1 = -1e+30f;
        for (int _a2_ks = 0; _a2_ks < 2; _a2_ks++) {
            float acc1__p = scaled_dot_product_attention__partial[_a2_ks * 16384 + a0 * 2048 + a1 * 32 + a6];
            float acc1__p__gn = fmaxf(acc1, acc1__p);
            float acc1__p__alpha = expf((acc1 - acc1__p__gn));
            float acc1__p__beta = expf((acc1__p - acc1__p__gn));
            float acc1__p__acc3_sa = acc1__p__alpha * acc3;
            float acc3__p = scaled_dot_product_attention__partial[32768 + _a2_ks * 16384 + a0 * 2048 + a1 * 32 + a6];
            float acc1__p__acc3_sb = acc1__p__beta * acc3__p;
            acc3 = acc1__p__acc3_sa + acc1__p__acc3_sb;
            float acc1__p__acc5__sum_sa = acc1__p__alpha * acc5__sum;
            float acc5__sum__p = scaled_dot_product_attention__partial[65536LL + _a2_ks * 16384 + a0 * 2048 + a1 * 32 + a6];
            float acc1__p__acc5__sum_sb = acc1__p__beta * acc5__sum__p;
            acc5__sum = acc1__p__acc5__sum_sa + acc1__p__acc5__sum_sb;
            acc1 = fmaxf(acc1, acc1__p);
        }
        float acc5 = acc5__sum * (1.0f / acc3);
        scaled_dot_product_attention[a0 * 2048 + a1 * 32 + a6] = __float2half(acc5);
}
