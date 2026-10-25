#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_386ea3(const float* matmul_1__place_834b9a8054_0, const __half* mul_c1, const __half* v, __half* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 352) {
        int a0_b = _gid / 22;
        int a5_b = (_gid) % 22;
        int a0_u = (_gid) % 1;
        int a5_u = (_gid) % 1;
        __half in0 = mul_c1[0];
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        __half acc1__c0_0 = __float2half(-1e+30f);
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        __half acc1__c0_1 = __float2half(-1e+30f);
        float acc3__c0_2 = 0.0f;
        float acc5__sum__c0_2 = 0.0f;
        __half acc1__c0_2 = __float2half(-1e+30f);
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        __half acc1__c1_0 = __float2half(-1e+30f);
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        __half acc1__c1_1 = __float2half(-1e+30f);
        float acc3__c1_2 = 0.0f;
        float acc5__sum__c1_2 = 0.0f;
        __half acc1__c1_2 = __float2half(-1e+30f);
        float acc3__c2_0 = 0.0f;
        float acc5__sum__c2_0 = 0.0f;
        __half acc1__c2_0 = __float2half(-1e+30f);
        float acc3__c2_1 = 0.0f;
        float acc5__sum__c2_1 = 0.0f;
        __half acc1__c2_1 = __float2half(-1e+30f);
        float acc3__c2_2 = 0.0f;
        float acc5__sum__c2_2 = 0.0f;
        __half acc1__c2_2 = __float2half(-1e+30f);
        float acc3__c3_0 = 0.0f;
        float acc5__sum__c3_0 = 0.0f;
        __half acc1__c3_0 = __float2half(-1e+30f);
        float acc3__c3_1 = 0.0f;
        float acc5__sum__c3_1 = 0.0f;
        __half acc1__c3_1 = __float2half(-1e+30f);
        float acc3__c3_2 = 0.0f;
        float acc5__sum__c3_2 = 0.0f;
        __half acc1__c3_2 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ws834b9a8054__ar0 = matmul_1__place_834b9a8054_0[(a0_b * 4 + a0_u * 4) * 256 + a1];
            __half v1__ws834b9a8054__c0_0 = __float2half(acc0__ws834b9a8054__ar0);
            __half v2__ws834b9a8054__c0_0 = in0 * v1__ws834b9a8054__c0_0;
            float one__c0_0 = 1.0f;
            float v2__ws834b9a8054__gn__c0_0 = fmaxf(__half2float(acc1__c0_0), __half2float(v2__ws834b9a8054__c0_0));
            float v2__ws834b9a8054__dg__c0_0 = __half2float(acc1__c0_0) - v2__ws834b9a8054__gn__c0_0;
            float v2__ws834b9a8054__alpha__c0_0 = expf(v2__ws834b9a8054__dg__c0_0);
            float v2__ws834b9a8054__dg_o__c0_0 = __half2float(v2__ws834b9a8054__c0_0) - v2__ws834b9a8054__gn__c0_0;
            float v2__ws834b9a8054__beta__c0_0 = expf(v2__ws834b9a8054__dg_o__c0_0);
            float v2__ws834b9a8054__acc3_sa__c0_0 = acc3__c0_0 * v2__ws834b9a8054__alpha__c0_0;
            float v2__ws834b9a8054__acc3_sb__c0_0 = one__c0_0 * v2__ws834b9a8054__beta__c0_0;
            acc3__c0_0 = v2__ws834b9a8054__acc3_sa__c0_0 + v2__ws834b9a8054__acc3_sb__c0_0;
            float v2__ws834b9a8054__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v2__ws834b9a8054__alpha__c0_0;
            __half in7__bc0 = v[a1 * 64 + (a5_b * 3 + a5_u * 3) % 64];
            float v2__ws834b9a8054__acc5__sum_sb__c0_0 = __half2float(in7__bc0) * v2__ws834b9a8054__beta__c0_0;
            acc5__sum__c0_0 = v2__ws834b9a8054__acc5__sum_sa__c0_0 + v2__ws834b9a8054__acc5__sum_sb__c0_0;
            acc1__c0_0 = __hmax(acc1__c0_0, v2__ws834b9a8054__c0_0);
            __half v1__ws834b9a8054__c0_1 = __float2half(acc0__ws834b9a8054__ar0);
            __half v2__ws834b9a8054__c0_1 = in0 * v1__ws834b9a8054__c0_1;
            float one__c0_1 = 1.0f;
            float v2__ws834b9a8054__gn__c0_1 = fmaxf(__half2float(acc1__c0_1), __half2float(v2__ws834b9a8054__c0_1));
            float v2__ws834b9a8054__dg__c0_1 = __half2float(acc1__c0_1) - v2__ws834b9a8054__gn__c0_1;
            float v2__ws834b9a8054__alpha__c0_1 = expf(v2__ws834b9a8054__dg__c0_1);
            float v2__ws834b9a8054__dg_o__c0_1 = __half2float(v2__ws834b9a8054__c0_1) - v2__ws834b9a8054__gn__c0_1;
            float v2__ws834b9a8054__beta__c0_1 = expf(v2__ws834b9a8054__dg_o__c0_1);
            float v2__ws834b9a8054__acc3_sa__c0_1 = acc3__c0_1 * v2__ws834b9a8054__alpha__c0_1;
            float v2__ws834b9a8054__acc3_sb__c0_1 = one__c0_1 * v2__ws834b9a8054__beta__c0_1;
            acc3__c0_1 = v2__ws834b9a8054__acc3_sa__c0_1 + v2__ws834b9a8054__acc3_sb__c0_1;
            float v2__ws834b9a8054__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v2__ws834b9a8054__alpha__c0_1;
            __half in7__bc1 = v[a1 * 64 + (a5_b * 3 + a5_u * 3 + 1) % 64];
            float v2__ws834b9a8054__acc5__sum_sb__c0_1 = __half2float(in7__bc1) * v2__ws834b9a8054__beta__c0_1;
            acc5__sum__c0_1 = v2__ws834b9a8054__acc5__sum_sa__c0_1 + v2__ws834b9a8054__acc5__sum_sb__c0_1;
            acc1__c0_1 = __hmax(acc1__c0_1, v2__ws834b9a8054__c0_1);
            __half v1__ws834b9a8054__c0_2 = __float2half(acc0__ws834b9a8054__ar0);
            __half v2__ws834b9a8054__c0_2 = in0 * v1__ws834b9a8054__c0_2;
            float one__c0_2 = 1.0f;
            float v2__ws834b9a8054__gn__c0_2 = fmaxf(__half2float(acc1__c0_2), __half2float(v2__ws834b9a8054__c0_2));
            float v2__ws834b9a8054__dg__c0_2 = __half2float(acc1__c0_2) - v2__ws834b9a8054__gn__c0_2;
            float v2__ws834b9a8054__alpha__c0_2 = expf(v2__ws834b9a8054__dg__c0_2);
            float v2__ws834b9a8054__dg_o__c0_2 = __half2float(v2__ws834b9a8054__c0_2) - v2__ws834b9a8054__gn__c0_2;
            float v2__ws834b9a8054__beta__c0_2 = expf(v2__ws834b9a8054__dg_o__c0_2);
            float v2__ws834b9a8054__acc3_sa__c0_2 = acc3__c0_2 * v2__ws834b9a8054__alpha__c0_2;
            float v2__ws834b9a8054__acc3_sb__c0_2 = one__c0_2 * v2__ws834b9a8054__beta__c0_2;
            acc3__c0_2 = v2__ws834b9a8054__acc3_sa__c0_2 + v2__ws834b9a8054__acc3_sb__c0_2;
            float v2__ws834b9a8054__acc5__sum_sa__c0_2 = acc5__sum__c0_2 * v2__ws834b9a8054__alpha__c0_2;
            __half in7__bc2 = v[a1 * 64 + (a5_b * 3 + a5_u * 3 + 2) % 64];
            float v2__ws834b9a8054__acc5__sum_sb__c0_2 = __half2float(in7__bc2) * v2__ws834b9a8054__beta__c0_2;
            acc5__sum__c0_2 = v2__ws834b9a8054__acc5__sum_sa__c0_2 + v2__ws834b9a8054__acc5__sum_sb__c0_2;
            acc1__c0_2 = __hmax(acc1__c0_2, v2__ws834b9a8054__c0_2);
            float acc0__ws834b9a8054__ar1 = matmul_1__place_834b9a8054_0[(a0_b * 4 + a0_u * 4 + 1) * 256 + a1];
            __half v1__ws834b9a8054__c1_0 = __float2half(acc0__ws834b9a8054__ar1);
            __half v2__ws834b9a8054__c1_0 = in0 * v1__ws834b9a8054__c1_0;
            float one__c1_0 = 1.0f;
            float v2__ws834b9a8054__gn__c1_0 = fmaxf(__half2float(acc1__c1_0), __half2float(v2__ws834b9a8054__c1_0));
            float v2__ws834b9a8054__dg__c1_0 = __half2float(acc1__c1_0) - v2__ws834b9a8054__gn__c1_0;
            float v2__ws834b9a8054__alpha__c1_0 = expf(v2__ws834b9a8054__dg__c1_0);
            float v2__ws834b9a8054__dg_o__c1_0 = __half2float(v2__ws834b9a8054__c1_0) - v2__ws834b9a8054__gn__c1_0;
            float v2__ws834b9a8054__beta__c1_0 = expf(v2__ws834b9a8054__dg_o__c1_0);
            float v2__ws834b9a8054__acc3_sa__c1_0 = acc3__c1_0 * v2__ws834b9a8054__alpha__c1_0;
            float v2__ws834b9a8054__acc3_sb__c1_0 = one__c1_0 * v2__ws834b9a8054__beta__c1_0;
            acc3__c1_0 = v2__ws834b9a8054__acc3_sa__c1_0 + v2__ws834b9a8054__acc3_sb__c1_0;
            float v2__ws834b9a8054__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v2__ws834b9a8054__alpha__c1_0;
            float v2__ws834b9a8054__acc5__sum_sb__c1_0 = __half2float(in7__bc0) * v2__ws834b9a8054__beta__c1_0;
            acc5__sum__c1_0 = v2__ws834b9a8054__acc5__sum_sa__c1_0 + v2__ws834b9a8054__acc5__sum_sb__c1_0;
            acc1__c1_0 = __hmax(acc1__c1_0, v2__ws834b9a8054__c1_0);
            __half v1__ws834b9a8054__c1_1 = __float2half(acc0__ws834b9a8054__ar1);
            __half v2__ws834b9a8054__c1_1 = in0 * v1__ws834b9a8054__c1_1;
            float one__c1_1 = 1.0f;
            float v2__ws834b9a8054__gn__c1_1 = fmaxf(__half2float(acc1__c1_1), __half2float(v2__ws834b9a8054__c1_1));
            float v2__ws834b9a8054__dg__c1_1 = __half2float(acc1__c1_1) - v2__ws834b9a8054__gn__c1_1;
            float v2__ws834b9a8054__alpha__c1_1 = expf(v2__ws834b9a8054__dg__c1_1);
            float v2__ws834b9a8054__dg_o__c1_1 = __half2float(v2__ws834b9a8054__c1_1) - v2__ws834b9a8054__gn__c1_1;
            float v2__ws834b9a8054__beta__c1_1 = expf(v2__ws834b9a8054__dg_o__c1_1);
            float v2__ws834b9a8054__acc3_sa__c1_1 = acc3__c1_1 * v2__ws834b9a8054__alpha__c1_1;
            float v2__ws834b9a8054__acc3_sb__c1_1 = one__c1_1 * v2__ws834b9a8054__beta__c1_1;
            acc3__c1_1 = v2__ws834b9a8054__acc3_sa__c1_1 + v2__ws834b9a8054__acc3_sb__c1_1;
            float v2__ws834b9a8054__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v2__ws834b9a8054__alpha__c1_1;
            float v2__ws834b9a8054__acc5__sum_sb__c1_1 = __half2float(in7__bc1) * v2__ws834b9a8054__beta__c1_1;
            acc5__sum__c1_1 = v2__ws834b9a8054__acc5__sum_sa__c1_1 + v2__ws834b9a8054__acc5__sum_sb__c1_1;
            acc1__c1_1 = __hmax(acc1__c1_1, v2__ws834b9a8054__c1_1);
            __half v1__ws834b9a8054__c1_2 = __float2half(acc0__ws834b9a8054__ar1);
            __half v2__ws834b9a8054__c1_2 = in0 * v1__ws834b9a8054__c1_2;
            float one__c1_2 = 1.0f;
            float v2__ws834b9a8054__gn__c1_2 = fmaxf(__half2float(acc1__c1_2), __half2float(v2__ws834b9a8054__c1_2));
            float v2__ws834b9a8054__dg__c1_2 = __half2float(acc1__c1_2) - v2__ws834b9a8054__gn__c1_2;
            float v2__ws834b9a8054__alpha__c1_2 = expf(v2__ws834b9a8054__dg__c1_2);
            float v2__ws834b9a8054__dg_o__c1_2 = __half2float(v2__ws834b9a8054__c1_2) - v2__ws834b9a8054__gn__c1_2;
            float v2__ws834b9a8054__beta__c1_2 = expf(v2__ws834b9a8054__dg_o__c1_2);
            float v2__ws834b9a8054__acc3_sa__c1_2 = acc3__c1_2 * v2__ws834b9a8054__alpha__c1_2;
            float v2__ws834b9a8054__acc3_sb__c1_2 = one__c1_2 * v2__ws834b9a8054__beta__c1_2;
            acc3__c1_2 = v2__ws834b9a8054__acc3_sa__c1_2 + v2__ws834b9a8054__acc3_sb__c1_2;
            float v2__ws834b9a8054__acc5__sum_sa__c1_2 = acc5__sum__c1_2 * v2__ws834b9a8054__alpha__c1_2;
            float v2__ws834b9a8054__acc5__sum_sb__c1_2 = __half2float(in7__bc2) * v2__ws834b9a8054__beta__c1_2;
            acc5__sum__c1_2 = v2__ws834b9a8054__acc5__sum_sa__c1_2 + v2__ws834b9a8054__acc5__sum_sb__c1_2;
            acc1__c1_2 = __hmax(acc1__c1_2, v2__ws834b9a8054__c1_2);
            float acc0__ws834b9a8054__ar2 = matmul_1__place_834b9a8054_0[(a0_b * 4 + a0_u * 4 + 2) * 256 + a1];
            __half v1__ws834b9a8054__c2_0 = __float2half(acc0__ws834b9a8054__ar2);
            __half v2__ws834b9a8054__c2_0 = in0 * v1__ws834b9a8054__c2_0;
            float one__c2_0 = 1.0f;
            float v2__ws834b9a8054__gn__c2_0 = fmaxf(__half2float(acc1__c2_0), __half2float(v2__ws834b9a8054__c2_0));
            float v2__ws834b9a8054__dg__c2_0 = __half2float(acc1__c2_0) - v2__ws834b9a8054__gn__c2_0;
            float v2__ws834b9a8054__alpha__c2_0 = expf(v2__ws834b9a8054__dg__c2_0);
            float v2__ws834b9a8054__dg_o__c2_0 = __half2float(v2__ws834b9a8054__c2_0) - v2__ws834b9a8054__gn__c2_0;
            float v2__ws834b9a8054__beta__c2_0 = expf(v2__ws834b9a8054__dg_o__c2_0);
            float v2__ws834b9a8054__acc3_sa__c2_0 = acc3__c2_0 * v2__ws834b9a8054__alpha__c2_0;
            float v2__ws834b9a8054__acc3_sb__c2_0 = one__c2_0 * v2__ws834b9a8054__beta__c2_0;
            acc3__c2_0 = v2__ws834b9a8054__acc3_sa__c2_0 + v2__ws834b9a8054__acc3_sb__c2_0;
            float v2__ws834b9a8054__acc5__sum_sa__c2_0 = acc5__sum__c2_0 * v2__ws834b9a8054__alpha__c2_0;
            float v2__ws834b9a8054__acc5__sum_sb__c2_0 = __half2float(in7__bc0) * v2__ws834b9a8054__beta__c2_0;
            acc5__sum__c2_0 = v2__ws834b9a8054__acc5__sum_sa__c2_0 + v2__ws834b9a8054__acc5__sum_sb__c2_0;
            acc1__c2_0 = __hmax(acc1__c2_0, v2__ws834b9a8054__c2_0);
            __half v1__ws834b9a8054__c2_1 = __float2half(acc0__ws834b9a8054__ar2);
            __half v2__ws834b9a8054__c2_1 = in0 * v1__ws834b9a8054__c2_1;
            float one__c2_1 = 1.0f;
            float v2__ws834b9a8054__gn__c2_1 = fmaxf(__half2float(acc1__c2_1), __half2float(v2__ws834b9a8054__c2_1));
            float v2__ws834b9a8054__dg__c2_1 = __half2float(acc1__c2_1) - v2__ws834b9a8054__gn__c2_1;
            float v2__ws834b9a8054__alpha__c2_1 = expf(v2__ws834b9a8054__dg__c2_1);
            float v2__ws834b9a8054__dg_o__c2_1 = __half2float(v2__ws834b9a8054__c2_1) - v2__ws834b9a8054__gn__c2_1;
            float v2__ws834b9a8054__beta__c2_1 = expf(v2__ws834b9a8054__dg_o__c2_1);
            float v2__ws834b9a8054__acc3_sa__c2_1 = acc3__c2_1 * v2__ws834b9a8054__alpha__c2_1;
            float v2__ws834b9a8054__acc3_sb__c2_1 = one__c2_1 * v2__ws834b9a8054__beta__c2_1;
            acc3__c2_1 = v2__ws834b9a8054__acc3_sa__c2_1 + v2__ws834b9a8054__acc3_sb__c2_1;
            float v2__ws834b9a8054__acc5__sum_sa__c2_1 = acc5__sum__c2_1 * v2__ws834b9a8054__alpha__c2_1;
            float v2__ws834b9a8054__acc5__sum_sb__c2_1 = __half2float(in7__bc1) * v2__ws834b9a8054__beta__c2_1;
            acc5__sum__c2_1 = v2__ws834b9a8054__acc5__sum_sa__c2_1 + v2__ws834b9a8054__acc5__sum_sb__c2_1;
            acc1__c2_1 = __hmax(acc1__c2_1, v2__ws834b9a8054__c2_1);
            __half v1__ws834b9a8054__c2_2 = __float2half(acc0__ws834b9a8054__ar2);
            __half v2__ws834b9a8054__c2_2 = in0 * v1__ws834b9a8054__c2_2;
            float one__c2_2 = 1.0f;
            float v2__ws834b9a8054__gn__c2_2 = fmaxf(__half2float(acc1__c2_2), __half2float(v2__ws834b9a8054__c2_2));
            float v2__ws834b9a8054__dg__c2_2 = __half2float(acc1__c2_2) - v2__ws834b9a8054__gn__c2_2;
            float v2__ws834b9a8054__alpha__c2_2 = expf(v2__ws834b9a8054__dg__c2_2);
            float v2__ws834b9a8054__dg_o__c2_2 = __half2float(v2__ws834b9a8054__c2_2) - v2__ws834b9a8054__gn__c2_2;
            float v2__ws834b9a8054__beta__c2_2 = expf(v2__ws834b9a8054__dg_o__c2_2);
            float v2__ws834b9a8054__acc3_sa__c2_2 = acc3__c2_2 * v2__ws834b9a8054__alpha__c2_2;
            float v2__ws834b9a8054__acc3_sb__c2_2 = one__c2_2 * v2__ws834b9a8054__beta__c2_2;
            acc3__c2_2 = v2__ws834b9a8054__acc3_sa__c2_2 + v2__ws834b9a8054__acc3_sb__c2_2;
            float v2__ws834b9a8054__acc5__sum_sa__c2_2 = acc5__sum__c2_2 * v2__ws834b9a8054__alpha__c2_2;
            float v2__ws834b9a8054__acc5__sum_sb__c2_2 = __half2float(in7__bc2) * v2__ws834b9a8054__beta__c2_2;
            acc5__sum__c2_2 = v2__ws834b9a8054__acc5__sum_sa__c2_2 + v2__ws834b9a8054__acc5__sum_sb__c2_2;
            acc1__c2_2 = __hmax(acc1__c2_2, v2__ws834b9a8054__c2_2);
            float acc0__ws834b9a8054__ar3 = matmul_1__place_834b9a8054_0[(a0_b * 4 + a0_u * 4 + 3) * 256 + a1];
            __half v1__ws834b9a8054__c3_0 = __float2half(acc0__ws834b9a8054__ar3);
            __half v2__ws834b9a8054__c3_0 = in0 * v1__ws834b9a8054__c3_0;
            float one__c3_0 = 1.0f;
            float v2__ws834b9a8054__gn__c3_0 = fmaxf(__half2float(acc1__c3_0), __half2float(v2__ws834b9a8054__c3_0));
            float v2__ws834b9a8054__dg__c3_0 = __half2float(acc1__c3_0) - v2__ws834b9a8054__gn__c3_0;
            float v2__ws834b9a8054__alpha__c3_0 = expf(v2__ws834b9a8054__dg__c3_0);
            float v2__ws834b9a8054__dg_o__c3_0 = __half2float(v2__ws834b9a8054__c3_0) - v2__ws834b9a8054__gn__c3_0;
            float v2__ws834b9a8054__beta__c3_0 = expf(v2__ws834b9a8054__dg_o__c3_0);
            float v2__ws834b9a8054__acc3_sa__c3_0 = acc3__c3_0 * v2__ws834b9a8054__alpha__c3_0;
            float v2__ws834b9a8054__acc3_sb__c3_0 = one__c3_0 * v2__ws834b9a8054__beta__c3_0;
            acc3__c3_0 = v2__ws834b9a8054__acc3_sa__c3_0 + v2__ws834b9a8054__acc3_sb__c3_0;
            float v2__ws834b9a8054__acc5__sum_sa__c3_0 = acc5__sum__c3_0 * v2__ws834b9a8054__alpha__c3_0;
            float v2__ws834b9a8054__acc5__sum_sb__c3_0 = __half2float(in7__bc0) * v2__ws834b9a8054__beta__c3_0;
            acc5__sum__c3_0 = v2__ws834b9a8054__acc5__sum_sa__c3_0 + v2__ws834b9a8054__acc5__sum_sb__c3_0;
            acc1__c3_0 = __hmax(acc1__c3_0, v2__ws834b9a8054__c3_0);
            __half v1__ws834b9a8054__c3_1 = __float2half(acc0__ws834b9a8054__ar3);
            __half v2__ws834b9a8054__c3_1 = in0 * v1__ws834b9a8054__c3_1;
            float one__c3_1 = 1.0f;
            float v2__ws834b9a8054__gn__c3_1 = fmaxf(__half2float(acc1__c3_1), __half2float(v2__ws834b9a8054__c3_1));
            float v2__ws834b9a8054__dg__c3_1 = __half2float(acc1__c3_1) - v2__ws834b9a8054__gn__c3_1;
            float v2__ws834b9a8054__alpha__c3_1 = expf(v2__ws834b9a8054__dg__c3_1);
            float v2__ws834b9a8054__dg_o__c3_1 = __half2float(v2__ws834b9a8054__c3_1) - v2__ws834b9a8054__gn__c3_1;
            float v2__ws834b9a8054__beta__c3_1 = expf(v2__ws834b9a8054__dg_o__c3_1);
            float v2__ws834b9a8054__acc3_sa__c3_1 = acc3__c3_1 * v2__ws834b9a8054__alpha__c3_1;
            float v2__ws834b9a8054__acc3_sb__c3_1 = one__c3_1 * v2__ws834b9a8054__beta__c3_1;
            acc3__c3_1 = v2__ws834b9a8054__acc3_sa__c3_1 + v2__ws834b9a8054__acc3_sb__c3_1;
            float v2__ws834b9a8054__acc5__sum_sa__c3_1 = acc5__sum__c3_1 * v2__ws834b9a8054__alpha__c3_1;
            float v2__ws834b9a8054__acc5__sum_sb__c3_1 = __half2float(in7__bc1) * v2__ws834b9a8054__beta__c3_1;
            acc5__sum__c3_1 = v2__ws834b9a8054__acc5__sum_sa__c3_1 + v2__ws834b9a8054__acc5__sum_sb__c3_1;
            acc1__c3_1 = __hmax(acc1__c3_1, v2__ws834b9a8054__c3_1);
            __half v1__ws834b9a8054__c3_2 = __float2half(acc0__ws834b9a8054__ar3);
            __half v2__ws834b9a8054__c3_2 = in0 * v1__ws834b9a8054__c3_2;
            float one__c3_2 = 1.0f;
            float v2__ws834b9a8054__gn__c3_2 = fmaxf(__half2float(acc1__c3_2), __half2float(v2__ws834b9a8054__c3_2));
            float v2__ws834b9a8054__dg__c3_2 = __half2float(acc1__c3_2) - v2__ws834b9a8054__gn__c3_2;
            float v2__ws834b9a8054__alpha__c3_2 = expf(v2__ws834b9a8054__dg__c3_2);
            float v2__ws834b9a8054__dg_o__c3_2 = __half2float(v2__ws834b9a8054__c3_2) - v2__ws834b9a8054__gn__c3_2;
            float v2__ws834b9a8054__beta__c3_2 = expf(v2__ws834b9a8054__dg_o__c3_2);
            float v2__ws834b9a8054__acc3_sa__c3_2 = acc3__c3_2 * v2__ws834b9a8054__alpha__c3_2;
            float v2__ws834b9a8054__acc3_sb__c3_2 = one__c3_2 * v2__ws834b9a8054__beta__c3_2;
            acc3__c3_2 = v2__ws834b9a8054__acc3_sa__c3_2 + v2__ws834b9a8054__acc3_sb__c3_2;
            float v2__ws834b9a8054__acc5__sum_sa__c3_2 = acc5__sum__c3_2 * v2__ws834b9a8054__alpha__c3_2;
            float v2__ws834b9a8054__acc5__sum_sb__c3_2 = __half2float(in7__bc2) * v2__ws834b9a8054__beta__c3_2;
            acc5__sum__c3_2 = v2__ws834b9a8054__acc5__sum_sa__c3_2 + v2__ws834b9a8054__acc5__sum_sb__c3_2;
            acc1__c3_2 = __hmax(acc1__c3_2, v2__ws834b9a8054__c3_2);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a5_b * 3 + a5_u * 3 + 0 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4) * 64 + (a5_b * 3 + a5_u * 3) % 64] = __float2half(acc5__c0_0);
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a5_b * 3 + a5_u * 3 + 1 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4) * 64 + (a5_b * 3 + a5_u * 3 + 1) % 64] = __float2half(acc5__c0_1);
        }
        float acc5__c0_2 = acc5__sum__c0_2 * (1.0f / acc3__c0_2);
        if (a5_b * 3 + a5_u * 3 + 2 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4) * 64 + (a5_b * 3 + a5_u * 3 + 2) % 64] = __float2half(acc5__c0_2);
        }
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        if (a5_b * 3 + a5_u * 3 + 0 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) * 64 + (a5_b * 3 + a5_u * 3) % 64] = __float2half(acc5__c1_0);
        }
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        if (a5_b * 3 + a5_u * 3 + 1 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) * 64 + (a5_b * 3 + a5_u * 3 + 1) % 64] = __float2half(acc5__c1_1);
        }
        float acc5__c1_2 = acc5__sum__c1_2 * (1.0f / acc3__c1_2);
        if (a5_b * 3 + a5_u * 3 + 2 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) * 64 + (a5_b * 3 + a5_u * 3 + 2) % 64] = __float2half(acc5__c1_2);
        }
        float acc5__c2_0 = acc5__sum__c2_0 * (1.0f / acc3__c2_0);
        if (a5_b * 3 + a5_u * 3 + 0 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) * 64 + (a5_b * 3 + a5_u * 3) % 64] = __float2half(acc5__c2_0);
        }
        float acc5__c2_1 = acc5__sum__c2_1 * (1.0f / acc3__c2_1);
        if (a5_b * 3 + a5_u * 3 + 1 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) * 64 + (a5_b * 3 + a5_u * 3 + 1) % 64] = __float2half(acc5__c2_1);
        }
        float acc5__c2_2 = acc5__sum__c2_2 * (1.0f / acc3__c2_2);
        if (a5_b * 3 + a5_u * 3 + 2 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) * 64 + (a5_b * 3 + a5_u * 3 + 2) % 64] = __float2half(acc5__c2_2);
        }
        float acc5__c3_0 = acc5__sum__c3_0 * (1.0f / acc3__c3_0);
        if (a5_b * 3 + a5_u * 3 + 0 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) * 64 + (a5_b * 3 + a5_u * 3) % 64] = __float2half(acc5__c3_0);
        }
        float acc5__c3_1 = acc5__sum__c3_1 * (1.0f / acc3__c3_1);
        if (a5_b * 3 + a5_u * 3 + 1 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) * 64 + (a5_b * 3 + a5_u * 3 + 1) % 64] = __float2half(acc5__c3_1);
        }
        float acc5__c3_2 = acc5__sum__c3_2 * (1.0f / acc3__c3_2);
        if (a5_b * 3 + a5_u * 3 + 2 < 64) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) * 64 + (a5_b * 3 + a5_u * 3 + 2) % 64] = __float2half(acc5__c3_2);
        }
    }
}
