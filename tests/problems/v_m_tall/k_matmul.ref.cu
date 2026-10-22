static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(256) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 5120;
        int a0_b = (_gid / 512) % 10;
        int a1_b = (_gid / 256) % 2;
        int a0_u = (_gid / 32) % 8;
        int a1_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c4_1 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c5_1 = 0.0f;
        float acc0__c6_0 = 0.0f;
        float acc0__c6_1 = 0.0f;
        float acc0__c7_0 = 0.0f;
        float acc0__c7_1 = 0.0f;
        float acc0__c8_0 = 0.0f;
        float acc0__c8_1 = 0.0f;
        float acc0__c9_0 = 0.0f;
        float acc0__c9_1 = 0.0f;
        float acc0__c10_0 = 0.0f;
        float acc0__c10_1 = 0.0f;
        float acc0__c11_0 = 0.0f;
        float acc0__c11_1 = 0.0f;
        float acc0__c12_0 = 0.0f;
        float acc0__c12_1 = 0.0f;
        float acc0__c13_0 = 0.0f;
        float acc0__c13_1 = 0.0f;
        __shared__ __align__(16) float _a_smem[448];
        __shared__ __align__(16) float _b_smem[256];
        for (int _ks = 0; _ks < 4; _ks += 4) {
            for (int _fa = a0_u * 32 + a1_u; _fa < 112; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[((a0_b * 112 + _fa < 1024) ? (a0_b * 112 + _fa) : (1023)) * 32 + (_a2_ks * 4 + _ks)]);
            }
            for (int _fb = a0_u * 32 + a1_u; _fb < 64; _fb += 256) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[(_a2_ks * 4 + (_ks + _fb / 16)) * 128 + (a1_b * 64 + 4 * (_fb % 16))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 4; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u * 2 - a1_b * 64)];
                float in0__c0_0 = _a_smem[(a0_b * 112 + a0_u * 14 - a0_b * 112) * 4 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u * 2 + 1 - a1_b * 64)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in0__c1_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 1 - a0_b * 112) * 4 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                float in0__c2_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 2 - a0_b * 112) * 4 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float acc0__v__c2_1 = in1__c0_1 * in0__c2_0;
                acc0__c2_1 += acc0__v__c2_1;
                float in0__c3_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 3 - a0_b * 112) * 4 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float acc0__v__c3_1 = in1__c0_1 * in0__c3_0;
                acc0__c3_1 += acc0__v__c3_1;
                float in0__c4_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 4 - a0_b * 112) * 4 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float acc0__v__c4_1 = in1__c0_1 * in0__c4_0;
                acc0__c4_1 += acc0__v__c4_1;
                float in0__c5_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 5 - a0_b * 112) * 4 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float acc0__v__c5_1 = in1__c0_1 * in0__c5_0;
                acc0__c5_1 += acc0__v__c5_1;
                float in0__c6_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 6 - a0_b * 112) * 4 + _ki];
                float acc0__v__c6_0 = in1__c0_0 * in0__c6_0;
                acc0__c6_0 += acc0__v__c6_0;
                float acc0__v__c6_1 = in1__c0_1 * in0__c6_0;
                acc0__c6_1 += acc0__v__c6_1;
                float in0__c7_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 7 - a0_b * 112) * 4 + _ki];
                float acc0__v__c7_0 = in1__c0_0 * in0__c7_0;
                acc0__c7_0 += acc0__v__c7_0;
                float acc0__v__c7_1 = in1__c0_1 * in0__c7_0;
                acc0__c7_1 += acc0__v__c7_1;
                float in0__c8_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 8 - a0_b * 112) * 4 + _ki];
                float acc0__v__c8_0 = in1__c0_0 * in0__c8_0;
                acc0__c8_0 += acc0__v__c8_0;
                float acc0__v__c8_1 = in1__c0_1 * in0__c8_0;
                acc0__c8_1 += acc0__v__c8_1;
                float in0__c9_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 9 - a0_b * 112) * 4 + _ki];
                float acc0__v__c9_0 = in1__c0_0 * in0__c9_0;
                acc0__c9_0 += acc0__v__c9_0;
                float acc0__v__c9_1 = in1__c0_1 * in0__c9_0;
                acc0__c9_1 += acc0__v__c9_1;
                float in0__c10_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 10 - a0_b * 112) * 4 + _ki];
                float acc0__v__c10_0 = in1__c0_0 * in0__c10_0;
                acc0__c10_0 += acc0__v__c10_0;
                float acc0__v__c10_1 = in1__c0_1 * in0__c10_0;
                acc0__c10_1 += acc0__v__c10_1;
                float in0__c11_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 11 - a0_b * 112) * 4 + _ki];
                float acc0__v__c11_0 = in1__c0_0 * in0__c11_0;
                acc0__c11_0 += acc0__v__c11_0;
                float acc0__v__c11_1 = in1__c0_1 * in0__c11_0;
                acc0__c11_1 += acc0__v__c11_1;
                float in0__c12_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 12 - a0_b * 112) * 4 + _ki];
                float acc0__v__c12_0 = in1__c0_0 * in0__c12_0;
                acc0__c12_0 += acc0__v__c12_0;
                float acc0__v__c12_1 = in1__c0_1 * in0__c12_0;
                acc0__c12_1 += acc0__v__c12_1;
                float in0__c13_0 = _a_smem[(a0_b * 112 + a0_u * 14 + 13 - a0_b * 112) * 4 + _ki];
                float acc0__v__c13_0 = in1__c0_0 * in0__c13_0;
                acc0__c13_0 += acc0__v__c13_0;
                float acc0__v__c13_1 = in1__c0_1 * in0__c13_0;
                acc0__c13_1 += acc0__v__c13_1;
            }
            __syncthreads();
        }
        if (a0_b * 112 + a0_u * 14 + 0 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c0_0);
        }
        if (a0_b * 112 + a0_u * 14 + 0 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c0_1);
        }
        if (a0_b * 112 + a0_u * 14 + 1 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 1) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c1_0);
        }
        if (a0_b * 112 + a0_u * 14 + 1 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 1) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c1_1);
        }
        if (a0_b * 112 + a0_u * 14 + 2 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 2) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c2_0);
        }
        if (a0_b * 112 + a0_u * 14 + 2 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 2) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c2_1);
        }
        if (a0_b * 112 + a0_u * 14 + 3 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 3) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c3_0);
        }
        if (a0_b * 112 + a0_u * 14 + 3 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 3) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c3_1);
        }
        if (a0_b * 112 + a0_u * 14 + 4 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 4) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c4_0);
        }
        if (a0_b * 112 + a0_u * 14 + 4 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 4) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c4_1);
        }
        if (a0_b * 112 + a0_u * 14 + 5 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 5) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c5_0);
        }
        if (a0_b * 112 + a0_u * 14 + 5 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 5) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c5_1);
        }
        if (a0_b * 112 + a0_u * 14 + 6 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 6) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c6_0);
        }
        if (a0_b * 112 + a0_u * 14 + 6 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 6) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c6_1);
        }
        if (a0_b * 112 + a0_u * 14 + 7 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 7) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c7_0);
        }
        if (a0_b * 112 + a0_u * 14 + 7 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 7) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c7_1);
        }
        if (a0_b * 112 + a0_u * 14 + 8 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 8) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c8_0);
        }
        if (a0_b * 112 + a0_u * 14 + 8 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 8) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c8_1);
        }
        if (a0_b * 112 + a0_u * 14 + 9 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 9) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c9_0);
        }
        if (a0_b * 112 + a0_u * 14 + 9 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 9) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c9_1);
        }
        if (a0_b * 112 + a0_u * 14 + 10 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 10) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c10_0);
        }
        if (a0_b * 112 + a0_u * 14 + 10 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 10) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c10_1);
        }
        if (a0_b * 112 + a0_u * 14 + 11 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 11) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c11_0);
        }
        if (a0_b * 112 + a0_u * 14 + 11 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 11) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c11_1);
        }
        if (a0_b * 112 + a0_u * 14 + 12 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 12) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c12_0);
        }
        if (a0_b * 112 + a0_u * 14 + 12 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 12) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c12_1);
        }
        if (a0_b * 112 + a0_u * 14 + 13 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 13) % 1024 * 128 + (a1_b * 64 + a1_u * 2)], acc0__c13_0);
        }
        if (a0_b * 112 + a0_u * 14 + 13 < 1024) {
            atomicAdd(&matmul[(a0_b * 112 + a0_u * 14 + 13) % 1024 * 128 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c13_1);
        }
}
