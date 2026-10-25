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
__launch_bounds__(256) void k_matmul(const int* x1, const int* x0, int* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 1024;
        int a0_b = (_gid / 512) % 2;
        int a1_b = (_gid / 256) % 2;
        int a0_u = (_gid / 32) % 8;
        int a1_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c5_0 = 0.0f;
        float acc0__c6_0 = 0.0f;
        float acc0__c7_0 = 0.0f;
        float acc0__c8_0 = 0.0f;
        float acc0__c9_0 = 0.0f;
        __shared__ __align__(16) int _a_smem[320];
        __shared__ __align__(16) int _b_smem[128];
        for (int _ks = 0; _ks < 4; _ks += 4) {
            for (int _fa = a0_u * 32 + a1_u; _fa < 80; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[((a0_b * 80 + _fa < 128) ? (a0_b * 80 + _fa) : (127)) * 256 + (_a2_ks * 4 + _ks)]);
            }
            for (int _fb = a0_u * 32 + a1_u; _fb < 32; _fb += 256) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[(_a2_ks * 4 + (_ks + _fb / 8)) * 64 + (a1_b * 32 + 4 * (_fb % 8))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 4; _ki++) {
                int in1__c0_0 = _b_smem[_ki * 32 + (a1_b * 32 + a1_u - a1_b * 32)];
                int in0__c0_0 = _a_smem[(a0_b * 80 + a0_u * 10 - a0_b * 80) * 4 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                int in0__c1_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 1 - a0_b * 80) * 4 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                int in0__c2_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 2 - a0_b * 80) * 4 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                int in0__c3_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 3 - a0_b * 80) * 4 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                int in0__c4_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 4 - a0_b * 80) * 4 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                int in0__c5_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 5 - a0_b * 80) * 4 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                int in0__c6_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 6 - a0_b * 80) * 4 + _ki];
                float acc0__v__c6_0 = in1__c0_0 * in0__c6_0;
                acc0__c6_0 += acc0__v__c6_0;
                int in0__c7_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 7 - a0_b * 80) * 4 + _ki];
                float acc0__v__c7_0 = in1__c0_0 * in0__c7_0;
                acc0__c7_0 += acc0__v__c7_0;
                int in0__c8_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 8 - a0_b * 80) * 4 + _ki];
                float acc0__v__c8_0 = in1__c0_0 * in0__c8_0;
                acc0__c8_0 += acc0__v__c8_0;
                int in0__c9_0 = _a_smem[(a0_b * 80 + a0_u * 10 + 9 - a0_b * 80) * 4 + _ki];
                float acc0__v__c9_0 = in1__c0_0 * in0__c9_0;
                acc0__c9_0 += acc0__v__c9_0;
            }
            __syncthreads();
        }
        if (a0_b * 80 + a0_u * 10 + 0 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c0_0);
        }
        if (a0_b * 80 + a0_u * 10 + 1 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 1) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c1_0);
        }
        if (a0_b * 80 + a0_u * 10 + 2 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 2) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c2_0);
        }
        if (a0_b * 80 + a0_u * 10 + 3 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 3) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c3_0);
        }
        if (a0_b * 80 + a0_u * 10 + 4 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 4) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c4_0);
        }
        if (a0_b * 80 + a0_u * 10 + 5 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 5) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c5_0);
        }
        if (a0_b * 80 + a0_u * 10 + 6 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 6) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c6_0);
        }
        if (a0_b * 80 + a0_u * 10 + 7 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 7) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c7_0);
        }
        if (a0_b * 80 + a0_u * 10 + 8 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 8) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c8_0);
        }
        if (a0_b * 80 + a0_u * 10 + 9 < 128) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 9) % 128 * 64 + (a1_b * 32 + a1_u)], acc0__c9_0);
        }
}
