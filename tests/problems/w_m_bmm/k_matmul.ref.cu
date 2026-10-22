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
        int _a3_ks = _gid / 2048;
        int a0 = (_gid / 512) % 4;
        int a1_b = (_gid / 256) % 2;
        int a2_b = (_gid / 256) % 1;
        int a1_u = (_gid / 16) % 16;
        int a2_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c0_3 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c1_2 = 0.0f;
        float acc0__c1_3 = 0.0f;
        __shared__ __align__(16) float _a_smem[128];
        __shared__ __align__(16) float _b_smem[256];
        for (int _ks = 0; _ks < 4; _ks += 4) {
            for (int _fa = a1_u * 16 + a2_u; _fa < 32; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[a0 * 2048 + (a1_b * 32 + _fa) * 32 + (_a3_ks * 4 + _ks)]);
            }
            for (int _fb = a1_u * 16 + a2_u; _fb < 64; _fb += 256) {
                emmy_cp_async_cg(&_b_smem[_fb * 4], &x1[a0 * 2048 + (_a3_ks * 4 + (_ks + _fb / 16)) * 64 + (a2_b * 64 + 4 * (_fb % 16))]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 4; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 64 + (a2_b * 64 + a2_u * 4 - a2_b * 64)];
                float in0__c0_0 = _a_smem[(a1_b * 32 + a1_u * 2 - a1_b * 32) * 4 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[_ki * 64 + (a2_b * 64 + a2_u * 4 + 1 - a2_b * 64)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in1__c0_2 = _b_smem[_ki * 64 + (a2_b * 64 + a2_u * 4 + 2 - a2_b * 64)];
                float acc0__v__c0_2 = in1__c0_2 * in0__c0_0;
                acc0__c0_2 += acc0__v__c0_2;
                float in1__c0_3 = _b_smem[_ki * 64 + (a2_b * 64 + a2_u * 4 + 3 - a2_b * 64)];
                float acc0__v__c0_3 = in1__c0_3 * in0__c0_0;
                acc0__c0_3 += acc0__v__c0_3;
                float in0__c1_0 = _a_smem[(a1_b * 32 + a1_u * 2 + 1 - a1_b * 32) * 4 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                float acc0__v__c1_2 = in1__c0_2 * in0__c1_0;
                acc0__c1_2 += acc0__v__c1_2;
                float acc0__v__c1_3 = in1__c0_3 * in0__c1_0;
                acc0__c1_3 += acc0__v__c1_3;
            }
            __syncthreads();
        }
        atomicAdd(&matmul[a0 * 4096 + (a1_b * 32 + a1_u * 2) * 64 + (a2_b * 64 + a2_u * 4)], acc0__c0_0);
        atomicAdd(&matmul[a0 * 4096 + (a1_b * 32 + a1_u * 2) * 64 + (a2_b * 64 + a2_u * 4 + 1)], acc0__c0_1);
        atomicAdd(&matmul[a0 * 4096 + (a1_b * 32 + a1_u * 2) * 64 + (a2_b * 64 + a2_u * 4 + 2)], acc0__c0_2);
        atomicAdd(&matmul[a0 * 4096 + (a1_b * 32 + a1_u * 2) * 64 + (a2_b * 64 + a2_u * 4 + 3)], acc0__c0_3);
        atomicAdd(&matmul[a0 * 4096 + (a1_b * 32 + a1_u * 2 + 1) * 64 + (a2_b * 64 + a2_u * 4)], acc0__c1_0);
        atomicAdd(&matmul[a0 * 4096 + (a1_b * 32 + a1_u * 2 + 1) * 64 + (a2_b * 64 + a2_u * 4 + 1)], acc0__c1_1);
        atomicAdd(&matmul[a0 * 4096 + (a1_b * 32 + a1_u * 2 + 1) * 64 + (a2_b * 64 + a2_u * 4 + 2)], acc0__c1_2);
        atomicAdd(&matmul[a0 * 4096 + (a1_b * 32 + a1_u * 2 + 1) * 64 + (a2_b * 64 + a2_u * 4 + 3)], acc0__c1_3);
}
