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
__launch_bounds__(256) void k_matmul(const double* x1, const double* x0, double* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 1024;
        int a0_b = (_gid / 256) % 4;
        int a1_b = (_gid / 256) % 1;
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
        __shared__ __align__(16) double _a_smem[256];
        __shared__ __align__(16) double _b_smem[512];
        for (int _ks = 0; _ks < 8; _ks += 8) {
            for (int _fa = a0_u * 32 + a1_u; _fa < 128; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[_fa * 2], &x0[(a0_b * 32 + _fa / 4) * 256 + (_a2_ks * 8 + (_ks + 2 * (_fa % 4)))]);
            }
            emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 32 + a1_u) * 2 / 64) * 64 + ((a0_u * 32 + a1_u) * 2 % 64)], &x1[(_a2_ks * 8 + (_ks + (a0_u * 32 + a1_u) / 32)) * 64 + (a1_b * 64 + 2 * ((a0_u * 32 + a1_u) % 32))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 8; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u * 2 - a1_b * 64)];
                float in0__c0_0 = _a_smem[(a0_b * 32 + a0_u * 4 - a0_b * 32) * 8 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[_ki * 64 + (a1_b * 64 + a1_u * 2 + 1 - a1_b * 64)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in0__c1_0 = _a_smem[(a0_b * 32 + a0_u * 4 + 1 - a0_b * 32) * 8 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                float in0__c2_0 = _a_smem[(a0_b * 32 + a0_u * 4 + 2 - a0_b * 32) * 8 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float acc0__v__c2_1 = in1__c0_1 * in0__c2_0;
                acc0__c2_1 += acc0__v__c2_1;
                float in0__c3_0 = _a_smem[(a0_b * 32 + a0_u * 4 + 3 - a0_b * 32) * 8 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float acc0__v__c3_1 = in1__c0_1 * in0__c3_0;
                acc0__c3_1 += acc0__v__c3_1;
            }
            __syncthreads();
        }
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 4) * 64 + (a1_b * 64 + a1_u * 2)], acc0__c0_0);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 4) * 64 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c0_1);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 1) * 64 + (a1_b * 64 + a1_u * 2)], acc0__c1_0);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 1) * 64 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c1_1);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 2) * 64 + (a1_b * 64 + a1_u * 2)], acc0__c2_0);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 2) * 64 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c2_1);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 3) * 64 + (a1_b * 64 + a1_u * 2)], acc0__c3_0);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 4 + 3) * 64 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c3_1);
}
