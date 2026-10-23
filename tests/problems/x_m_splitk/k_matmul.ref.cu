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
__launch_bounds__(512) void k_matmul(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 512;
        int a0_b = (_gid / 512) % 1;
        int a1_b = (_gid / 512) % 1;
        int a0_u = (_gid / 32) % 16;
        int a1_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        __shared__ __align__(16) float _a_smem[2048];
        __shared__ __align__(16) float _b_smem[2048];
        for (int _ks = 0; _ks < 64; _ks += 64) {
            emmy_cp_async_cg(&_a_smem[0 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)], &x0[(a0_b * 32 + (a0_u * 32 + a1_u) / 16) * 4096 + (_a2_ks * 64 + (_ks + 4 * ((a0_u * 32 + a1_u) % 16)))]);
            emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 32 + a1_u) * 4 / 32) * 32 + ((a0_u * 32 + a1_u) * 4 % 32)], &x1[(_a2_ks * 64 + (_ks + (a0_u * 32 + a1_u) / 8)) * 32 + (a1_b * 32 + 4 * ((a0_u * 32 + a1_u) % 8))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<0>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 64; _ki++) {
                float in1__c0_0 = _b_smem[_ki * 32 + (a1_b * 32 + a1_u - a1_b * 32)];
                float in0__c0_0 = _a_smem[(a0_b * 32 + a0_u * 2 - a0_b * 32) * 64 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in0__c1_0 = _a_smem[(a0_b * 32 + a0_u * 2 + 1 - a0_b * 32) * 64 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
            }
            __syncthreads();
        }
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 2) * 32 + (a1_b * 32 + a1_u)], acc0__c0_0);
        atomicAdd(&matmul[(a0_b * 32 + a0_u * 2 + 1) * 32 + (a1_b * 32 + a1_u)], acc0__c1_0);
}
