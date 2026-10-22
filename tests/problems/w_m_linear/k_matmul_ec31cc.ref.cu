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
__launch_bounds__(128) void k_matmul_ec31cc(const float* b, const float* x, const float* w, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 4096;
        int a1_b = (_gid / 128) % 32;
        int a0_u = (_gid / 16) % 8;
        int a1_u = (_gid) % 16;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        __shared__ __align__(16) float _a_smem[4096];
        __shared__ __align__(16) float _b_smem[4096];
        emmy_cp_async_cg(&_a_smem[0 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[(a0_b * 16 + (a0_u * 16 + a1_u) / 32) * 256 + 4 * ((a0_u * 16 + a1_u) % 32)]);
        emmy_cp_async_cg(&_a_smem[512 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[(a0_b * 16 + ((a0_u * 16 + a1_u) / 32 + 4)) * 256 + 4 * ((a0_u * 16 + a1_u) % 32)]);
        emmy_cp_async_cg(&_a_smem[1024 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[(a0_b * 16 + ((a0_u * 16 + a1_u) / 32 + 8)) * 256 + 4 * ((a0_u * 16 + a1_u) % 32)]);
        emmy_cp_async_cg(&_a_smem[1536 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[(a0_b * 16 + ((a0_u * 16 + a1_u) / 32 + 12)) * 256 + 4 * ((a0_u * 16 + a1_u) % 32)]);
        emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 16 + a1_u) * 4 / 16) * 16 + ((a0_u * 16 + a1_u) * 4 % 16)], &w[(a0_u * 16 + a1_u) / 4 * 512 + (a1_b * 16 + 4 * ((a0_u * 16 + a1_u) % 4))]);
        emmy_cp_async_cg(&_b_smem[512 + ((a0_u * 16 + a1_u) * 4 / 16) * 16 + ((a0_u * 16 + a1_u) * 4 % 16)], &w[((a0_u * 16 + a1_u) / 4 + 32) * 512 + (a1_b * 16 + 4 * ((a0_u * 16 + a1_u) % 4))]);
        emmy_cp_async_cg(&_b_smem[1024 + ((a0_u * 16 + a1_u) * 4 / 16) * 16 + ((a0_u * 16 + a1_u) * 4 % 16)], &w[((a0_u * 16 + a1_u) / 4 + 64) * 512 + (a1_b * 16 + 4 * ((a0_u * 16 + a1_u) % 4))]);
        emmy_cp_async_cg(&_b_smem[1536 + ((a0_u * 16 + a1_u) * 4 / 16) * 16 + ((a0_u * 16 + a1_u) * 4 % 16)], &w[((a0_u * 16 + a1_u) / 4 + 96) * 512 + (a1_b * 16 + 4 * ((a0_u * 16 + a1_u) % 4))]);
        emmy_cp_async_commit();
        for (int _ks = 0; _ks < 256; _ks += 128) {
            emmy_cp_async_cg(&_a_smem[(_ks / 128 + 1) % 2 * 16 * 128 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[(a0_b * 16 + (a0_u * 16 + a1_u) / 32) * 256 + (((_ks + 128 < 256) ? (_ks + 128) : (128)) + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[((_ks / 128 + 1) % 2 * 16 + 4) * 128 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[(a0_b * 16 + ((a0_u * 16 + a1_u) / 32 + 4)) * 256 + (((_ks + 128 < 256) ? (_ks + 128) : (128)) + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[((_ks / 128 + 1) % 2 * 16 + 8) * 128 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[(a0_b * 16 + ((a0_u * 16 + a1_u) / 32 + 8)) * 256 + (((_ks + 128 < 256) ? (_ks + 128) : (128)) + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_a_smem[((_ks / 128 + 1) % 2 * 16 + 12) * 128 + ((a0_u * 16 + a1_u) * 4 / 128) * 128 + ((a0_u * 16 + a1_u) * 4 % 128)], &x[(a0_b * 16 + ((a0_u * 16 + a1_u) / 32 + 12)) * 256 + (((_ks + 128 < 256) ? (_ks + 128) : (128)) + 4 * ((a0_u * 16 + a1_u) % 32))]);
            emmy_cp_async_cg(&_b_smem[(_ks / 128 + 1) % 2 * 128 * 16 + ((a0_u * 16 + a1_u) * 4 / 16) * 16 + ((a0_u * 16 + a1_u) * 4 % 16)], &w[(((_ks + 128 < 256) ? (_ks + 128) : (128)) + (a0_u * 16 + a1_u) / 4) * 512 + (a1_b * 16 + 4 * ((a0_u * 16 + a1_u) % 4))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 128 + 1) % 2 * 128 + 32) * 16 + ((a0_u * 16 + a1_u) * 4 / 16) * 16 + ((a0_u * 16 + a1_u) * 4 % 16)], &w[(((_ks + 128 < 256) ? (_ks + 128) : (128)) + ((a0_u * 16 + a1_u) / 4 + 32)) * 512 + (a1_b * 16 + 4 * ((a0_u * 16 + a1_u) % 4))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 128 + 1) % 2 * 128 + 64) * 16 + ((a0_u * 16 + a1_u) * 4 / 16) * 16 + ((a0_u * 16 + a1_u) * 4 % 16)], &w[(((_ks + 128 < 256) ? (_ks + 128) : (128)) + ((a0_u * 16 + a1_u) / 4 + 64)) * 512 + (a1_b * 16 + 4 * ((a0_u * 16 + a1_u) % 4))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 128 + 1) % 2 * 128 + 96) * 16 + ((a0_u * 16 + a1_u) * 4 / 16) * 16 + ((a0_u * 16 + a1_u) * 4 % 16)], &w[(((_ks + 128 < 256) ? (_ks + 128) : (128)) + ((a0_u * 16 + a1_u) / 4 + 96)) * 512 + (a1_b * 16 + 4 * ((a0_u * 16 + a1_u) % 4))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<1>();
            __syncthreads();
            for (int _ki = 0; _ki < 128; _ki++) {
                float in0__c0_0 = _b_smem[(_ks / 128 % 2 * 128 + _ki) * 16 + (a1_b * 16 + a1_u - a1_b * 16)];
                float in1__c0_0 = _a_smem[(_ks / 128 % 2 * 16 + (a0_b * 16 + a0_u * 2 - a0_b * 16)) * 128 + _ki];
                float acc0__v__c0_0 = in0__c0_0 * in1__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c1_0 = _a_smem[(_ks / 128 % 2 * 16 + (a0_b * 16 + a0_u * 2 + 1 - a0_b * 16)) * 128 + _ki];
                float acc0__v__c1_0 = in0__c0_0 * in1__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
            }
            __syncthreads();
        }
        float in2__c0_0 = b[a1_b * 16 + a1_u];
        float v1__c0_0 = acc0__c0_0 + in2__c0_0;
        add[(a0_b * 16 + a0_u * 2) * 512 + (a1_b * 16 + a1_u)] = v1__c0_0;
        float in2__c1_0 = b[a1_b * 16 + a1_u];
        float v1__c1_0 = acc0__c1_0 + in2__c1_0;
        add[(a0_b * 16 + a0_u * 2 + 1) * 512 + (a1_b * 16 + a1_u)] = v1__c1_0;
}
