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
    extern __shared__ __align__(16) unsigned char _smem_pool[];  // 55296 bytes
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int _a2_ks = _gid / 14336;
        int a0_b = (_gid / 2048) % 7;
        int a1_b = (_gid / 256) % 8;
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
        int* _a_smem = reinterpret_cast<int*>(_smem_pool + 0);
        int* _b_smem = reinterpret_cast<int*>(_smem_pool + 30720);
        for (int _fa = a0_u * 32 + a1_u; _fa < 640; _fa += 256) {
            emmy_cp_async_cg(&_a_smem[_fa * 4], &x0[((a0_b * 80 + _fa / 8 < 512) ? (a0_b * 80 + _fa / 8) : (511)) * 512 + (_a2_ks * 256 + 4 * (_fa % 8))]);
        }
        emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)], &x1[(_a2_ks * 256 + (a0_u * 32 + a1_u) / 16) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);
        emmy_cp_async_cg(&_b_smem[1024 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)], &x1[(_a2_ks * 256 + ((a0_u * 32 + a1_u) / 16 + 16)) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);
        emmy_cp_async_commit();
        for (int _fa = a0_u * 32 + a1_u; _fa < 640; _fa += 256) {
            emmy_cp_async_cg(&_a_smem[(80 + _fa / 8) * 32 + 4 * (_fa % 8)], &x0[((a0_b * 80 + _fa / 8 < 512) ? (a0_b * 80 + _fa / 8) : (511)) * 512 + (_a2_ks * 256 + (32 + 4 * (_fa % 8)))]);
        }
        emmy_cp_async_cg(&_b_smem[2048 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)], &x1[(_a2_ks * 256 + (32 + (a0_u * 32 + a1_u) / 16)) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);
        emmy_cp_async_cg(&_b_smem[3072 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)], &x1[(_a2_ks * 256 + (32 + ((a0_u * 32 + a1_u) / 16 + 16))) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);
        emmy_cp_async_commit();
        for (int _ks = 0; _ks < 256; _ks += 32) {
            for (int _fa = a0_u * 32 + a1_u; _fa < 640; _fa += 256) {
                emmy_cp_async_cg(&_a_smem[((_ks / 32 + 2) % 3 * 80 + _fa / 8) * 32 + 4 * (_fa % 8)], &x0[((a0_b * 80 + _fa / 8 < 512) ? (a0_b * 80 + _fa / 8) : (511)) * 512 + (_a2_ks * 256 + (((_ks + 64 < 256) ? (_ks + 64) : (224)) + 4 * (_fa % 8)))]);
            }
            emmy_cp_async_cg(&_b_smem[(_ks / 32 + 2) % 3 * 32 * 64 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)], &x1[(_a2_ks * 256 + (((_ks + 64 < 256) ? (_ks + 64) : (224)) + (a0_u * 32 + a1_u) / 16)) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 32 + 2) % 3 * 32 + 16) * 64 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)], &x1[(_a2_ks * 256 + (((_ks + 64 < 256) ? (_ks + 64) : (224)) + ((a0_u * 32 + a1_u) / 16 + 16))) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);
            emmy_cp_async_commit();
            emmy_cp_async_wait<2>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 32; _ki++) {
                int in1__c0_0 = _b_smem[(_ks / 32 % 3 * 32 + _ki) * 64 + (a1_b * 64 + a1_u * 2 - a1_b * 64)];
                int in0__c0_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                int in1__c0_1 = _b_smem[(_ks / 32 % 3 * 32 + _ki) * 64 + (a1_b * 64 + a1_u * 2 + 1 - a1_b * 64)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                int in0__c1_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 1 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
                int in0__c2_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 2 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c2_0 = in1__c0_0 * in0__c2_0;
                acc0__c2_0 += acc0__v__c2_0;
                float acc0__v__c2_1 = in1__c0_1 * in0__c2_0;
                acc0__c2_1 += acc0__v__c2_1;
                int in0__c3_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 3 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c3_0 = in1__c0_0 * in0__c3_0;
                acc0__c3_0 += acc0__v__c3_0;
                float acc0__v__c3_1 = in1__c0_1 * in0__c3_0;
                acc0__c3_1 += acc0__v__c3_1;
                int in0__c4_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 4 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c4_0 = in1__c0_0 * in0__c4_0;
                acc0__c4_0 += acc0__v__c4_0;
                float acc0__v__c4_1 = in1__c0_1 * in0__c4_0;
                acc0__c4_1 += acc0__v__c4_1;
                int in0__c5_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 5 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c5_0 = in1__c0_0 * in0__c5_0;
                acc0__c5_0 += acc0__v__c5_0;
                float acc0__v__c5_1 = in1__c0_1 * in0__c5_0;
                acc0__c5_1 += acc0__v__c5_1;
                int in0__c6_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 6 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c6_0 = in1__c0_0 * in0__c6_0;
                acc0__c6_0 += acc0__v__c6_0;
                float acc0__v__c6_1 = in1__c0_1 * in0__c6_0;
                acc0__c6_1 += acc0__v__c6_1;
                int in0__c7_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 7 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c7_0 = in1__c0_0 * in0__c7_0;
                acc0__c7_0 += acc0__v__c7_0;
                float acc0__v__c7_1 = in1__c0_1 * in0__c7_0;
                acc0__c7_1 += acc0__v__c7_1;
                int in0__c8_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 8 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c8_0 = in1__c0_0 * in0__c8_0;
                acc0__c8_0 += acc0__v__c8_0;
                float acc0__v__c8_1 = in1__c0_1 * in0__c8_0;
                acc0__c8_1 += acc0__v__c8_1;
                int in0__c9_0 = _a_smem[(_ks / 32 % 3 * 80 + (a0_b * 80 + a0_u * 10 + 9 - a0_b * 80)) * 32 + _ki];
                float acc0__v__c9_0 = in1__c0_0 * in0__c9_0;
                acc0__c9_0 += acc0__v__c9_0;
                float acc0__v__c9_1 = in1__c0_1 * in0__c9_0;
                acc0__c9_1 += acc0__v__c9_1;
            }
            __syncthreads();
        }
        if (a0_b * 80 + a0_u * 10 + 0 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c0_0);
        }
        if (a0_b * 80 + a0_u * 10 + 0 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c0_1);
        }
        if (a0_b * 80 + a0_u * 10 + 1 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 1) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c1_0);
        }
        if (a0_b * 80 + a0_u * 10 + 1 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 1) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c1_1);
        }
        if (a0_b * 80 + a0_u * 10 + 2 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 2) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c2_0);
        }
        if (a0_b * 80 + a0_u * 10 + 2 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 2) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c2_1);
        }
        if (a0_b * 80 + a0_u * 10 + 3 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 3) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c3_0);
        }
        if (a0_b * 80 + a0_u * 10 + 3 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 3) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c3_1);
        }
        if (a0_b * 80 + a0_u * 10 + 4 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 4) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c4_0);
        }
        if (a0_b * 80 + a0_u * 10 + 4 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 4) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c4_1);
        }
        if (a0_b * 80 + a0_u * 10 + 5 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 5) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c5_0);
        }
        if (a0_b * 80 + a0_u * 10 + 5 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 5) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c5_1);
        }
        if (a0_b * 80 + a0_u * 10 + 6 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 6) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c6_0);
        }
        if (a0_b * 80 + a0_u * 10 + 6 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 6) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c6_1);
        }
        if (a0_b * 80 + a0_u * 10 + 7 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 7) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c7_0);
        }
        if (a0_b * 80 + a0_u * 10 + 7 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 7) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c7_1);
        }
        if (a0_b * 80 + a0_u * 10 + 8 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 8) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c8_0);
        }
        if (a0_b * 80 + a0_u * 10 + 8 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 8) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c8_1);
        }
        if (a0_b * 80 + a0_u * 10 + 9 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 9) % 512 * 512 + (a1_b * 64 + a1_u * 2)], acc0__c9_0);
        }
        if (a0_b * 80 + a0_u * 10 + 9 < 512) {
            atomicAdd(&matmul[(a0_b * 80 + a0_u * 10 + 9) % 512 * 512 + (a1_b * 64 + a1_u * 2 + 1)], acc0__c9_1);
        }
}
