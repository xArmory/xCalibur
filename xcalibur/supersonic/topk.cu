#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include "ptx.inl"

template <bool softmax>
__global__ void topk_kernel(
    const __nv_bfloat16* router_logits,
    uint32_t* tKwi,
    const int K,
    const int N,
    const int E
){
    if (E <= 0 || E > 65536 || N <= 0 || N > 65536 || K <= 0 || K > 16 || K > E) return;
    if (blockDim.x != 8 || blockDim.y != 4 || blockDim.z != 8) return;
    float softmax_deno = 0.0f;
    float softmax_max = -FLT_MAX;
    uint32_t rmem[32];
    const int lane = (threadIdx.x + (threadIdx.y << 3));
    if (((uint64_t(blockIdx.x) << 3) + threadIdx.z) >= N) {
        return;
    }
    if (softmax) {
        for (int i = 0; i < ((E + 255) >> 8); i++) {
            if ((lane << 3) + (i << 8) < E) {
                ldcg_b16x8(
                    router_logits + (((uint64_t(blockIdx.x) << 3) + threadIdx.z) * E)
                    + (lane << 3) + (i << 8), rmem,
                    min(8, E - (lane << 3) - (i << 8))
                );
                #pragma unroll 4
                for (int j = 0; j < 4; j++) {
                    if ((lane << 3) + (i << 8) + (j << 1) < E) {
                        softmax_max = fmaxf(softmax_max, __uint_as_float(rmem[j] << 16));
                    }
                    if ((lane << 3) + (i << 8) + (j << 1) + 1 < E) {
                        softmax_max = fmaxf(softmax_max, __uint_as_float(rmem[j] & 0xffff'0000u));
                    }
                }
            }
        }
        #pragma unroll 5
        for (int i = 16; i > 0; i >>= 1) {
            softmax_max = fmaxf(softmax_max, __shfl_xor_sync(0xffff'ffffu, softmax_max, i));
        }
    }
    #pragma unroll 8
    for (int k = 0; k < K; k++) {
        rmem[16 + k] = 0u;
    }
    for (int i = 0; i <= ((E + 255) >> 8); i++) {
        if (i && ((lane << 3) + ((i - 1) << 8)) < E) {
            for (int j = (((i - 1) & 1) << 2); j < ((((i - 1) & 1) << 2) + 4); j++) {
                if ((lane << 3) + ((i - 1) << 8) + ((j & 3) << 1) >= E) continue;
                if (softmax) {
                    rmem[j] = softmax_bf16x2(rmem[j], softmax_max);
                    softmax_deno += __uint_as_float(rmem[j] << 16);
                    if ((lane << 3) + ((i - 1) << 8) + ((j & 3) << 1) + 1 < E) {
                        softmax_deno += __uint_as_float(rmem[j] & 0xffff'0000u);
                    }
                }
                if (!softmax) {
                    rmem[j] = softmax_bf16x2(rmem[j] ^ 0x8000'8000u);
                    add_bf16x2(rmem[j], 0x3f80'3f80u);
                    rcp_bf16x2(rmem[j]);
                }
                uint16_t e_offset = (uint16_t)(0xffffu - ((lane << 3) + ((i - 1) << 8) + ((j & 3) << 1)));
                rmem[j ^ 4] =  (rmem[j] << 16) | e_offset;
                rmem[j] =  (rmem[j] & 0xffff'0000u) | (e_offset - 1u);
                #pragma unroll 2
                for (int candidate = 0; candidate < 2; candidate++) {
                    if ((lane << 3) + ((i - 1) << 8) + ((j & 3) << 1) + candidate >= E) continue;
                    uint32_t key = candidate ? rmem[j] : rmem[j ^ 4];
                    for (int k = 0; k < min(K, ((i - 1) << 3) + ((j & 3) << 1) + candidate + 1); k++) {
                        uint32_t tmp = rmem[16 + k];
                        rmem[16 + k] = max(tmp, key);
                        key = min(tmp, key);
                    }
                }
            }
        }
        if (((lane << 3) + (i << 8)) < E) {
            ldcg_b16x8(
                router_logits
                + (((uint64_t(blockIdx.x) << 3) + threadIdx.z) * E)
                + (lane << 3) + (i << 8),
                &rmem[(i & 1) << 2], min(8, E - (lane << 3) - (i << 8))
            );
        }
    }
    if (softmax) {
        #pragma unroll 5
        for (int i = 16; i > 0; i >>= 1) {
            softmax_deno += __shfl_xor_sync(0xffff'ffffu, softmax_deno, i);
        }
        softmax_deno = rcp_f32(softmax_deno);
    }
    int p = 16;
    uint32_t key = rmem[p];
    for (int k = 0; k < K; k++) {
        uint32_t tmp = key;
        #pragma unroll 5
        for (int i = 16; i > 0; i >>= 1) {
            tmp = max(tmp, __shfl_xor_sync(0xffff'ffffu, tmp, i));
        }
        rmem[k] = tmp;
        if (key == tmp) {
            ++p;
            key = (p < K + 16) ? rmem[p] : 0u;
        }
    }
    if (lane < (K << 1)) {
        int k = (lane < K) ? lane : lane - K;
        if (lane >= K) {
            if (softmax) rmem[k] = softmax_mul(rmem[k], softmax_deno);
            tKwi[(((uint64_t(blockIdx.x) << 3) + threadIdx.z) * K) + (uint64_t)(k)] = rmem[k];
        }
    }
}
