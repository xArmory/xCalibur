#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include "ptx.inl"


__device__ __forceinline__ void xR38FF1_bf16(
	uint32_t* W13,
	uint32_t* Xs,
    __nv_bfloat16* Y,
	uint32_t* rmem,
	uint32_t* smem,
    int32_t E, int32_t N, int32_t I, int32_t H
){

	for (int32_t i = (threadIdx.x >> 2); i < I; i += (CTA >> 1)) {

		for (int32_t kt = 0; kt < (H >> 1); kt += CTA) {

			asm volatile(
				"cp.async.ca.shared::cta.global.L2::no_allocate [%0], [%1], 16;\n\t"
				"cp.async.commit_group;\n\t"
				"cp.async.ca.shared::cta.global.L2::no_allocate [%0], [%1], 16;\n\t"
				"cp.async.wait_all;\n\t"
				"cp.async.commit_group;\n\t"
				"cp.async.wait_all;\n\t" //@TODO pipeline
				: "r"(
					(uint32_t)__cvta_shared_to_generic(
						smem + 8 + (threadIdx.x << 8)
					))
				:"l"(
					(uint64_t)__cvta_global_to_generic(
						Xs + ((blockIdx.x) * (H << 2)) + (threadIdx.x << 8)
					)
				)
			);
					
			for (int32_t h4 = kt + (threadIdx.x & 3); (h4 < kt + CTA) && (kt + CTA < (H >> 1)); h4 += 32) {

				
				ldcg_b32v4(
						(uint64_t)__cvta_global_to_generic(
							W13 + (uint64_t)(blockIdx.x * I * (H >> 1)) + (i * (H >> 1)) + (h4 << 2)
						),
						rmem + ((h4 >> 5) << 2)
				);
				
				asm volatile(
						"ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
						"{%0, %1, %2, %3}, [%4];\n"
						: "=r"(rmem[25]), "=r"(rmem[26]), "=r"(rmem[27]), "=r"(rmem[28])
						: "r"(
							(uint32_t)__cvta_generic_to_shared(
							smem + 8
							+ (h32 << 7)
							+ ((threadIdx.x & 31) << 2))
						)
						: "memory"
				);

				ldcg_b32v4(
						(uint64_t)__cvta_global_to_generic(
							W13 + (uint64_t)(blockIdx.x * I * (H >> 1)) + (i * (H >> 1)) + ((h4 + 1) << 2)
						),
						rmem + (((h4 >> 5) + 1) << 2)
				);

				mma_sp_m16n8k32_bf16<0>(
					rmem[0], rmem[2], rmem[4], rmem[6],
					rmem + 25, rmem + 29, 0x44444444u
				);

				ldcg_b32v4(
						(uint64_t)__cvta_global_to_generic(
							W13 + (uint64_t)(blockIdx.x * I * (H >> 1)) + (i * (H >> 1)) + ((h4 + 2) << 2)
						),
						rmem + (((h4 >> 5) + 2) << 2)
				);

				mma_sp_m16n8k32_bf16<0>(
					rmem[1], rmem[3], rmem[5], rmem[7],
					rmem + 25, rmem + 29, 0xEEEEEEEEu
				);

				ldcg_b32v4(
						(uint64_t)__cvta_global_to_generic(
							W13 + (uint64_t)(blockIdx.x * I * (H >> 1)) + (i * (H >> 1)) + ((h4 + 3) << 2)
						),
						rmem + (((h4 >> 5) + 3) << 2)
				);

				mma_sp_m16n8k32_bf16<1>(
					rmem[8], rmem[10], rmem[12], rmem[14],
					rmem + 27, rmem + 29, 0x44444444u
				);

				mma_sp_m16n8k32_bf16<1>(
					rmem[9], rmem[11], rmem[13], rmem[15],
					rmem + 27, rmem + 29, 0xEEEEEEEEu
				);
			}
		}
	}	
}


__global__ __launch_bounds__(CTA, 2)
void xR38GFF1_bf16(
    __nv_bfloat16* W13,
    __nv_bfloat16* X,
	__nv_bfloat16* Xs, // [E, (8*H) >> 4] @TODO: use global scratch
    __nv_bfloat16* Y,
    uint32_t* tKwi,
    int32_t E, int32_t N, int32_t I, int32_t H, int32_t K
){

    uint32_t rmem[38];
    __shared__ __align__(4) uint32_t smem[8192];

    for (int32_t n = (threadIdx.x >> 1); n < N; n+=(CTA >> 1)) {
		
		#pragma unroll 4
		for (int32_t rk = 0; rk < 4; rk++) {
			rmem[1 + rk] = 0u;
		}

		for (int32_t kwip = 0; kwip < (K >> 3); kwip++) {
			if (!rmem[1]) {
				ldcg_b32v4(
					(uint64_t)__cvta_global_to_generic(
						tKwi + n * K + ((threadIdx.x & 1) << 2)
					), 
					&rmem[1]
				);

				#pragma unroll 4
				for (int32_t kwi = 0; kwi < 4; kwi++) {
					if ((rmem[1 + kwi]) & 0x0000'ffffu)==(blockIdx.x) {
						rmem[1] = rmem[1 + kwi];
						break;
					}
				}
			}
		}
		rmem[1] = max(rmem[1], __shfl_xor_sync(0xffff'ffffu, rmem[1], 1, 2));

		for (int32_t np = 0; np < (CTA >> 1); np++) {
			if ((((threadIdx.x >> 1))==np) && rmem[1]) {
				smem[rmem[0]] = (rmem[1] & 0xffff'0000u) | n
			}
			__syncthreads();
			if (smem[rmem[0]]) {
				ldcg_b32v4(
					(uint64_t)__cvta_global_to_generic(
						X + (smem[rmem[0]] & 0x0000'ffffu) * H + (threadIdx.x << 3)
					),
					&rmem[2]
				);
				// (blockIdx.x) * (8 * H / 2) + (threadIdx.x << 5) + (rmem[0] << 2)
				rmem[0]++;
			}
			if (rmem[0]==8) {
				xR38FF1_bf16(W13, Xs, Y, rmem + 2, smem);
				rmem[0] = 0u;
			}
		}
	}
	//tail < N clause
}