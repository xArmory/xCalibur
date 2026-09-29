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

	for (int32_t i = (threadIdx.x >> 2); i < I; i += (blockDim.x >> 1)) {

		for (int32_t kt = 0; kt < (H >> 1); kt += blockDim.x) {
			asm volatile(
				"cp.async.ca.shared::cta.global.L2::no_allocate [%0], [%1], 16;\n\t"
				"cp.async.commit_group;\n\t"
				"cp.async.ca.shared::cta.global.L2::no_allocate [%0 + 4], [%1 + 16], 16;\n\t"
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
					
			for (int32_t h4 = kt + (threadIdx.x & 3); (h4 < kt + blockDim.x) && (kt + blockDim.x < (H >> 1)); h4 += 32) {				
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
		//swiglu + topk + writeout
	}	
}

//@TODO fix rmem alloc
__device__ __forceinline__ void router(
	uint32_t res,
	uint32_t ni,
	uint32_t* tKiw,
	int32_t K
) {

	uint32_t rmem[4] = {0u, 0u, 0u, 0u};

	#pragma unroll 2
	for (int32_t k4 = 0; k4 < K; k4 += 4) {
		
		asm volatile(
			"ld.cg.nc.global.v4.b32 {%0, %1, %2, %3}, [%1];\n\t"
			: "=r"(rmem[0]), "=r"(rmem[1]), "=r"(rmem[2]), "=r"(rmem[3])
			: "l"(
				(uint64_t)__cvta_global_to_generic(
					tKiw + (ni * K) + k4)
				)
		);
		
		#pragma unroll 4
		for (int32_t k = 0; k < 4; k++) {
			if (!(((uint16_t)rmem[k])^((uint16_t)blockIdx.x))) {
				res = rmem[k];
				return;
			}
		}
	}
}

__global__ __launch_bounds__(CTA, 2)
void xR38GFF1_bf16(
    uint32_t* W13,
    uint32_t* X,
	uint32_t* Xs,
    uint32_t* Y,
    uint32_t* tKwi,
    int32_t E, int32_t N, int32_t I, int32_t H, int32_t K
) {

	uint32_t rmem[38] = {0u};
	__shared__ __align__(16) uint32_t smem[8192];

	while ((threadIdx.x + (rmem[0] * blockDim.x) < N)) {

		router(
			rmem[1], 
			(threadIdx.x + (rmem[0] * blockDim.x)).
			&tKwi,
			K
		);

		for (int32_t np = 0; np < CTA; np++) {
			if ((threadIdx.x==np) && rmem[1]) {
				smem[rmem[3]] = ((rmem[1] & 0xffff'0000u) | (threadIdx.x + (rmem[0] * blockDim.x)));
			}
			__syncthreads();
			if (smem[rmem[3]]) {
				//@TODO refine
				asm volatile( 
					".reg .b32 a, b, c, d;\n\t"
					"ld.global.v4.u32 {a, b, c, d}, [%0];\n\t"
					"st.global.v4.u32 [%1], {a, b, c, d};\n\t
					: "l"(
						(uint64_t)__cvta_global_to_generic(
							Xs + (blockIdx.x) * (8 * H / 2) + (threadIdx.x << 5) + (rmem[3] << 2)
						)
					) 
					: "l"(
						(uint64_t)__cvta_global_to_generic(
							X + (smem[rmem[3]] & 0x0000'ffffu) * H + (threadIdx.x << 3)
						)
					)
				);
				rmem[3]++;
			}
			if (rmem[0]==8) {
				xR38FF1_bf16(W13, Xs, Y, smem);
				rmem[0] = 0u;
			}
		}
	}
}