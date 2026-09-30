#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "ptx.inl"

#define CTA 768

__device__ __forceinline__ void xR38F1Epilogue_bf16(
	uint32_t* Y,
	uint32_t* rmem,
	uint32_t* smem,
	int32_t i, int32_t I
){
	if (i + ((threadIdx.x >> 5) << 3) >= I) return;

	if (i + (threadIdx.x >> 2) < I) {
		rmem[20] = cvt_bf16x2_f32(rmem[20], rmem[21]);
		rmem[22] = cvt_bf16x2_f32(rmem[22], rmem[23]);
		rmem[21] = ((smem[(threadIdx.x & 3) << 1] == 0xffff'ffffu)
			? 0u : (smem[(threadIdx.x & 3) << 1] >> 16))
			| ((smem[((threadIdx.x & 3) << 1) + 1] == 0xffff'ffffu)
			? 0u : (smem[((threadIdx.x & 3) << 1) + 1] & 0xffff'0000u));
		swiglu_topkw_bf16x2(rmem[20], rmem[22], rmem[21]);
	} else rmem[20] = 0u;

	rmem[22] = __shfl_xor_sync(0xffff'ffffu, rmem[20], 4);
	rmem[20] = (threadIdx.x & 4) ? ((rmem[22] >> 16) | (rmem[20] & 0xffff'0000u))
		: ((rmem[20] & 0x0000'ffffu) | (rmem[22] << 16));
	stg_b32(
		Y + (uint64_t((i >> 3) + (threadIdx.x >> 5)) << 5)
		+ ((threadIdx.x & 3) << 3) + (threadIdx.x & 4) + ((threadIdx.x & 31) >> 3),
		rmem[20]
	);
}

__device__ __forceinline__ void xR38F1Compute_bf16(
	uint32_t* W13,
	uint32_t* Xs,
	uint32_t* Y,
	uint32_t* rmem,
	uint32_t* smem,
	int32_t E, int32_t N, int32_t I, int32_t H
){
	__syncthreads();
	if (threadIdx.x < 8) {
		stg_b32(Y + threadIdx.x, (smem[threadIdx.x] == 0xffff'ffffu)
			? 0xffff'ffffu : (smem[threadIdx.x] & 0x0000'ffffu));
	}

	for (int32_t i = 0; i < I; i += (blockDim.x >> 2)) {
		#pragma unroll 4
		for (int32_t j = 20; j < 24; j++) rmem[j] = 0u;

		for (int32_t kt = 0; kt < (H >> 1); kt += blockDim.x) {
			cp_smem_cs_b32v4(
				Xs + (uint64_t(blockIdx.x) * (H << 2))
				+ ((kt + threadIdx.x < (H >> 1)) ? ((kt + threadIdx.x) << 3) : 0),
				smem + 8 + (threadIdx.x << 3),
				kt + threadIdx.x < (H >> 1)
			);
			cp_smem_cs_b32v4(
				Xs + (uint64_t(blockIdx.x) * (H << 2))
				+ ((kt + threadIdx.x < (H >> 1)) ? ((kt + threadIdx.x) << 3) : 0) + 4,
				smem + 12 + (threadIdx.x << 3),
				kt + threadIdx.x < (H >> 1)
			);
			__syncthreads();

			for (int32_t h4 = (kt << 1); h4 < H && h4 < ((kt + blockDim.x) << 1); h4 += 64) {
				#pragma unroll 16
				for (int32_t j = 0; j < 16; j++) rmem[j] = 0u;

				if (i + (threadIdx.x >> 2) < I) {
					if (h4 + ((threadIdx.x & 3) << 2) < H) {
						ldnc_evict_last_b32v4(
							W13 + ((uint64_t(blockIdx.x) * I + i + (threadIdx.x >> 2)) * H)
							+ h4 + ((threadIdx.x & 3) << 2), rmem
						);
					}
					if (h4 + 16 + ((threadIdx.x & 3) << 2) < H) {
						ldnc_evict_last_b32v4(
							W13 + ((uint64_t(blockIdx.x) * I + i + (threadIdx.x >> 2)) * H)
							+ h4 + 16 + ((threadIdx.x & 3) << 2), rmem + 4
						);
					}
					if (h4 + 32 + ((threadIdx.x & 3) << 2) < H) {
						ldnc_evict_last_b32v4(
							W13 + ((uint64_t(blockIdx.x) * I + i + (threadIdx.x >> 2)) * H)
							+ h4 + 32 + ((threadIdx.x & 3) << 2), rmem + 8
						);
					}
					if (h4 + 48 + ((threadIdx.x & 3) << 2) < H) {
						ldnc_evict_last_b32v4(
							W13 + ((uint64_t(blockIdx.x) * I + i + (threadIdx.x >> 2)) * H)
							+ h4 + 48 + ((threadIdx.x & 3) << 2), rmem + 12
						);
					}
				}

				ldmatrix_b16x4(
					smem + 8 + ((h4 - (kt << 1)) << 2) + ((threadIdx.x & 31) << 2),
					rmem + 16
				);
				mma_sp_m16n8k32_bf16<0>(
					rmem[0], rmem[2], rmem[4], rmem[6],
					rmem + 16, rmem + 20, 0x44444444u
				);
				mma_sp_m16n8k32_bf16<0>(
					rmem[1], rmem[3], rmem[5], rmem[7],
					 rmem + 16, rmem + 20, 0xEEEEEEEEu
				);
				ldmatrix_b16x4(
					smem + 8 + ((h4 + 32 - (kt << 1)) << 2) + ((threadIdx.x & 31) << 2),
					rmem + 16
				);
				mma_sp_m16n8k32_bf16<1>(
					rmem[8], rmem[10], rmem[12], rmem[14],
					rmem + 16, rmem + 20, 0x44444444u
				);
				mma_sp_m16n8k32_bf16<1>(
					rmem[9], rmem[11], rmem[13], rmem[15],
					rmem + 16, rmem + 20, 0xEEEEEEEEu
				);
			}
			__syncthreads();
		}
		xR38F1Epilogue_bf16(Y + 8, rmem, smem, i, I);
	}
}

__device__ __forceinline__ void xR38F1Route(
	uint32_t& res,
	uint32_t ni,
	uint32_t* tKiw,
	uint32_t* rmem,
	int32_t K
){

	res = 0xffff'ffffu;

	for (int32_t k4 = 0; k4 < K; k4 += 4) {
		if (!(K & 3)) {
			ldcg_b32v4(tKiw + (uint64_t(ni) * K) + k4, rmem);
		} else {
			for (int32_t k = 0; k < 4; k++) {
				if (k4 + k < K) ldcg_b32(tKiw + (uint64_t(ni) * K) + k4 + k, rmem[k]);
			}
		}
		for (int32_t k = 0; k < 4; k++) {
			if (k4 + k < K && ((rmem[k] & 0x0000'ffffu) ^ 0x0000'ffffu) == blockIdx.x) {
				res = rmem[k];
				return;
			}
		}
	}
}

__global__ __launch_bounds__(CTA, 2)
void xR38F1_bf16(
	uint32_t* W13,
	uint32_t* X,
	uint32_t* Xs,
	uint32_t* Y,
	uint32_t* tKwi,
	int32_t E, int32_t N, int32_t I, int32_t H, int32_t K
){

	if (blockIdx.x >= E || E > 65536 || N <= 0 || N > 65536 || I <= 0 || H <= 0 || (H & 7) || K <= 0 || K > E) return;
	if (blockDim.x != CTA || blockDim.y != 1 || blockDim.z != 1) return;

	uint32_t rmem[28] = {0u};
	__shared__ __align__(16) uint32_t smem[8192];

	Y += uint64_t(blockIdx.x) * (8 + ((uint64_t(N) + 7) >> 3) * (8 + (((uint64_t(I) + 7) >> 3) << 5)));

	while (rmem[0] * blockDim.x < N) {
		rmem[1] = 0xffff'ffffu;
		if (threadIdx.x + (rmem[0] * blockDim.x) < N) {
			xR38F1Route(rmem[1], threadIdx.x + (rmem[0] * blockDim.x), tKwi, rmem + 4, K);
		}

		for (int32_t np = 0; np < blockDim.x && np + (rmem[0] * blockDim.x) < N; np++) {
			if (threadIdx.x == np) {
				smem[rmem[3]] = (rmem[1] == 0xffff'ffffu) ? 0xffff'ffffu
					: ((rmem[1] & 0xffff'0000u) | (np + (rmem[0] * blockDim.x)));
			}
			__syncthreads();
			if (smem[rmem[3]] != 0xffff'ffffu) {
				for (int32_t h4 = threadIdx.x; h4 < (H >> 3); h4 += blockDim.x) {
					cp_gmem_b32v4(
						X + (uint64_t(smem[rmem[3]] & 0x0000'ffffu) * (H >> 1)) + (h4 << 2),
						Xs + (uint64_t(blockIdx.x) * (H << 2)) + (h4 << 5) + (rmem[3] << 2)
					);
				}
				rmem[3]++;
				if (rmem[3] == 8) {
					xR38F1Compute_bf16(
						W13, Xs, Y + 8 + uint64_t(rmem[2]) * (8 + (((uint64_t(I) + 7) >> 3) << 5)),
						rmem + 4, smem, E, N, I, H
					);
					rmem[2]++;
					rmem[3] = 0u;
				}
			}
			__syncthreads();
		}
		rmem[0]++;
	}
	if (rmem[3]) {
		if (threadIdx.x >= rmem[3] && threadIdx.x < 8) smem[threadIdx.x] = 0xffff'ffffu;
		for (int32_t h4 = threadIdx.x; h4 < (H >> 3); h4 += blockDim.x) {
			for (int32_t np = rmem[3]; np < 8; np++) {
				stg_b32v4(
					Xs + (uint64_t(blockIdx.x) * (H << 2)) + (h4 << 5) + (np << 2),
					0u, 0u, 0u, 0u
				);
			}
		}
		xR38F1Compute_bf16(
			W13, Xs, Y + 8 + uint64_t(rmem[2]) * (8 + (((uint64_t(I) + 7) >> 3) << 5)),
			rmem + 4, smem, E, N, I, H
		);
		rmem[2]++;
	}
	if (threadIdx.x < 8) stg_b32(Y + threadIdx.x, threadIdx.x ? 0u : rmem[2]);
}
