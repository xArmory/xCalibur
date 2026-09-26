#pragma once
#include <"ptx.inl">

__global__ __launch_bounds__(CTA, 2)
void xR38FF1_bf16(
    __nv_bfloat16* W13,
    __nv_bfloat16* X,
    __nv_bfloat16* Xs, // [8, H] (unless L2 prefetch works)
    __nv_bfloat16* Y,
    uint32_t* tKwi,
    int32_t E, int32_t N,  int32_t K, 
    int32_t H, 
    int32_t I 
){

    uint32_t rmem[38];
    __shared__ __align__(4) __nv_bfloat16 smem[16384];

    for (int32_t n = 0; n < N; n+=512) {

	    if (((threadIdx.x >> 1) + n) < N) {

			#pragma unroll 4
			for (int rk = 0; rk < 4; rk++) {
				rmem[33 + rk] = 0u;
			}

			for (int kwip = 0; kwip < (K >> 3); kwip++) {

				if (!rmem[37]) {
					ldgv4_u32(
					(uint64_t)__cvta_global_to_generic(tKwi + ((threadIdx.x >> 1) + n) * K +((threadIdx.x & 1) << 2)), 
						&rmem[33]
					);
				
					#pragma unroll 4
					for (int kwi = 0; kwi < 4; kwi++) {
						if ((rmem[33 + kwi]) & 0xffff'0000u)==(blockIdx.x) {
							rmem[37] = rmem[33 + kwi];
							break;
						}
					}
				}
			}
			
			rmem[37] = max(rmem[37], __shfl_xor_sync(0xffff'ffffu, rmem[37], 1, 2));
		    
			//@TODO spread loads
			if (rmem[37]) {
				ldgv4_u32(
					(uint64_t)__cvta_global_to_generic(
						X + ((threadIdx.x >> 1) + n) * H +((threadIdx.x & 1) << 2)), 
						&rmem[33]
					);
			}
	    }
    }
}