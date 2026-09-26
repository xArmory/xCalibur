#pragma once
#include <"ptx.inl">

__global__ __launch_bounds__(CTA, 2)
void xR31FF1_bf16(
    const __nv_bfloat16* W13,
    const __nv_bfloat16* X,
    __nv_bfloat16* Xs, // [8, H] (unless L2 prefetch works)
    __nv_bfloat16* Y,
    const uint32_t* tKwi,
    const int32_t E, const int32_t N,  const int32_t K, 
    const int32_t H, 
    const int32_t I 
){

    uint32_t rmem[24];
    __shared__ __align__(4) __nv_bfloat16 smem[24576];

    for (int32_t n = 0; n < N; n+=512) {

	    if (((threadIdx.x >> 1) + n) < N) {
		    ldgv4_u32((uint64_t)__cvta_global_to_generic(tKwi + ((threadIdx.x >> 1) + n) * K +((threadIdx.x & 1) << 2)),  
				    &rmem[19]
				    );

		    #pragma unroll 4
		    for (int kwi = 0; kwi < K; kwi++) {
			    if ((rmem[19 + kwi]) & 0xffff'0000u)==(blockIdx.x) {
				    rmem[19] = rmem[19 + kwi]
				    break;
			    }
		    }

		    rmem[20] = __shfl_xor_sync(0xffff'ffffu, rmem[19], 1, 2);
		    rmem[19] = max(rmem[19], rmem[20]) & 0xffff'0000u;

		    //prefetch / load to global scratch Xs
	    }
    }


}
