__global__ xR58_bf16(
    const __nv_bfloat16* W13, // [E, 2I, H] -> [E, H / k, 2I * k]
    const __nv_bfloat16* X, // [N, H]
    __nv_bfloat16* Y, // [E, N, I]
    const uint32_t* topk_twiddle, // [N, K]
){

}