__global__ void xr58ff1bf16(
    const __nv_bfloat16* W13, // [E, H >> 4, I2 << 4]
    const __nv_bfloat16* X, // [N, H]
    __nv_bfloat16* Y, // [E, N, I]
    const uint32_t* topk_hot_twiddle, // [E, N] (topkW, topkW)
    const int32_t E, const int32_t N, const int32_t Hd16, const int32_t I216,
    const int32_t K
){

    uint32_t rmem[48];
    __shared__ alignas(4) uint32_t smem[8192];

    

    
}