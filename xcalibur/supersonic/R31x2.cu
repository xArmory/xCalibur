__global__ __launch_bounds__(CTA, 2)
void xS31(
    const __nv_bfloat16* W13, // [E, H >> 4, I2 << 4]
    const __nv_bfloat16* X, // [N, H]
    __nv_bfloat16* Y, // [E, N, I]
    const uint16_t* tKi, // [E, (N/16)]
    const uint16_t* tKw, // [E, N+2] packed to be contiguous (padded to N), end offset (2 -> 32bit)
    const int32_t E, const int32_t N, const int32_t Hd16, const int32_t I216,
    const int32_t K
){

    uint32_t rmem[28];
    __shared__ __align__(4) uint32_t smem[8192];

    ldug_u32(
        tKw + (uint64_t)(blockIdx.x * (N+2)),
        &rmem[27]
    );

    if (!rmem[27]) return;
    
}