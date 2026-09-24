__global__ __launch_bounds(CTA, 2)
void xS31(
    const __nv_bfloat16* W13, // [E, H >> 4, I2 << 4]
    const __nv_bfloat16* X, // [N, H]
    __nv_bfloat16* Y, // [E, N, I]
    const uint16_t* tKi, // [E, (N/16)]
    const uint16_t* tKw, // [E, N+2] packed to be contiguous (padded to N), end offset (2 -> 32bit)
    const int32_t E, const int32_t N, const int32_t Hd16, const int32_t I216,
    const int32_t K
){

    uint32_t rmem[27];
    __shared__ __align__(4) uint32_t smem[8192];

    ldug_u32(
        tKw + (uint64_t)(blockIdx.x * (N+2)),
        &rmem[26]
    );

    if (!rmem[26]) return;

    rmem[25] = 0u;
    for (int32_t n = 0; (n << 4) < N && rmem[25] <= rmem[26]; n++) {

        ldug_u32(
            tKi + (uint64_t)((blockIdx.x * (N >> 4)) + (n16)),
            &rmem[24]
        );

        rmem[25] += __popc(rmem[24]);
        

        if (__popc(rmem[24]) > 7)
        


    }
       

    
}