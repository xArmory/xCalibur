//imports


__global__ __launch_bounds__(CTA, 2)
void xR31FF1_bf16(
    const __nv_bfloat16* W13,
    const __nv_bfloat16* X, 
    __nv_bfloat16* Y, 
    const uint16_t* tKw,
    const int32_t E, const int32_t N,  const int32_t K, 
    const int32_t H, 
    const int32_t I,   
){


}