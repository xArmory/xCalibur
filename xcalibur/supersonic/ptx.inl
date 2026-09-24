

__device__ __forceinline__ void ldug_u32(
    const void* src,
    uint32_t* dst
){
    asm volatile("ldu.global.u32 %0, [%1];"
        : "=r"(dst[0])
        : "l"(
            (uint64_t)__cvta_generic_to_global(
              src
            )
        )
    );
}