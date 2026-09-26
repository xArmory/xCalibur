

__device__ __forceinline__ void ldgv4_u32(
    const void* src,
    uint32_t* dst
){
    asm volatile("ld.cg.gpu.aquire.global.v4.u32.L1::no_allocate {%0, %1, %2, %3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])

        : "l"(
            (uint64_t)__cvta_generic_to_global(
              src
            )
        )
    );
}
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
