#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define f322b(x) __float_as_uint(x)
#define u162bf16(x) __ushort_as_bfloat16(x)

__device__ __forceinline__ uint32_t softmax_bf16x2(
    uint32_t x, float m = 0.0f
){
    float w = __uint_as_float(x << 16) - m;
    float z = __uint_as_float(x & 0xffff'0000u) - m;
    float y = 1.4426950408889634f;
    asm volatile(
        "mul.f32 %0, %0, %2;\n\t"
        "mul.f32 %1, %1, %2;\n\t"
        "ex2.approx.ftz.f32 %0, %0;\n\t"
        "ex2.approx.ftz.f32 %1, %1;\n\t"
        : "+&f"(w), "+&f"(z)
        : "f"(y)
    );
    __nv_bfloat162_raw tmp = __floats2bfloat162_rn(w, z);
    return uint32_t(tmp.x) | (uint32_t(tmp.y) << 16);
}

__device__ __forceinline__ void add_bf16x2(
    uint32_t& x, uint32_t y
){
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "fma.rn.bf16x2 %0, %0, %2, %1;\n\t"
        : "+r"(x)
        : "r"(y), "r"(0x3f80'3f80u)
    );
#else
    float w = __uint_as_float(x << 16) + __uint_as_float(y << 16);
    float z = __uint_as_float(x & 0xffff'0000u) + __uint_as_float(y & 0xffff'0000u);
    __nv_bfloat162_raw tmp = __floats2bfloat162_rn(w, z);
    x = uint32_t(tmp.x) | (uint32_t(tmp.y) << 16);
#endif
}

__device__ __forceinline__ float rcp_f32(
    float x
){
    asm volatile(
        "rcp.approx.ftz.f32 %0, %0;\n\t"
        : "+f"(x)
    );
    return x;
}

__device__ __forceinline__ void rcp_bf16x2(
    uint32_t& x
){
    float w = __uint_as_float(x << 16);
    float z = __uint_as_float(x & 0xffff'0000u);
    asm volatile(
        "rcp.approx.ftz.f32 %0, %0;\n\t"
        "rcp.approx.ftz.f32 %1, %1;\n\t"
        : "+&f"(w), "+&f"(z)
    );
    __nv_bfloat162_raw tmp = __floats2bfloat162_rn(w, z);
    x = uint32_t(tmp.x) | (uint32_t(tmp.y) << 16);
}

__device__ __forceinline__ void add_bf16x2x1(
    uint32_t x, uint32_t& y
){
    uint32_t tmp = x << 16;
    add_bf16x2(tmp, x & 0xffff'0000u);
    add_bf16x2(y, tmp & 0xffff'0000u);
}

__device__ __forceinline__ uint32_t softmax_mul(
    uint32_t x, uint32_t y
){
    uint32_t idx = x & 0x0000'ffffu;
    x &= 0xffff'0000u;
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "mul.bf16x2 %0, %0, %1;\n\t"
        : "+r"(x)
        : "r"(y)
    );
#else
    float w = __uint_as_float(x) * __uint_as_float(y & 0xffff'0000u);
    x = uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(w))) << 16;
#endif
    return (x & 0xffff'0000u) | idx;
}

__device__ __forceinline__ uint32_t softmax_mul(
    uint32_t x, float y
){
    float w = __uint_as_float(x & 0xffff'0000u) * y;
    return (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(w))) << 16)
        | (x & 0x0000'ffffu);
}

__device__ __forceinline__ void mul_bf16x2(
    uint32_t& x, uint32_t y
){
    asm volatile(
        "fma.rn.bf16x2 %0, %0, %1, %2;\n\t"
        : "+r"(x)
        : "r"(y), "r"(0x8000'8000u)
    );
}

__device__ __forceinline__ void swiglu_topkw_bf16x2(
    uint32_t& x, uint32_t y, uint32_t w
){
    uint32_t t = softmax_bf16x2((x & 0x7fff'7fffu) | 0x8000'8000u);
    uint32_t s = t;
    add_bf16x2(s, 0x3f80'3f80u);
    rcp_bf16x2(s);
    t = ((x & 0x0000'8000u) ? (t & 0x0000'ffffu) : 0x0000'3f80u)
        | ((x & 0x8000'0000u) ? (t & 0xffff'0000u) : 0x3f80'0000u);
    mul_bf16x2(s, t);
    mul_bf16x2(x, s);
    mul_bf16x2(x, y);
    mul_bf16x2(x, w);
}

__device__ __forceinline__ uint32_t cvt_bf16x2_f32(
    uint32_t x, uint32_t y
){
    asm volatile(
        "cvt.rn.bf16x2.f32 %0, %2, %1;\n\t"
        : "=r"(x)
        : "r"(x), "r"(y)
    );
    return x;
}

__device__ __forceinline__ void ldcg_b16(
    const void* src,
    uint16_t& dst
){
    asm volatile(
        "ld.global.cg.b16 %0, [%1];\n\t"
        : "=h"(dst)
        : "l"((uint64_t)__cvta_generic_to_global(src))
        : "memory"
    );
}

__device__ __forceinline__ void ldcg_b32(
    const void* src,
    uint32_t& dst
){
    asm volatile(
        "ld.global.cg.b32 %0, [%1];\n\t"
        : "=r"(dst)
        : "l"((uint64_t)__cvta_generic_to_global(src))
        : "memory"
    );
}

__device__ __forceinline__ void ldcg_b32v4(
    const void* src,
    uint32_t* dst
){
    asm volatile(
#if __CUDA_ARCH__ >= 750
        "ld.global.cg.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];\n\t"
#else
        "ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];\n\t"
#endif
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "l"((uint64_t)__cvta_generic_to_global(src))
        : "memory"
    );
}

__device__ __forceinline__ void ldnc_evict_last_b32v4(
    const void* src,
    uint32_t* dst
){
    asm volatile(
        "{\n\t"
        ".reg .b64 policy;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n\t"
        "ld.global.nc.L2::cache_hint.L2::256B.v4.b32 {%0, %1, %2, %3}, [%4], policy;\n\t"
        "}\n\t"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "l"((uint64_t)__cvta_generic_to_global(src))
        : "memory"
    );
}

__device__ __forceinline__ void ldcg_b16x8(
    const __nv_bfloat16* src,
    uint32_t* dst,
    int32_t size
){
    if (size == 8 && !(uint64_t(src) & 15u)) {
        ldcg_b32v4(src, dst);
        return;
    }
    #pragma unroll 4
    for (int32_t j = 0; j < 4; j++) {
        uint16_t tmp;
        dst[j] = 0u;
        if ((j << 1) < size) {
            ldcg_b16(src + (j << 1), tmp);
            dst[j] = tmp;
        }
        if (((j << 1) + 1) < size) {
            ldcg_b16(src + (j << 1) + 1, tmp);
            dst[j] |= uint32_t(tmp) << 16;
        }
    }
}

__device__ __forceinline__ void stg_b32(
    void* dst,
    uint32_t x
){
    asm volatile(
        "st.global.b32 [%0], %1;\n\t"
        :
        : "l"((uint64_t)__cvta_generic_to_global(dst)), "r"(x)
        : "memory"
    );
}

__device__ __forceinline__ void stg_b32v4(
    void* dst,
    uint32_t a, uint32_t b, uint32_t c, uint32_t d
){
    asm volatile(
        "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n\t"
        :
        : "l"((uint64_t)__cvta_generic_to_global(dst)),
          "r"(a), "r"(b), "r"(c), "r"(d)
        : "memory"
    );
}

__device__ __forceinline__ void cp_gmem_b32v4(
    const void* src,
    void* dst
){
    asm volatile(
        "{\n\t"
        ".reg .b32 a, b, c, d;\n\t"
        "ld.global.cg.v4.b32 {a, b, c, d}, [%0];\n\t"
        "st.global.v4.b32 [%1], {a, b, c, d};\n\t"
        "}\n\t"
        :
        : "l"((uint64_t)__cvta_generic_to_global(src)),
          "l"((uint64_t)__cvta_generic_to_global(dst))
        : "memory"
    );
}

__device__ __forceinline__ void cp_smem_cs_b32v4(
    const void* src,
    uint32_t* dst,
    bool valid = true
){
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 a, b, c, d;\n\t"
        "mov.b32 a, 0;\n\t"
        "mov.b32 b, 0;\n\t"
        "mov.b32 c, 0;\n\t"
        "mov.b32 d, 0;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "@p ld.global.cs.v4.b32 {a, b, c, d}, [%1];\n\t"
        "st.shared.v4.b32 [%0], {a, b, c, d};\n\t"
        "}\n\t"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(dst)),
          "l"((uint64_t)__cvta_generic_to_global(src)),
          "r"(uint32_t(valid))
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_ca_b32v4(
    const void* src,
    uint32_t* dst,
    int32_t size = 16
){
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n\t"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(dst)),
          "l"((uint64_t)__cvta_generic_to_global(src)),
          "r"(size)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit_group(){
    asm volatile(
        "cp.async.commit_group;\n\t"
        :
        :
        : "memory"
    );
}

template <int32_t N>
__device__ __forceinline__ void cp_async_wait_group(){
    static_assert(N >= 0, "cp.async wait group must be nonnegative");
    asm volatile(
        "cp.async.wait_group %0;\n\t"
        :
        : "n"(N)
        : "memory"
    );
}

__device__ __forceinline__ void ldmatrix_b16x4(
    const void* src,
    uint32_t* dst
){
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n\t"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"((uint32_t)__cvta_generic_to_shared(src))
        : "memory"
    );
}

template <int32_t fsel = 0>
__device__ __forceinline__ void mma_sp_m16n8k32_bf16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    const uint32_t* b,
    uint32_t* c,
    uint32_t e
){
    static_assert(fsel == 0 || fsel == 1, "mma.sp fsel must be 0 or 1");
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%0, %1, %2, %3}, %12, %13;\n\t"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(e), "n"(fsel)
    );
}
