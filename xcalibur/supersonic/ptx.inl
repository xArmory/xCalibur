#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf(float x){ return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ex(float x){
	x *= 1.4426950408889634f;
	asm("ex2.approx.f32 %0,%0;" : "+f"(x));
	return x;
}
__device__ __forceinline__ float divf(float x, float y){
	asm("div.full.f32 %0,%1,%2;" : "=f"(x) : "f"(x), "f"(y));
	return x;
}
__device__ __forceinline__ float swiglu(float g, float u){
	g=bf(g); u=bf(u);
	float t=bf(ex(-fabsf(g))), s=bf(divf(1.f,bf(1.f+t)));
	s=bf(s*(g<0 ? t : 1.f));
	return bf(bf(g*s)*u);
}
__device__ __forceinline__ uint32_t pack2(float a, float b){
	uint32_t x;
	asm("cvt.rn.bf16x2.f32 %0,%2,%1;" : "=r"(x) : "f"(a), "f"(b));
	return x;
}
__device__ __forceinline__ uint32_t mul2(uint32_t x, float w){
	uint32_t y=pack2(w,w);
	asm("fma.rn.bf16x2 %0,%1,%2,%3;" : "=r"(x) : "r"(x), "r"(y), "r"(0x80008000u));
	return x;
}
__device__ __forceinline__ uint32_t add2(uint32_t x, uint32_t y){
	asm("fma.rn.bf16x2 %0,%1,%2,%3;" : "=r"(x) : "r"(x), "r"(0x3f803f80u), "r"(y));
	return x;
}
__device__ __forceinline__ int ld(const int* p, bool valid, int x=0){
	asm volatile("{ .reg .pred p; setp.ne.u32 p,%2,0; @p ld.global.b32 %0,[%1]; }" : "+r"(x) : "l"(p),"r"(int(valid)) : "memory");
	return x;
}
__device__ __forceinline__ float ld(const float* p, bool valid){
	float x=0;
	asm volatile("{ .reg .pred p; setp.ne.u32 p,%2,0; @p ld.global.f32 %0,[%1]; }" : "+f"(x) : "l"(p),"r"(int(valid)) : "memory");
	return x;
}
__device__ __forceinline__ void store2(uint32_t* p, uint32_t x, bool valid){
	asm volatile("{ .reg .pred p; setp.ne.u32 p,%2,0; @p st.global.b32 [%0],%1; }" :: "l"(p),"r"(x),"r"(int(valid)) : "memory");
}
__device__ __forceinline__ void reduce2(uint32_t* p, uint32_t x, bool valid=true){
	asm volatile("{ .reg .pred p; .reg .b32 old,next,assumed; .reg .b64 policy;\n"
		"setp.ne.u32 p,%3,0; @!p bra done;\n"
		"createpolicy.fractional.L2::evict_last.b64 policy,1.0;\n"
		"ld.global.cg.L2::cache_hint.b32 old,[%0],policy;\n"
		"loop: mov.b32 assumed,old; fma.rn.bf16x2 next,old,%2,%1;\n"
		"atom.relaxed.gpu.global.cas.b32 old,[%0],assumed,next;\n"
		"setp.ne.b32 p,old,assumed; @p bra loop; done: }"
		:: "l"(p), "r"(x), "r"(0x3f803f80u), "r"(int(valid)) : "memory");
}
__device__ __forceinline__ int sw(int r, int k){ return r*64+(k^((r&7)<<3)); }

template <bool ALIGNED=true>
__device__ __forceinline__ void cp(bf16* dst, const bf16* src, int size){
	if (ALIGNED || (uintptr_t(src)&15)==0) {
		asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(uint32_t(__cvta_generic_to_shared(dst))),"l"(src),"r"(size*2) : "memory");
	} else {
		#pragma unroll
		for (int j=0;j<8;j++) dst[j]=j<size ? src[j] : __float2bfloat16_rn(0.f);
	}
}
__device__ __forceinline__ void commit(){ asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int G>
__device__ __forceinline__ void wait(){ asm volatile("cp.async.wait_group %0;" :: "n"(G) : "memory"); }
__device__ __forceinline__ void ldm4(uint32_t* x, const bf16* p){
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
		: "=r"(x[0]),"=r"(x[1]),"=r"(x[2]),"=r"(x[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void ldm2(uint32_t* x, const bf16* p){
	asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
		: "=r"(x[0]),"=r"(x[1]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b){
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
		: "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
		: "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

template <int M, int O, int W>
__device__ __forceinline__ void compute(const bf16* A,const bf16* B,float* c){
	constexpr int WM=(M/16 < (W>=16 ? 4:2) ? M/16 : (W>=16 ? 4:2)), WN=W/WM;
	constexpr int RM=M/(WM*16), RN=O/(WN*8);
	int lane=threadIdx.x&31,warp=threadIdx.x>>5,m=(warp/WN)*16,n=(warp%WN)*8;
	uint32_t a[RM*16],b[RN*8];
	#pragma unroll
	for (int k=0;k<4;k++) {
		#pragma unroll
		for (int j=0;j<RM;j++) ldm4(a+4*(k*RM+j),A+sw(m+j*WM*16+(lane&15),16*k+(lane>>4)*8));
	}
	#pragma unroll
	for (int k=0;k<2;k++) {
		#pragma unroll
		for (int j=0;j<RN;j++) {
			uint32_t v[4];ldm4(v,B+sw(n+j*WN*8+(lane&7),32*k+(lane>>3)*8));
			b[2*(2*k*RN+j)]=v[0];b[2*(2*k*RN+j)+1]=v[1];
			b[2*((2*k+1)*RN+j)]=v[2];b[2*((2*k+1)*RN+j)+1]=v[3];
		}
	}
	#pragma unroll
	for (int k=0;k<4;k++) {
		#pragma unroll
		for (int i=0;i<RM;i++) {
			#pragma unroll
			for (int j=0;j<RN;j++) mma(c+4*(i*RN+j),a+4*(k*RM+i),b+2*(k*RN+j));
		}
	}
}
