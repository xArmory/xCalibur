#include "ptx.inl"
#include <type_traits>

constexpr int N=MOE_N,E=MOE_E,K=MOE_K,H=MOE_H,I=MOE_I,S=MOE_S;
constexpr int M1=MOE_M1,B1=MOE_B1,W1=MOE_W1,M2=MOE_M2,B2=MOE_B2,W2=MOE_W2;
constexpr bool H4=MOE_H4, ATOMIC=MOE_ATOMIC;
constexpr int C1=W1*32,C2=W2*32, CR=S>=2048 ? 512:128;
constexpr int SM1=4*((2*B1+M1)*64>S+W1 ? (2*B1+M1)*64:S+W1),SM2=4*(M2+B2)*64;
using idx=std::conditional_t<(int64_t(E)*2*I*((H+63)/64)*64<2147483648ll && int64_t(E)*((I+63)/64)*H*64<2147483648ll && int64_t(N)*K*(H>I ? H:I)<2147483648ll),int,int64_t>;

template <bool SUM>
__device__ __forceinline__ float blockf(float x, float* s){
	int t=threadIdx.x;
	#pragma unroll
	for (int d=16;d;d>>=1) {
		float y=__shfl_xor_sync(0xffffffff,x,d);
		x=SUM ? x+y : fmaxf(x,y);
	}
	if (!(t&31)) s[t>>5]=x;
	__syncthreads();
	x=SUM ? (s[0]+s[2])+(s[1]+s[3]) : fmaxf(fmaxf(s[0],s[2]),fmaxf(s[1],s[3]));
	__syncthreads();
	return x;
}

__global__ void k1(const bf16* L, float* Q, int* R, bf16* Z, bool clear){
	constexpr int V=(E+127)/128;
	__shared__ float s[4];
	__shared__ uint32_t keys[4];
	int t=threadIdx.x,n=blockIdx.x;
	float x[V],m=-CUDART_INF_F,total=0.f;
	uint32_t key[V];
	#pragma unroll
	for (int j=0;j<V;j++) { x[j]=t+j*128<E ? float(L[n*E+t+j*128]) : -CUDART_INF_F; m=fmaxf(m,x[j]); }
	m=blockf<false>(m,s);
	#pragma unroll
	for (int j=0;j<V;j++) { x[j]=ex(x[j]-m); total+=x[j]; }
	total=blockf<true>(total,s);
	#pragma unroll
	for (int j=0;j<V;j++) key[j]=t+j*128<E ? (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(divf(x[j],total))))<<16)|(65535-t-j*128) : 0;
	total=0.f;
	#pragma unroll
	for (int k=0;k<K;k++) {
		uint32_t v=0;
		#pragma unroll
		for (int j=0;j<V;j++) v=max(v,key[j]);
		v=__reduce_max_sync(0xffffffff,v);
		if (!(t&31)) keys[t>>5]=v;
		__syncthreads();
		v=max(max(keys[0],keys[2]),max(keys[1],keys[3]));
		__syncthreads();
		float weight=__uint_as_float(v&0xffff0000u);
		if (!t) { Q[n*K+k]=weight; R[n*K+k]=65535-int(v&65535); }
		total+=weight;
		#pragma unroll
		for (int j=0;j<V;j++) if (key[j]==v) key[j]=0;
	}
	__syncthreads();
	for (int k=t;k<K;k+=128) Q[n*K+k]=bf(divf(Q[n*K+k],total));
	if (clear) for (int h=t;h<H;h+=128) Z[n*H+h]=__float2bfloat16_rn(0.f);
}

template <int CTA>
__device__ __forceinline__ int compact(const int* R, int* ids, int* counts, int e, int begin, int end){
	int t=threadIdx.x,used=0;
	for (int n=begin;n<end;n+=CTA) {
		int q=N*K;
		#pragma unroll
		for (int k=0;k<K;k++) if (n+t<N && n+t<end && R[(n+t)*K+k]==e) q=(n+t)*K+k;
		uint32_t mask=__ballot_sync(0xffffffff,q<N*K);
		if (!(t&31)) counts[t>>5]=__popc(mask);
		__syncthreads();
		int offset=used+__popc(mask&((1u<<(t&31))-1));
		#pragma unroll
		for (int w=0;w<CTA/32;w++) { if (w<t/32) offset+=counts[w]; used+=counts[w]; }
		if (q<N*K) ids[offset]=q;
		__syncthreads();
	}
	return used;
}

template <int PART>
__device__ __forceinline__ void load1(bf16* s, const bf16* X, const bf16* W, const int* qload, int e, int p){
	int t=threadIdx.x,h=p*64;
	if constexpr(PART==0) {
	#pragma unroll
	for (int v=0;v<(2*B1*8+C1-1)/C1;v++) {
		int j=t+v*C1;if constexpr(2*B1*8%C1) if (j>=2*B1*8) continue;
		int r=j/8,k=(j&7)*8,i=blockIdx.x*B1+r/16*8+r%8;
		int valid=(I%B1==0 || i<I) ? (H%64==0 ? 8:max(0,min(8,H-h-k))) : 0;
		bf16* dst=s+sw(r,k);
		if constexpr(H4) {
			#pragma unroll 1
			for (int a=0;a<8;a++) dst[a]=a<valid ? W[((idx(e)*I+i)*((H+3)/4)+(h+k+a)/4)*8+(r%16/8)*4+(h+k+a)%4] : __float2bfloat16_rn(0.f);
		} else {
			idx off=((idx(e)*((H+63)/64)+p)*(2*I)+blockIdx.x*2*B1+r)*64+k;
			cp<true>(dst,W+(valid ? off : 0),valid);
		}
	}
	} else {
	#pragma unroll
	for (int v=0;v<(M1*8+C1-1)/C1;v++) {
		int j=t+v*C1;if constexpr(M1*8%C1) if (j>=M1*8) continue;
		int r=j/8,k=(j&7)*8,q=qload[v];
		int valid=q<N*K ? (H%64==0 ? 8:max(0,min(8,H-h-k))) : 0;
		cp<H%8==0>(s+2*B1*64+sw(r,k),X+(valid ? idx(q/K)*H+h+k : 0),valid);
	}
	}
}

__global__ __launch_bounds__(C1) void f1(const bf16* X,const bf16* W,bf16* Y,const int* R){
	extern __shared__ __align__(16) bf16 s[];
	int* ids=reinterpret_cast<int*>(s);
	int e=N==1 ? R[blockIdx.y] : blockIdx.y;
	int count;
	if constexpr(N==1) { if (!threadIdx.x) ids[0]=blockIdx.y; count=1; __syncthreads(); }
	else count=compact<C1>(R,ids,ids+S,e,blockIdx.z*S,min(N,int(blockIdx.z+1)*S));
	if (!count) return;
	constexpr int A=2*B1,WM=(A/16<(W1>=16 ? 4:2) ? A/16:(W1>=16 ? 4:2)),WN=W1/WM;
	constexpr int RM=A/(WM*16),RN=M1/(WN*8),P=(H+63)/64;
	int lane=threadIdx.x&31,warp=threadIdx.x>>5;
	int saved[(S+C1-1)/C1];
	#pragma unroll
	for (int j=0;j<(S+C1-1)/C1;j++) saved[j]=threadIdx.x+j*C1<count ? ids[threadIdx.x+j*C1] : N*K;
	__syncthreads();
	for (int batch=0;batch<count;batch+=M1) {
		#pragma unroll
		for (int j=0;j<(S+C1-1)/C1;j++) {
			int k=threadIdx.x+j*C1-batch;
			if (k>=0 && k<M1) ids[k]=saved[j];
		}
		__syncthreads();
		int qload[(M1*8+C1-1)/C1],qout[2*RN];
		#pragma unroll
		for (int j=0;j<(M1*8+C1-1)/C1;j++) { int r=(threadIdx.x+j*C1)/8; qload[j]=r<M1 && batch+r<count ? ids[r] : N*K; }
		#pragma unroll
		for (int n=0;n<RN;n++) {
			int j=(warp%WN+n*WN)*8+2*(lane&3);
			qout[2*n]=batch+j<count ? ids[j] : N*K;qout[2*n+1]=batch+j+1<count ? ids[j+1] : N*K;
		}
		__syncthreads();
		float c[RM*RN*4]={};
		load1<0>(s,X,W,qload,e,0);commit();load1<1>(s,X,W,qload,e,0);commit();
		if constexpr(P>1) load1<0>(s+(A+M1)*64,X,W,qload,e,1);
		commit();
		if constexpr(P>1) load1<1>(s+(A+M1)*64,X,W,qload,e,1);
		commit();wait<2>();__syncthreads();
		for (int p=0;p<P;p++) {
			bf16* a=s+(p&1)*(A+M1)*64;
			compute<A,M1,W1>(a,a+A*64,c);__syncthreads();
			if (p+2<P) load1<0>(a,X,W,qload,e,p+2);
			commit();
			if (p+2<P) load1<1>(a,X,W,qload,e,p+2);
			commit();wait<2>();__syncthreads();
		}
		#pragma unroll
		for (int m=0;m<RM;m++) {
			#pragma unroll
			for (int n=0;n<RN;n++) {
				float* a=c+4*(m*RN+n);
				float y0=swiglu(a[0],a[2]),y1=swiglu(a[1],a[3]);
				float z0=__shfl_xor_sync(0xffffffff,y0,4),z1=__shfl_xor_sync(0xffffffff,y1,4);
				int i=blockIdx.x*B1+(warp/WN+m*WM)*8+(lane>>2);
				if (!(lane&4) && (I%B1==0 || i<I)) {
					if (qout[2*n]<N*K) *reinterpret_cast<uint32_t*>(Y+idx(qout[2*n])*I+i)=pack2(y0,z0);
					if (qout[2*n+1]<N*K) *reinterpret_cast<uint32_t*>(Y+idx(qout[2*n+1])*I+i)=pack2(y1,z1);
				}
			}
		}
	}
}

__global__ __launch_bounds__(CR) void routes(const int* R,int* T,int* C,int* D){
	__shared__ int hist[E],meta[3],counts[CR/32];
	int t=threadIdx.x,e=blockIdx.x;
	for (int j=t;j<E;j+=CR) hist[j]=0;
	__syncthreads();
	for (int q=t;q<N*K;q+=CR) atomicAdd(hist+R[q],1);
	__syncthreads();
	if (!t) {
		int base=0,start=0,tiles=0;
		for (int j=0;j<E;j++) { if (j<e) { base+=hist[j];start+=(hist[j]+M2-1)/M2; } tiles+=(hist[j]+M2-1)/M2; }
		meta[0]=base;meta[1]=start;meta[2]=hist[e];C[2*e]=base;C[2*e+1]=hist[e];
		if (!e) D[0]=tiles;
	}
	__syncthreads();
	for (int b=t;b*M2<meta[2];b+=CR) { int j=1+3*(meta[1]+b);D[j]=e;D[j+1]=meta[0]+b*M2;D[j+2]=min(M2,meta[2]-b*M2); }
	compact<CR>(R,T+meta[0],counts,e,0,N);
}

template <int PART>
__device__ __forceinline__ void load2(bf16* s,const bf16* Y,const bf16* W,const int* qload,int e,int p){
	int t=threadIdx.x;
	if constexpr(PART==0) {
	#pragma unroll
	for (int v=0;v<(M2*8+C2-1)/C2;v++) {
		int j=t+v*C2;if constexpr(M2*8%C2) if (j>=M2*8) continue;
		int r=j/8,k=(j&7)*8,q=qload[v];
		int valid=q<N*K && p<(I+63)/64 ? (I%64==0 ? 8:max(0,min(8,I-p*64-k))) : 0;
		cp<true>(s+sw(r,k),Y+(valid ? idx(q)*I+p*64+k : 0),valid);
	}
	} else {
	#pragma unroll
	for (int v=0;v<(B2*8+C2-1)/C2;v++) {
		int j=t+v*C2;if constexpr(B2*8%C2) if (j>=B2*8) continue;
		int r=j/8,k=(j&7)*8,h=blockIdx.x*B2+r;
		int valid=(p<(I+63)/64 && (H%B2==0 || h<H)) ? (I%64==0 ? 8:max(0,min(8,I-p*64-k))) : 0;
		idx off=((idx(e)*((I+63)/64)+p)*H+h)*64+k;
		cp<true>(s+M2*64+sw(r,k),W+(valid ? off : 0),valid);
	}
	}
}

__global__ __launch_bounds__(C2) void f2(const bf16* Y,const bf16* W,bf16* Z,const float* Q,const int* R,const int* T,const int* D){
	extern __shared__ __align__(16) bf16 s[];
	int job=blockIdx.y,e,base,count;
	if constexpr(N==1) { e=R[job];base=job;count=1; }
	else { if (job>=D[0]) return;e=D[1+3*job];base=D[2+3*job];count=D[3+3*job]; }
	constexpr int WM=(M2/16<(W2>=16 ? 4:2) ? M2/16:(W2>=16 ? 4:2)),WN=W2/WM;
	constexpr int RM=M2/(WM*16),RN=B2/(WN*8),P=(I+63)/64;
	int qload[(M2*8+C2-1)/C2];
	#pragma unroll
	for (int v=0;v<(M2*8+C2-1)/C2;v++) { int r=(threadIdx.x+v*C2)/8;qload[v]=N==1 ? (r<count ? base:N*K):ld(T+(r<count ? base+r:0),r<count,N*K); }
	float c[RM*RN*4]={};
	load2<1>(s,Y,W,qload,e,0);commit();load2<0>(s,Y,W,qload,e,0);commit();
	if constexpr(P>1) load2<1>(s+(M2+B2)*64,Y,W,qload,e,1);
	commit();
	if constexpr(P>1) load2<0>(s+(M2+B2)*64,Y,W,qload,e,1);
	commit();wait<2>();__syncthreads();
	#pragma unroll (P<=8 ? P:1)
	for (int p=0;p<P;p++) {
		bf16* a=s+(p&1)*(M2+B2)*64;
		compute<M2,B2,W2>(a,a+M2*64,c);__syncthreads();
		load2<1>(a,Y,W,qload,e,p+2);
		commit();
		load2<0>(a,Y,W,qload,e,p+2);
		commit();wait<2>();__syncthreads();
	}
	wait<0>();__syncthreads();
	int lane=threadIdx.x&31,warp=threadIdx.x>>5;
	#pragma unroll
	for (int m=0;m<RM;m++) {
		#pragma unroll
		for (int r=0;r<2;r++) {
			int j=(warp/WN+m*WM)*16+(lane>>2)+8*r;
			bool live=j<count;
			int q=N==1 ? base:ld(T+(live ? base+j:0),live);
			float weight=ld(Q+q,live);
			#pragma unroll
			for (int n=0;n<RN;n++) {
				int h=blockIdx.x*B2+(warp%WN+n*WN)*8+2*(lane&3);
				bool valid=live && (H%B2==0 || h<H);
				float* a=c+4*(m*RN+n)+2*r;
				uint32_t v=mul2(pack2(a[0],a[1]),weight);
				uint32_t* dst=reinterpret_cast<uint32_t*>(Z+(valid ? idx(ATOMIC ? q/K:q)*H+h:0));
				if constexpr(ATOMIC) reduce2(dst,v,valid);else store2(dst,v,valid);
			}
		}
	}
}

__global__ void sum2(const uint32_t* P,uint32_t* Z){
	int p=blockIdx.x*256+threadIdx.x;
	if (p>=N*(H/2)) return;
	int n=p/(H/2),h=p%(H/2);uint32_t a=0;
	#pragma unroll
	for (int k=0;k<K;k++) a=add2(a,P[(n*K+k)*(H/2)+h]);
	Z[p]=a;
}

extern "C" int init(){
	cudaError_t e=cudaFuncSetAttribute(f1,cudaFuncAttributeMaxDynamicSharedMemorySize,SM1);
	if (e!=cudaSuccess) return e;
	return cudaFuncSetAttribute(f2,cudaFuncAttributeMaxDynamicSharedMemorySize,SM2);
}
extern "C" const char* error(int status){ return cudaGetErrorString(cudaError_t(status)); }
extern "C" int clear(bf16* Z,cudaStream_t stream){ return cudaMemsetAsync(Z,0,int64_t(N)*H*2,stream); }
extern "C" int route(const bf16* L,float* Q,int* R,bf16* Z,bool clear,cudaStream_t stream){
	k1<<<N,128,0,stream>>>(L,Q,R,Z,clear);return cudaGetLastError();
}
extern "C" int first(const bf16* X,const bf16* W,bf16* Y,const int* R,cudaStream_t stream){
	f1<<<dim3((I+B1-1)/B1,N==1 ? K:E,N==1 ? 1:(N+S-1)/S),C1,SM1,stream>>>(X,W,Y,R);return cudaGetLastError();
}
extern "C" int finish(const bf16* Y,const bf16* W,bf16* Z,const float* Q,const int* R,int* T,int* C,int* D,bf16* P,cudaStream_t stream){
	if constexpr(N>1) routes<<<E,CR,0,stream>>>(R,T,C,D);
	f2<<<dim3((H+B2-1)/B2,N==1 ? K:(N*K+M2-1)/M2+E),C2,SM2,stream>>>(Y,W,ATOMIC ? Z:P,Q,R,T,D);
	if constexpr(!ATOMIC) sum2<<<(N*(H/2)+255)/256,256,0,stream>>>(reinterpret_cast<const uint32_t*>(P),reinterpret_cast<uint32_t*>(Z));
	return cudaGetLastError();
}
extern "C" int stats(int stage,int* out){
	const void* f=stage==0 ? (const void*)k1 : stage==1 ? (const void*)f1 : stage==2 ? (const void*)routes : stage==3 ? (const void*)f2 : (const void*)sum2;
	cudaFuncAttributes a;auto e=cudaFuncGetAttributes(&a,f);if (e!=cudaSuccess) return e;
	int dynamic=stage==1 ? SM1:stage==3 ? SM2:0,cta=stage==1 ? C1:stage==2 ? CR:stage==3 ? C2:stage==4 ? 256:128;
	out[0]=a.numRegs;out[1]=a.sharedSizeBytes+dynamic;out[2]=a.localSizeBytes;
	return cudaOccupancyMaxActiveBlocksPerMultiprocessor(out+3,f,cta,dynamic);
}
