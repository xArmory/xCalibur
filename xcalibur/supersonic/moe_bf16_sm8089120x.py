import triton
import triton.language as tl

@triton.jit
def bf(x):
	return x.to(tl.bfloat16).to(tl.float32)

@triton.jit
def swiglu(g, u):
	g, u = bf(g), bf(u)
	t = bf(tl.exp(-tl.abs(g)))
	s = bf(1. / bf(1. + t))
	s = bf(s * tl.where(g < 0, t, 1.))
	return bf(bf(g * s) * u)

@triton.jit
def weighted2(a, b, w):
	return tl.inline_asm_elementwise("""{
		.reg .b32 weight, zero;
		cvt.rn.bf16x2.f32 $0, $2, $1;
		cvt.rn.bf16x2.f32 weight, $3, $3;
		mov.b32 zero, 0x80008000;
		fma.rn.bf16x2 $0, $0, weight, zero;
	}""", constraints='=r,f,f,f', args=[a,b,w], dtype=tl.uint32, is_pure=True, pack=1)

@triton.jit
def store2(P, v, mask):
	tl.inline_asm_elementwise('{ .reg .pred p; mov.b32 $0,0; setp.ne.u32 p,$3,0; @p st.global.b32 [$1],$2; }',
		constraints='=r,l,r,r',args=[P,v,mask.to(tl.int32)],dtype=tl.uint32,is_pure=False,pack=1)

@triton.jit
def reduce2(P, v, mask):
	code:tl.constexpr = '''{ .reg .pred p; .reg .b32 old,assumed,next,one; .reg .b64 policy;
		mov.b32 $0,0; setp.ne.u32 p,$3,0; @!p bra done;
		createpolicy.fractional.L2::evict_last.b64 policy,1.0;
		mov.b32 one,0x3f803f80;
		ld.global.cg.L2::cache_hint.b32 old,[$1],policy;
	loop: mov.b32 assumed,old; fma.rn.bf16x2 next,old,one,$2;
		atom.relaxed.gpu.global.cas.b32 old,[$1],assumed,next;
		setp.ne.b32 p,old,assumed; @p bra loop;
	done: }'''
	tl.inline_asm_elementwise(code,constraints='=r,l,r,r',args=[P,v,mask.to(tl.int32)],dtype=tl.uint32,is_pure=False,pack=1)

@triton.jit
def sum2(P, Z, N:tl.constexpr, H:tl.constexpr, K:tl.constexpr, B:tl.constexpr):
	p = tl.program_id(0)*B+tl.arange(0,B)
	n, h = p//(H//2), p%(H//2)
	a = tl.full((B,),0,tl.uint32); one = tl.full((B,),0x3f803f80,tl.uint32)
	for k in tl.static_range(K):
		v = tl.load(P.to(tl.pointer_type(tl.uint32))+(n*K+k)*(H//2)+h,n<N,0)
		a = tl.inline_asm_elementwise('fma.rn.bf16x2 $0, $1, $2, $3;',constraints='=r,r,r,r',args=[a,one,v],dtype=tl.uint32,is_pure=True,pack=1)
	tl.store(Z.to(tl.pointer_type(tl.uint32))+p,a,n<N)

@triton.jit
def topk(L, Q, R, Z, E:tl.constexpr, H:tl.constexpr, K:tl.constexpr, B:tl.constexpr, CLEAR:tl.constexpr):
	n = tl.program_id(0)
	e = tl.arange(0, B)
	x = tl.load(L + n * E + e, e < E, -float('inf')).to(tl.float32)
	x = tl.exp(x - tl.max(x, 0))
	x = bf(x / tl.sum(x, 0))
	s = 0.
	for k in tl.static_range(K):
		v = tl.max(tl.where(e < E, x, -1.), 0)
		i = tl.min(tl.where((x == v) & (e < E), e, 2147483647), 0)
		tl.store(R + n * K + k, i)
		tl.store(Q + n * K + k, v)
		s += v
		x = tl.where(e == i, -1., x)
	for k in tl.static_range(K):
		tl.store(Q + n * K + k, bf(tl.load(Q + n * K + k) / s))
	if CLEAR:
		h = tl.arange(0, triton.next_power_of_2(H))
		tl.store(Z + n * H + h, 0., h < H)

@triton.jit
def routes(R, N:tl.constexpr, K:tl.constexpr, S:tl.constexpr, DIRECT:tl.constexpr):
	if DIRECT:
		r = tl.where(tl.arange(0, S) == 0, tl.program_id(1), N * K)
		e = tl.load(R + tl.program_id(1))
		count = 1
	else:
		e = tl.program_id(1)
		n = tl.program_id(2) * S + tl.arange(0, S)
		k = tl.arange(0, triton.next_power_of_2(K))
		x = tl.load(R + n[:, None] * K + k[None, :], (n < N)[:, None] & (k < K)[None, :], -1)
		r = tl.min(tl.where(x == e, n[:, None] * K + k[None, :], N * K), 1)
		count = tl.sum((r < N * K).to(tl.int32), 0)
		if S > 1024:
			r = tl.cumsum((r < N * K).to(tl.int32), 0)
		else:
			r = tl.sort(r)
	return r, e.to(tl.int64), count

@triton.jit
def gather(R, r, j, e, count, N:tl.constexpr, K:tl.constexpr, S:tl.constexpr, BM:tl.constexpr):
	if S > 1024:
		lo, hi = tl.full((BM,),0,tl.int32), tl.full((BM,),S,tl.int32)
		for _ in tl.static_range(triton.next_power_of_2(S).bit_length()):
			mid = (lo + hi) // 2
			v = tl.gather(r, tl.minimum(mid,S-1), 0)
			lo, hi = tl.where(v <= j,mid+1,lo), tl.where(v > j,mid,hi)
		n = tl.program_id(2)*S + lo
		k = tl.arange(0,triton.next_power_of_2(K))
		x = tl.load(R+n[:,None]*K+k[None,:],(j<count)[:,None] & (n<N)[:,None] & (k<K)[None,:],-1)
		q = tl.min(tl.where(x==e,n[:,None]*K+k[None,:],N*K),1)
	else:
		q = tl.gather(r, tl.minimum(j,S-1), 0)
	return tl.where(j<count,q,N*K).to(tl.int64)

@triton.jit
def f1(X, W, Y, R, N:tl.constexpr, H:tl.constexpr, I:tl.constexpr, K:tl.constexpr,
	S:tl.constexpr, BM:tl.constexpr, BI:tl.constexpr, BK:tl.constexpr, DIRECT:tl.constexpr, H4:tl.constexpr):
	r, e, count = routes(R, N, K, S, DIRECT)
	v = tl.arange(0, 2 * BI)
	i = tl.program_id(0) * BI + v // 16 * 8 + v % 8
	gate = v % 16 // 8
	k = tl.arange(0, BK)
	for t in range(tl.cdiv(count, BM)):
		j = t * BM + tl.arange(0, BM)
		q = gather(R,r,j,e,count,N,K,S,BM)
		a = tl.full((2 * BI, BM), 0, tl.float32)
		for h in range(tl.cdiv(H, BK)):
			hh = h * BK + k
			if H4:
				p = ((e * I + i[:, None]) * tl.cdiv(H, 4) + hh[None, :] // 4) * 8 + gate[:, None] * 4 + hh[None, :] % 4
			else:
				p = ((e * tl.cdiv(H, 64) + hh[None, :] // 64) * (2 * I) + v[:, None] + tl.program_id(0) * (2 * BI)) * 64 + hh[None, :] % 64
			w = tl.load(W + p, (i < I)[:, None] & (hh < H)[None, :], 0, cache_modifier='.cg')
			x = tl.load(X + (q // K)[None, :] * H + hh[:, None], (q < N * K)[None, :] & (hh < H)[:, None], 0, cache_modifier='.ca')
			a = tl.dot(w, x, a)
		g, u = tl.split(a.reshape(BI // 8, 2, 8, BM).permute(0, 2, 3, 1))
		y = swiglu(g, u).reshape(BI, BM)
		ii = tl.program_id(0) * BI + tl.arange(0, BI)
		tl.store(Y + q[None, :] * I + ii[:, None], y, (q < N * K)[None, :] & (ii < I)[:, None])

@triton.jit
def route2(R, T, C, D, N:tl.constexpr, E:tl.constexpr, K:tl.constexpr, S:tl.constexpr, BM:tl.constexpr):
	e = tl.program_id(0)
	n = tl.arange(0,S); k = tl.arange(0,triton.next_power_of_2(K))
	hist = tl.full((triton.next_power_of_2(E),),0,tl.int32)
	for p in range(tl.cdiv(N,S)):
		q = (p*S+n[:,None])*K+k[None,:]
		x = tl.load(R+q,(p*S+n<N)[:,None] & (k<K)[None,:],-1)
		hist += tl.histogram(x.reshape(S*triton.next_power_of_2(K)),triton.next_power_of_2(E),(x>=0).reshape(S*triton.next_power_of_2(K)))
	a = tl.arange(0,triton.next_power_of_2(E))
	base = tl.sum(tl.where(a<e,hist,0),0); count = tl.sum(tl.where(a==e,hist,0),0)
	tl.store(C+2*e,base); tl.store(C+2*e+1,count)
	blocks = tl.cdiv(hist,BM); start = tl.sum(tl.where(a<e,blocks,0),0)
	if e==0:tl.store(D,tl.sum(blocks,0))
	b = tl.arange(0,triton.next_power_of_2(triton.cdiv(N,BM))); valid = b*BM<count
	tl.store(D+1+3*(start+b),e,valid)
	tl.store(D+2+3*(start+b),base+b*BM,valid)
	tl.store(D+3+3*(start+b),tl.minimum(BM,count-b*BM),valid)
	for p in range(tl.cdiv(N,S)):
		q = (p*S+n[:,None])*K+k[None,:]
		x = tl.load(R+q,(p*S+n<N)[:,None] & (k<K)[None,:],-1)
		q = tl.min(tl.where(x==e,q,N*K),1)
		live = (q<N*K).to(tl.int32)
		tl.store(T+base+tl.cumsum(live,0)-1,q,q<N*K)
		base += tl.sum(live,0)

@triton.jit
def f2(Y, W, Z, Q, R, T, C, D, N:tl.constexpr, H:tl.constexpr, I:tl.constexpr, K:tl.constexpr,
	S:tl.constexpr, BM:tl.constexpr, BH:tl.constexpr, BK:tl.constexpr, DIRECT:tl.constexpr, E:tl.constexpr, ATOMIC:tl.constexpr):
	index:tl.constexpr = tl.int32 if E>0 and E*triton.cdiv(I,64)*H*64<2147483648 and N*K*max(H,I)<2147483648 else tl.int64
	if DIRECT:
		e = tl.load(R+tl.program_id(1)).to(index)
		base, count = tl.program_id(1), 1
	else:
		job = tl.program_id(1); valid = job<tl.load(D)
		e = tl.load(D+1+3*job,valid,0).to(index)
		base = tl.load(D+2+3*job,valid,0); count = tl.load(D+3+3*job,valid,0)
	h = tl.program_id(0) * BH + tl.arange(0, BH)
	k = tl.arange(0, BK)
	if count>0:
		j = tl.arange(0, BM)
		if DIRECT:q = tl.where(j<count,base,N*K).to(index)
		else:q = tl.load(T+base+j,j<count,N*K).to(index)
		a = tl.full((BM, BH), 0, tl.float32)
		for i in range(tl.cdiv(I, BK)):
			ii = i * BK + k
			p = ((e * tl.cdiv(I, 64) + ii[:, None] // 64) * H + h[None, :]) * 64 + ii[:, None] % 64
			w = tl.load(W + p, (h < H)[None, :] & (ii < I)[:, None], 0, cache_modifier='.cg')
			y = tl.load(Y + q[:, None] * I + ii[None, :], (q < N * K)[:, None] & (ii < I)[None, :], 0, cache_modifier='.ca')
			a = tl.dot(y, w, a)
		w = tl.load(Q + q, q < N * K, 0)
		a, b = tl.split(a.reshape(BM, BH // 2, 2))
		hh = tl.program_id(0) * BH + 2 * tl.arange(0, BH // 2)
		v = weighted2(a,b,w[:,None]); mask = (q<N*K)[:,None] & (hh<H)[None,:]
		if ATOMIC:reduce2(Z.to(tl.pointer_type(tl.uint32))+(q//K)[:,None]*(H//2)+hh[None,:]//2,v,mask)
		else:store2(Z.to(tl.pointer_type(tl.uint32))+q[:,None]*(H//2)+hh[None,:]//2,v,mask)

def pack(W, gate=False, h4=False):
	import torch
	E, J, D = W.shape
	if h4:
		assert gate
		w = torch.nn.functional.pad(W, (0, -D % 4))
		return w.reshape(E, 2, J // 2, -1, 4).permute(0, 2, 3, 1, 4).contiguous()
	if gate:
		assert J % 16 == 0
		W = W.reshape(E, 2, J // 16, 8, D).permute(0, 2, 1, 3, 4).reshape(E, J, D)
	W = torch.nn.functional.pad(W, (0, -D % 64))
	return W.reshape(E, J, -1, 64).permute(0, 2, 1, 3).contiguous()

def config(N, E, K, H, I, sms=58):
	import math
	S = max(16, min(triton.next_power_of_2(N), 4096))
	while S > 16 and min(E,N*K)*triton.cdiv(I,32)*triton.cdiv(N,S) < 2*sms:S //= 2
	mean = min(N,S)*K/E
	M = max(16, min(128, triton.next_power_of_2(math.ceil(mean+2*math.sqrt(mean)))))
	return dict(S=S, BM=min(M,S), BI=64 if S>=2048 and M==128 else 32, BH=64, BK=64,
		warps=16 if S>=2048 and M==128 else 8 if M>=64 else 4, stages=3)

class MoE:
	def __init__(self, W, V, N, K, H, I, cfg=None, h4=False, f2cfg=None):
		import torch
		self.W, self.V, self.N, self.K, self.H, self.I = W, V, N, K, H, I
		self.E, self.h4 = W.shape[0], h4
		assert I % 8 == 0 and H % 2 == 0 and 0 < K <= self.E and N > 0
		assert W.is_contiguous() and V.is_contiguous() and W.dtype == V.dtype == torch.bfloat16
		assert W.shape == ((self.E,I,triton.cdiv(H,4),2,4) if h4 else (self.E,triton.cdiv(H,64),2*I,64))
		self.config = cfg or config(N, self.E, K, H, I)
		small = N<=512 and I>=H
		self.f2config = f2cfg or dict(BM=min(max(16,triton.next_power_of_2(N)),64 if small else 128),BH=64 if small else 128,BK=64,warps=4 if small else 8,stages=3,atomic=N<=512)
		assert V.shape == (self.E,triton.cdiv(I,64),H,64) and N*K<2147483648
		def empty(shape, dtype):return torch.empty(shape, dtype=dtype, device=W.device)
		self.Y = empty((N*K,I), torch.bfloat16)
		self.Z = empty((N,H), torch.bfloat16)
		self.R = empty((N,K), torch.int32)
		self.Q = empty((N,K), torch.float32)
		self.T = empty((N*K,), torch.int32)
		self.C = empty((2*self.E,), torch.int32)
		self.D = empty((1+3*(triton.cdiv(N*K,self.f2config['BM'])+self.E),),torch.int32)
		self.P = None if self.f2config.get('atomic',False) else empty((N*K,H),torch.bfloat16)

	def route(self, L, clear=False):
		topk[(self.N,)](L,self.Q,self.R,self.Z,self.E,self.H,self.K,triton.next_power_of_2(self.E),clear)
		return self.Q, self.R

	def __call__(self, X, Q, R, clear=True):
		if clear:self.Z.zero_()
		c = self.config; S,M,B = c['S'],c['BM'],c['BK']; direct = self.N == 1
		args = (self.N,self.H,self.I,self.K,S,M)
		launch = dict(num_warps=c['warps'],num_stages=c['stages'],enable_fp_fusion=False)
		grid = (self.K,1) if direct else (self.E,triton.cdiv(self.N,S))
		self.f1 = f1[(triton.cdiv(self.I,c['BI']),*grid)](X,self.W,self.Y,R,*args,c['BI'],B,direct,self.h4,**launch)
		return self.finish(Q,R)

	def finish(self,Q,R):
		S=self.config['S'];direct=self.N==1
		c2 = self.f2config
		atomic=c2.get('atomic',False)
		if not direct:self.router = route2[(self.E,)](R,self.T,self.C,self.D,self.N,self.E,self.K,S,c2['BM'],num_warps=16 if S>=2048 else 4)
		self.f2 = f2[(triton.cdiv(self.H,c2['BH']),self.K if direct else triton.cdiv(self.N*self.K,c2['BM'])+self.E)](self.Y,self.V,self.Z if atomic else self.P,Q,R,self.T,self.C,self.D,self.N,self.H,self.I,self.K,S,c2['BM'],c2['BH'],c2['BK'],direct,
			self.E,atomic,num_warps=c2['warps'],num_stages=c2['stages'],enable_fp_fusion=False)
		if not atomic:self.reduce = sum2[(triton.cdiv(self.N*self.H,512),)](self.P,self.Z,self.N,self.H,self.K,256)
		return self.Z

	def forward(self, X, L):
		Q,R = self.route(L,True)
		return self(X,Q,R,False)
