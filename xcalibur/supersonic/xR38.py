from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo

class XR38:
    def __init__(self, E, I, H, N, K, tN, tI, warps):
        self.dims = (I, H, N, K, E, tN, tI)
        self.warps = warps
        
    def __call__(self, W, X, topKi, topKw):
        I, H, N, K, E = self.dims[:-2]
        Y = Tensor.empty((E, N, I), dtype=dtypes.bfloat16, device=X.device)
        
    def f1(self, W, X, Y, topKw):
        bid = list(map(lambda x: UOp.special(self.dims[-3:][x], f"gidx{x}"), list(range(3))))
        tidx = UOp.special(self.warps*32, "lidx0")
        nidx = UOp.placeholder(((self.warps << 5),), dtypes.int32, slot=0, addrspace=AddrSpace.LOCAL)

        nt = UOp.range((self.dims[-2] + (self.warps << 5) - 1) // (self.warps << 5), 0, AxisType.LOOP)
        hit = UOp.const(0, dtypes.uint32)
        v = UOp.range(4, 1, AxisType.UPCAST)
        
        for k4 in range(0, topKw.shape[1], 4):
            entry = topKw[nt * (self.warps << 5) + tidx, k4 + v].load()
            match = (entry >> 16).eq(bid[0])
            hit = hit | match.cast(dtypes.uint32).reduce(v, arg=Ops.MAX)
                
        st = nidx[tidx].store(hit.cast(dtypes.bool).where(nt * (self.warps << 5) + tidx, -1).cast(dtypes.int32))
        nidx = nidx.after(st.barrier())

        rA = UOp.placeholder((12,), dtypes.uint32, slot=1, addrspace=AddrSpace.REG)
        rB = UOp.placeholder((6,), dtypes.uint32, slot=1, addrspace=AddrSpace.REG)
                
        rC = UOp.placeholder((4,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
        rC = rC.after(rC.store(rC.const_like(0.0)))
