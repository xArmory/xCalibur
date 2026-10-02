from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo

class XR38:
    def __init__(self, E, I, H, N, K, tN, tI, warps):
        self.dims = (E, I, H, N, K)
        self.tiles = (E, tN, tI)
        self.warps = warps
        
    def __call__(self, W, X, topKi, topKw):
        E, I, H, N, K = self.dims
        Y = Tensor.empty((E, N, I), dtype=dtypes.bfloat16, device=X.device)
        Xs = Tensor.empty((E, 8 if N < 4096 else 32, H), dtype=dtypes.bfloat16, device=X.device)

    def f1(self, W, X, Xs, Y, topKi, topKw):
        E, I, H, N, K = self.dims
        bid = list(map(lambda x: UOp.special(self.tiles[x], f"gidx{x}"), list(range(3))))
        tidx = UOp.special(self.warps*32, "lidx0")
        lidx, widx = tidx & 31, tidx >> 5