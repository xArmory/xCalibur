import torch
from ._C import topk


def xR38F1(W13, X, routes):
    raise RuntimeError("xR38F1 is paused: the restored pre-epilogue source does not write output")


def topk_bitmap(logits, K, softmax=True):
    if logits.ndim != 2 or logits.dtype != torch.bfloat16 or logits.requires_grad:
        raise ValueError("logits must be BF16 [N,E], without gradients")
    N, E = logits.shape
    if not (1 <= N <= 65536 and 1 <= E <= 65536 and 1 <= K <= min(16, E)):
        raise ValueError("1 <= N,E <= 65536; 1 <= K <= min(16,E) required")
    x = logits.float()
    if softmax:
        a = (x - x.amax(-1, keepdim=True)).exp().bfloat16()
    else:
        a = ((-x).exp().bfloat16() + 1).reciprocal()
    keys = (a.view(torch.int16).int() << 16) | (65535 - torch.arange(E, device=x.device, dtype=torch.int32))
    ids = keys.topk(K, dim=-1).indices
    w = a.gather(1, ids)
    if softmax:
        w = (w.float() / a.float().sum(-1, keepdim=True)).bfloat16()
    topkw = torch.zeros(E, N, dtype=torch.bfloat16, device=x.device)
    topkw.scatter_(0, ids.T, w.T)
    n = torch.arange(N, device=x.device)
    words = (N + 31) // 32
    bitmap = torch.zeros(E * words, dtype=torch.int32, device=x.device)
    bitmap.scatter_add_(0, (ids.T * words + n // 32).flatten(),
                        (1 << (n & 31)).int().expand(K, N).flatten())
    return bitmap.view(E, words), topkw


def pack_w13(gate, up):
    if (gate.ndim != 3 or gate.shape != up.shape or gate.shape[-1] % 8
            or gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16
            or gate.requires_grad or up.requires_grad):
        raise ValueError("gate/up must be BF16 [E,I,H], H%8=0, without gradients")
    E, I, H = gate.shape
    return torch.stack((gate.reshape(E, I, H // 4, 4),
                        up.reshape(E, I, H // 4, 4)), dim=-2).flatten(-3).view(torch.int32)
