import torch
from ._C import topk, xR38F1


def pack_w13(gate, up):
    if (gate.ndim != 3 or gate.shape != up.shape or gate.shape[-1] % 8
            or gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16
            or gate.requires_grad or up.requires_grad):
        raise ValueError("gate/up must be BF16 [E,I,H], H%8=0, without gradients")
    E, I, H = gate.shape
    return torch.stack((gate.reshape(E, I, H // 4, 4),
                        up.reshape(E, I, H // 4, 4)), dim=-2).flatten(-3).view(torch.int32)
