# Supersonic MoE
> arch: {sm80}, sm89, {sm120x}
> dtype: bfloat16
> regime: Midfill

@TODO: REWORK CODESIGN

## Kernel 1: topk

topk (softmax, sigmoid)

## Kernel 2: xR38F1

Gather (pack contiguous / l2 prefetch); Gate,Up; Activation; topkW; Write packed contigous

Kernel 1 depends on what Kernel 2 needs.

## Kernel 3: xR38F2

Down; write out scatter reduce

Note:

Weights are formated with our layouts at checkpoint creation time to make the downstream kernels optimal.

## sm89 specifications

SMs: 58
L1: 128 KB
L2: 48 MB
GPU: 24 GB
Max Warps / SM: 48
Constant memory / SM: 8KB

## Configuration

CTA = 768, 2xCTA / SM

1. smem = 32kb / CTA // 64kb / SM L1 cache
2. rmem < 43 reg / thread

smem = 8192 reg / CTA
-> 8*2048 = 16,384 bf16 vals
-> 8192 32b vals

- [ ] bank configuration

each panel is 64 32b values
2 banks per panel

```text

+----------------+
| mma split v1   |
+________________+
i015             |
| i1531          |
v v ...          |
0 1 2 ... 31 wid |
-----------------+
```

Layout:
```text

use mma.sp::ordered_metadata.m16n8k32 (paraphrase)

w1i0h01, w1i0h23, w3i0h01, w3i0h23, w1i0h45, w1i0h67, w3i0h45, w3i0h67,


w1i0h1617, w1i0h1819, w1i0h3233, w1i0h3435, w1i0h6465, w1i0h6667,

w3i0h1617, w3i0h1819, w3i0h3233, w3i0h3435, w3i0h6465, w3i0h6667,


metadata:
0x4444'4444 -> w1i0h01, w3i0h01, w1i0h1617, w3i0h1617 fsel=0 or 1
0xEEEE'EEEE -> w1i0h23, w3i0h23, w1i0h1819, w3i0h1819 fsel=1 or 0
```

@TODO review layout algebra

[Co-design video](https://youtu.be/br5LvDz76v8)

[Co-design notes: layouts and bit matrices](xR38.md)
