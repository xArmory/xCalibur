# Supersonic MoE

> BF16 · CUDA / PTX · midfill · [implementation](moe_bf16_sm8089120x.cu) · [Triton reference](moe_bf16_sm8089120x.py)

> Comparable perf to vendor operators (co-design was ultimately generated through an agent loop)

Weights get formatted once at checkpoint creation. Inference runs on the packed layout.

```text
K1: logits → softmax → top-K → Q, R
F1: X, W13, R → gather → Gate/Up → SwiGLU → Y
F2: R → route2 → jobs
    Y, W2, jobs, Q → Down → × Q ┬→ scatter CAS → Z  (N ≤ 512)
                               └→ P → sum → Z      (N > 512)
```

## Co-design

**K1.** One CTA per token. FP32 softmax, rounded to BF16 before ranking; lower expert ID wins ties. Renormalize the selected weights, round to BF16. `Q` holds these values in FP32 storage; `R` holds expert IDs. K1 also clears `Z`.

**F1 / xR38F1.** One CTA per expert × token window × I tile. Compact matching routes locally; skip empty windows. Read X directly through the cache, traverse all H in H64 panels. Gate8 / Up8 occupy rows of one accumulator fragment. BF16 SwiGLU writes `Y[q,i]`, where `q = n*K+k`.

**F2 / xR38F2.** `route2` builds compact expert jobs on GPU. Each CTA owns a token tile × H tile and traverses all I in I64 panels. Apply top-K weights here. Small N scatters with packed BF16 CAS; large N writes `P[q,h]`, then adds routes in top-K order. No CPU count read.

Both GEMMs use BF16 tensor cores with FP32 accumulators. SwiGLU uses approximate exp and rounds Gate, Up, exp, denominator, reciprocal and products to BF16. F2 rounds its projection before weighting; each expert addition rounds to BF16. CAS addition order can vary.

Full operator: 4 launches for `1 < N ≤ 512`, 5 for `N > 512`. `N=1` skips route preparation. One stream; launch boundaries publish routing and outputs.

## Tiles

F1 splits N, E and I; each CTA computes the full H reduction.

- Start with `S = clamp(nextpow2(N),16,4096)`. Halve S until `min(E,N*K)*ceil(I/32)*ceil(N/S) ≥ 2*SMs`, or S reaches 16.
- `μ = min(N,S)*K/E`; token tile `M = clamp(nextpow2(ceil(μ+2√μ)),16,128)`, capped at S. Loop if more routes are live.
- `BI=64`, 16 warps when `S≥2048` and `M=128`; otherwise `BI=32`, 8 warps for `M≥64`, else 4.
- F2: `BM=128, BH=128`, 8 warps. For `N≤512` and `I≥H`: `64×64`, 4 warps. Cap BM at `max(16,nextpow2(N))`.
- Both use `BK=64`, two shared panels, and `cp.async.cg → ldmatrix → mma`. `op.resources` reports compiled registers, shared memory and resident CTAs.

The CUDA wrapper reads the device's SM count and compiles for SM80, SM89 or SM120a.

## Use

```python
from moe import MoE, pack

# CUDA BF16: W13[E,2I,H] (Gate then Up), W2[E,H,I]
op = MoE(pack(W13, gate=True), pack(W2), N, K, H, I)
Z = op.forward(X, logits)  # X[N,H], logits[N,E], both CUDA BF16
```

Requires CUDA PyTorch and `nvcc` (`NVCC` overrides its path). First construction compiles and caches the module under `build/`.

Pack and construct once. `I % 8 == 0`, `H % 2 == 0`, `0 < K ≤ E`. Buffers belong to the instance; copy Z to retain it across calls.

## Appendix: layouts

| Buffer | Shape | Storage |
|---|---|---|
| W13 | `[E,ceil(H/64),2I,64]` | BF16, Gate8 / Up8 |
| W2 | `[E,ceil(I/64),H,64]` | BF16 |
| Y | `[N*K,I]` | BF16, route order |
| Q / R | `[N,K]` | FP32 / int32 |
| T / C | `[N*K]` / `[2E]` | int32 route IDs / expert base,count |
| D | `[1+3*(ceil(N*K/BM)+E)]` | int32 job count; expert,base,live |
| P | `[N*K,H]`, only `N>512` | BF16 partials |
| Z | `[N,H]` | BF16 |

Y costs `2*N*K*I` bytes; P adds `2*N*K*H`. No separate Xs buffer. Weights pad the reduction dimension to 64.

SM89 lowering: two shared K64 panels, `cp.async → ldmatrix → mma`. Shared rows use BF16 offset `row*64 + (k ^ ((row&7)*8))`; synchronization publishes copies and protects panel reuse.

Within each W13 group: `row = 16*(i//8) + 8*up + i%8`.

```text
row bits    up i2 i1 i0
r3           1  0  0  0
r2           0  1  0  0
r1           0  0  1  0
r0           0  0  0  1
```

SM89 `mma.m16n8k16` accumulator: lane bits `l4..l0`, register bits `r1,r0`. Tile offsets sit above these bits.

```text
F1         l4 l3 l2 l1 l0 r1 r0
Gate/Up     0  0  0  0  0  1  0
i2          1  0  0  0  0  0  0
i1          0  1  0  0  0  0  0
i0          0  0  1  0  0  0  0
token2      0  0  0  1  0  0  0
token1      0  0  0  0  1  0  0
token0      0  0  0  0  0  0  1

F2         l4 l3 l2 l1 l0 r1 r0
token3      0  0  0  0  0  1  0
token2      1  0  0  0  0  0  0
token1      0  1  0  0  0  0  0
token0      0  0  1  0  0  0  0
h2          0  0  0  1  0  0  0
h1          0  0  0  0  1  0  0
h0          0  0  0  0  0  0  1
```

[Co-design video](https://youtu.be/br5LvDz76v8)
