# SuperSonic MoE
> **Variant**: Midfill / Decode;
> **Target arch**: {sm80}, sm89, {sm120};
> **DiDo**: bfloat16

## Formulation:

### Prelim Notation:

$\textbf{S}_i := \{\text{thread}_i : \forall i \in (0, I]\}$ (typically $I \le 1024, I \in  \N_1$).

$\S \exists T \le I$

$\implies W \le T$  typically, (16, 32, 64)

$\textbf{SM}_i := \{\textbf{CTA}_k := (\text{thread}_j, \forall j \in (k, k+T]), \forall k \in (0, T, \ldots,|S_i|-T) \} $

$\textbf{CTA}_i := \{\textbf{warp}_k := (\text{thread}_j, \forall j \in (k, k+W]), \forall k \in (0, W, \ldots,|\textbf{CTA}_i|-W) \} $

$\text{rmem}_{\text{arch}}(c) = \text{min}(\text{RMEM[arch]} / c, 256)$

$\text{smem}_{\text{arch}}(c) = \text{SMEM[arch]} / c$

$\textbf{SM}'_i \subset \textbf{SM}_i : |\textbf{SM}'_i| \approx 256/\text{rmem}_{\text{arch}}(c)$

Questions:

1. How do we select $\Theta_i$, for a given problem? Such that the objective for-each parameter is met and the work done is complete.

2. $\Theta_i := \{T, \text{I/O}_{\{GMEM, TMEM, SMEM, RMEM\}}, \text{ALU}, \text{TC}, \text{SYNC}_{\{warp^i_{j, k}, CTA^i_j\}}\}$

### Kernel 1: TopK

$256 \le T \le I, I := 1024, W := 32$

$T^* := (256, 256+W, 256+2W, \ldots, I)$

Given: router_logits o (N, E) : (1, N), K

* load router logits, softmax/sigmoid, local topk, global topk, write out