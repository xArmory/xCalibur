# SuperSonic MoE
> **Variant**: Midfill / Decode;
> **Target arch**: {sm80}, sm89, {sm120};
> **DiDo**: bfloat16

## Formulation:

### Prelim Notation:

$\textbf{S}_i := \{\text{thread}_i : \forall i \in (0, I]\}$ (typically $I \le 1024, I \in  \N_1$).

$\S \exists T \le I$

$\textbf{SM}_i := \{\textbf{CTA}_k := (\text{thread}_j, \forall j \in (k, k+T]), \forall k \in (0, T, \ldots,|S_i|-T) \} $

$\text{rmem}_{\text{arch}}(c) = \text{min}(\text{RMEM[arch]} / c, 255)$

$\text{smem}_{\text{arch}}(c) = \text{SMEM[arch]} / c$

$\textbf{SM}'_i \subset \textbf{SM}_i : |\textbf{SM}'_i| \approx 256/\text{rmem}_{\text{arch}}(c)$

Questions:

1. How do we compute/select $T$, for a given problem?
2. Given $T$ how do we choose the right cute layout?