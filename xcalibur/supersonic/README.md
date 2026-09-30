# Supersonic MoE

Supersonic is a CUDA/PTX kernel project for the mixture-of-experts (MoE) forward pass: select experts for each token, run their feed-forward networks, and combine the results.

Current focus: bfloat16 on SM89, processing routed tokens in batches of eight per expert.

## Co-design video

[![Watch video](https://i9.ytimg.com/vi/br5LvDz76v8/mqdefault.jpg?sqp=CKSW8dUG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGGYgZihmMA8=&rs=AOn4CLDEq1nBrPGDO0kCjTg_oMVm7FVpHg)](https://youtu.be/br5LvDz76v8)

## Pipeline

| Kernel | Work | Output |
|---|---|---|
| K1 — [topk](topk.cu) | Apply softmax or sigmoid to router scores; select the top K experts per token. | Expert IDs and routing weights (`tKwi`). |
| K2 — [xR38F1](xR38F1.cu) | Gather an expert's tokens; compute Gate and Up projections; apply SwiGLU and routing weights. | Packed intermediate activations (`Y`) and original token IDs. |
| K3 — xR38F2 (planned) | Apply the Down projection; sum expert contributions into their original token rows. | One output vector per token. |

K1 is a separate kernel launch; K2 consumes its completed `tKwi`. Weights are packed at checkpoint creation; K2 writes Y in the order K3 will consume it, avoiding another packing step.

## xR38F1

`xR38F1_bf16` is the K2 kernel:

- One thread block owns one expert: 768 threads, 24 warps, 32 KiB shared memory.
- Each warp computes eight intermediate channels for eight tokens, reducing over the full hidden width.
- Gate and Up share one FP32 accumulator fragment. The epilogue writes BF16 Y into a reserved region for each expert.
- Target: two resident blocks per SM; occupancy remains unverified.

## Status

K1 and K2 have source implementations. K2 synchronization still needs work; K3 and host integration are pending. CUDA correctness and performance have not been validated.

[Full co-design: layouts, register maps, bit matrices and helper contracts →](README2.md)
