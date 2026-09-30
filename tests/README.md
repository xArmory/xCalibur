# Kernel checks

Small BF16 cases for the AA2 L4 setup. Runs on each visible SM89 GPU; one L4 is enough. No model download, distributed launch or speed target. Exact student dimensions are still open in AA2.

```bash
python -m pytest -q tests
```

- K1: softmax and sigmoid; BF16 activation ranking, smaller-ID ties, weights, odd expert counts, K=16 and token tails.
- K2: K1 routes → packed W13 → Gate/Up → SwiGLU × weight → packed Y. Dense FP32 dot reference, then BF16 Gate/Up and BF16 rounding after each epilogue operation. Checks counts, token IDs, empty experts and exact zero padding.
- Shapes cross token8, I192, H1536 and N768 boundaries. Largest weight case: E32, I256, H2048. These are kernel cases, not a full AA2 model evaluation.
- IDs/counts/padding are exact. BF16 weights use `rtol=0.008, atol=2e-5`; K2 uses `rtol=0.02, atol=2e-4` against the staged BF16 reference. FP32 accumulation order can cross a BF16 Gate rounding boundary; staged products produced two-step output differences. An isolated helper check matched the same rounded inputs exactly.
- Missing CUDA/SM89 skips; a missing extension on SM89 fails.

2026-09-30, Colab L4, CUDA 12.8, PyTorch 2.11.0+cu128: the [validated snapshot](l4_sources.sha256) passes 19/19. Racecheck, memcheck and synccheck each pass all 19 with zero errors.

Both compute barriers are restored. W13 uses `.nc`, L2 evict-last and 256B prefetch. Xs uses coherent `.cs` vector loads followed by shared stores: evict-first in L1/L2, with zero-filled tails.

`ptxas -v`, SM89:

| Kernel | Registers/thread | Stack bytes/thread | Spill stores/loads (bytes reported) |
|---|---:|---:|---:|
| K1 sigmoid | 39 | 128 | 0 / 0 |
| K1 softmax | 40 | 128 | 0 / 0 |
| K2 | 40 | 64 | 124 / 168 |

SASS contains local loads/stores in both kernels. K1's indexed array uses stack storage despite zero reported spills; K2 has register spills. Neither compacting K2's array from 38 to 28 slots nor switching to the packed BF16 epilogue removed them. The cache update raised reported spill loads from 164B to 168B. These are compiler counts, not measured runtime traffic.

CUDA's occupancy API on the compiled kernels reports at most 6 K1 blocks or 2 K2 blocks per SM. L4 has 58 SMs. This is a residency limit; achieved occupancy was not profiled. [Tested source hashes](l4_sources.sha256).

For shared-memory races on an L4 with Compute Sanitizer installed:

```bash
compute-sanitizer --tool racecheck --error-exitcode 1 python -m pytest -xq tests -k xr38f1
```

## Speed

```bash
python tests/bench_supersonic.py bench.csv
```

Same L4/environment above; [full results](bench_l4.csv). Median synchronized wall time over 40 calls/provider, alternating order after 10 warm-ups. Includes dispatch, allocation, routing and output packing; inputs and weight packing stay outside timing. Outputs are checked first.

Published layer dimensions:

| Model | E | K | H | I |
|---|---:|---:|---:|---:|
| [OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json) | 64 | 8 | 2048 | 1024 |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) | 128 | 8 | 2048 | 768 |

N = 1, 8, 32, 128, 512, 2048, 4096. Synthetic weights and router logits; xCalibur's BF16 routing/epilogue contract. Model dimensions only: Qwen's selected-weight renormalization and full-model execution are outside this benchmark.

Baseline: eager PyTorch. K1 uses rounded BF16 activation and integer top-K keys. K2 sorts routes once, gathers once, copies expert counts to the CPU once, then runs one combined Gate/Up GEMM per active expert, BF16 SwiGLU/weighting and Y packing. Empty experts skip compute. Timings include Python and launch overhead; full-model speed remains unmeasured.

K2 milliseconds, **xCalibur / PyTorch**:

| N | OLMoE | Qwen3 |
|---:|---:|---:|
| 1 | 0.333 / 3.539 | 0.197 / 3.546 |
| 8 | 1.751 / 17.549 | 1.433 / 20.535 |
| 32 | 2.351 / 25.281 | 3.051 / 42.868 |
| 128 | 5.694 / 25.803 | 5.043 / 50.181 |
| 512 | 19.830 / 25.975 | 15.735 / 50.177 |
| 2048 | 75.450 / 26.122 | 58.555 / 51.218 |
| 4096 | 150.188 / 27.145 | 114.464 / 51.689 |

All 42 benchmark rows pass their output checks. K1 takes 21.8–41.2 µs across both modes/shapes. K2 loses to this baseline at N=2048 and 4096 for both shapes.

## Co-design review

Source counts below describe the current kernel. The device profile below records the earlier cache configuration.

- **Routing:** every expert scans N tokens, with two CTA barriers/token. N=4096 means 8192 route barriers/CTA. Total barriers = `2*N + B_e*(1 + 2*ceil(I/192)*ceil(H/1536))`, where `B_e` is that expert's batch count. Keep the outer scan loop uniform; its inner guard includes `threadIdx.x`.
- **Weight reuse:** each batch of eight rereads the expert's full W13. Nominal requested bytes = `4*I*H*sum(B_e)`; cache/DRAM traffic is unmeasured. Batches run serially within each expert CTA.
- **Staging:** coherent `.cs` load → shared store → CTA barrier. Xs requests L1/L2 evict-first. No overlap with the next stage. X occupies 24 KiB of the 32 KiB shared allocation.
- **Registers:** `rmem[28]` compiles to 40 registers/thread plus spills. Two-block launch bounds constrain the compiler; changing the array length alone did not remove spills.
- **Y:** each lane stores one packed BF16 pair; a warp writes 128 contiguous payload bytes. K3 owns the token scatter. At N=4096, slabs reserve 513 MiB / 770 MiB for OLMoE / Qwen3; valid payload is 64 MiB / 48 MiB, excluding padding and metadata.
- **Weights:** `ld.global.nc.L2::cache_hint.L2::256B.v4.b32` has four destination registers/lane and an L2 evict-last policy. [PTX contract](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld).

### Cache comparison

[Previous sweep](bench_l4_before_cache.csv); [paired device timings](cache_l4_comparison.json). Same L4 session and identical inputs, N4096, preallocated buffers, 12 alternating CUDA-event samples after three warm-ups. Old/new outputs match exactly. Baseline: `.cg` + 128B W13 loads, `cp.async.ca` Xs staging, both barriers present.

| Shape | Before | New policies |
|---|---:|---:|
| OLMoE | 151.960 ms | 150.126 ms |
| Qwen3 | 116.940 ms | 115.165 ms |

About 1–2% faster in this run. This measures the combined cache/staging change.

### Device profiling

[Timings](profile_l4.csv), [Nsight counters](profile_l4_counters.json), [source hashes](profile_l4_sources.sha256). Diagnostic builds; preallocated buffers, CUDA-event medians, 12 alternating samples after three warm-ups. All packing is outside timing.

- OLMoE N4096: original 152.6 ms; allocation included 153.0 ms; routing/gather only 1.30 ms. Removing prefetch: 163.9 ms. One-CTA launch bound, zero spills: 141.6 ms.
- CTA512, retaining both compute barriers: 64 registers, zero stack/spills. OLMoE 147.6 vs CTA768 152.8 ms; Qwen3 117.0 vs 115.7 ms at N4096. Six K2 tests and all three sanitizers pass; six model-dimension cases match CTA768 exactly.
- Original OLMoE profile: 41.05 GB DRAM reads, 0.44 GB writes, 89.24% DRAM throughput, 3.93% SM throughput. These counters used uncontrolled clocks/caches; event medians above are the timing comparison. Aggregate L2 hit rate 65.12%; token-only hit rate is unmeasured.
- `no_epilogue` is not a valid estimate of epilogue cost: the compiler also removes all tensor instructions. Its CSV rows are diagnostic only.

### Excluded cache diagnostic

The [shared-weight experiment](diagnostic_shared_weights_l4.csv) uses identical weights across experts: 64 copies (512 MiB) versus one copy (8 MiB). This changes the workload. Its timings are excluded from the ablation data and cannot establish a model speedup or quantify cache costs with independent expert weights.

## Bindings

All launch inputs: contiguous, same CUDA device, no gradients. Uses PyTorch's current stream. Forward only.

- `topk(logits, K, softmax=True)`: BF16 `[N,E]` → int32 `[N,K]`. Finite logits; 1≤N,E≤65536; 1≤K≤min(16,E). Each output word is `[BF16 weight | 0xffff-expert]`.
- `pack_w13(gate, up)`: BF16 `[E,I,H]` on CPU or CUDA → int32 `[E,I,H]`. H%8=0. Packs each H4 as Gate h01/h23, Up h01/h23; do this once when preparing weights.
- `xR38F1(W13, X, routes)`: packed int32 weights, BF16 `[N,H]`, K1's int32 routes → int32 expert slabs. Inputs need 16-byte alignment. Allocates Xs scratch internally.

Y shape: `[E, 8 + ceil(N/8)*(8 + 32*ceil(I/8))]` words. Read the batch count in word 0, then only that many records. Unused records are uninitialized. [Packing contract](../xcalibur/supersonic/xR38.md#e-y--k3).
