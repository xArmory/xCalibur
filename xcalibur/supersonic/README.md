# Supersonic MoE
> arch: {sm80}, sm89, {sm120x}
> dtype: bfloat16
> regime: Midfill

## Kernel 1

topk (softmax, sigmoid)

## Kernel 2:

Gather (pack contiguous / l2 prefetch); Gate,Up; Activation; topkW; Write packed contigous

Kernel 1 depends on what Kernel 2 needs.

## Kernel 3:

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

-----------------+
| mma split v2   |
+----------------+
|      t0        |
+----------------+
i015h015         | 
| i015h1531      | 
v v ...          |
0 1 2 ... 31 wid |
+----------------+
| stream offset  |
+----------------+
|      t1        |
+----------------+
i015h512527      | 
| i015h528543    |
v v ...          |
0 1 2 ... 31 wid |
-----------------+
```

v3: v1 + v2

Layout:
````text
+_______________________________________________________________________________________k16_________________________________________________________________________k16
|    32b        32b        32b        32b        32b        32b        32b        32b    |       |       32b              32b              32b              32b      |
+----------------------------------------------------------------------------------------+-------+-------------------------------------------------------------------+
| w1i015h01, w3i015h01, w1i015h23, w3i015h23, w1i015h45, w3i015h45, w1i015h67, w3i015h67,| ,..., | w1i015h20442045, w3i015h20442045, w1i015h20462047, w3i015h20462047| -> w13ih1531, ...., I*16 (k)
+----------------------------------------------------------------------------------------+-------+-------------------------------------------------------------------+
.
.
.
.
H/2048 (sm smem config based formula)


```