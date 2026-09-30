# Supersonic co-design

> sm89; bf16; midfill

`topk` (K1) → `tKwi` → `xR38F1` (K2) → `Y` → `xR38F2` (K3).

## K1: topk

`topk_kernel<softmax>` in [topk.cu](topk.cu). Separate launch; completes `tKwi` before K2.

- Grid = `(ceil(N/8),1,1)`; block = `(8,4,8)`; K≤16; finite BF16 logits.
- Softmax/sigmoid → pairs via `j^4` → local top-K → XOR merge. One warp/token; smaller expert ID wins ties.
- **Softmax:** max pass → shifted BF16 exp → FP32 sum over E → normalize selected weights at write-out.
- **Sigmoid:** BF16 activation ranking.
- **Loads:** tail/unaligned loads use scalar fallback.

## K2: xR38F1

`xR38F1_bf16` in [xR38F1.cu](xR38F1.cu).

- Grid = `(E,1,1)`; block = `(768,1,1)` = 24 warps; target = 2 CTA / SM.
- 1 CTA = 1 expert; 1 warp = I8 × token8; I step = 192.
- smem = 32 KiB; `rmem[38]`, 28 slots used. Physical register count unverified.

### Phase 1: Route lookup & Gather

`xR38F1Route` scans K1's N×K routes/expert. Publish in token order; gather eight accepted tokens. `ld.cg.v4` → `st.v4`. Reuse one Xs batch/expert; zero-fill final tail.

### Phase 2: Gate@Xs, Up@Xs

`xR38F1Compute_bf16`: W13 packed at checkpoint creation. All warps reduce full H; shared X reused across 24 warps.

- **Compute entry:** CTA barrier publishes gathered Xs and the route header to readers.
- **Stage H1536:** 2 × 16B `cp.async.ca`/thread → commit → `wait<0>` → CTA barrier.
- **Per H64:** 4 × `ld.cg.L2::128B.v4` W13/lane; 2 × `ldmatrix.x4`; 4 × `mma.sp.m16n8k32`.

Gate/Up occupy different rows of one FP32 fragment. Complementary masks reconstruct dense products.

### Phase 3: SwiGLU, topkW, write out

`xR38F1Epilogue_bf16`:

- After full H: `SiLU(Gate) × Up × topkW`. `ex2.approx` + `rcp.approx`, FTZ; one final BF16 rounding.
- XOR-4 pairs adjacent I rows. Fixed expert slabs; Y uses Xs packing along I.

## K3: xR38F2

Planned: `xR38F2.cu`. Down → scatter/reduce using Y's token IDs, after K2 completes.

## Open

- Route readers → slot reuse and shared readers → stage reuse need ordering. Tail pre-zero barrier redundant.
- W2/K3; pipeline; PTX helper review; banks/cache; host launch/allocation; CUDA correctness, registers, benchmarks.
- Bounds: 1≤E,N≤65536; 1≤K≤E; I,H>0; H%8=0; 16B vector alignment. Zero-fill H/I/token tails.

## Appendix: layouts / bit maps

- Offsets: uint32 words.
- Bit matrices: columns = input bits; rows = output bits; 1 copies a bit; bit 0 = LSB.
- Maps are local to the named tile; add outer strides separately.

### A. Ownership / storage

```text
                  token 0..7
             +----------------+
warp  0      | i   0..7       |
warp  1      | i   8..15      |
  ...        |                |
warp 23      | i 184..191     |
             +----------------+
                  ↓ next I192

smem: | routes: 8 words | X: 6144 words | unused: 2040 words |
                         H1536 × token8

K2 rmem: | 0 scan | 1 pending route | 2 batches | 3 fill |
         | 4..19 W13 | 20..28 unused |
         | 29..32 X | 33..36 C | 37 unused |
4..7 route lookup scratch first; compute base = rmem+4
compute-local: [0..15] W13; [25..28] X; [29..32] C

K1 rmem: | 0..7 paired loads / j^4 | 16..31 local winners |

            31                  16 15                   0
            +----------------------+----------------------+
tKwi        | BF16 weight          | 0xffff - expert      |
smem        | BF16 weight          | token ID             |
            +----------------------+----------------------+
empty = 0xffffffff
tKwi[t*K+k]: contiguous N*K words
```

### B. W13 → A

```text
W13[e,i,H4]: | w1h01 | w1h23 | w3h01 | w3h23 |  (32b words)

H32 tile; b=0 Gate, b=1 Up; a = local BF16 address
      b h4 h3 h2 h1 h0
 a5 [ 0  1  0  0  0  0 ]
 a4 [ 0  0  1  0  0  0 ]
 a3 [ 0  0  0  1  0  0 ]
 a2 [ 1  0  0  0  0  0 ]
 a1 [ 0  0  0  0  1  0 ]
 a0 [ 0  0  0  0  0  1 ]
word = (e*I+i)*H + h32 + (a>>1); half = a0
allocation = E*I*H words

A register r=0..3; half v; lane l; mask p=0:4444, p=1:EEEE
     l4 l3 l2 l1 l0 r1 r0  p  v
  b [ 0  0  0  0  0  0  1  0  0 ]
 i2 [ 1  0  0  0  0  0  0  0  0 ]
 i1 [ 0  1  0  0  0  0  0  0  0 ]
 i0 [ 0  0  1  0  0  0  0  0  0 ]
 h4 [ 0  0  0  0  0  1  0  0  0 ]
 h3 [ 0  0  0  1  0  0  0  0  0 ]
 h2 [ 0  0  0  0  1  0  0  0  0 ]
 h1 [ 0  0  0  0  0  0  0  1  0 ]
 h0 [ 0  0  0  0  0  0  0  0  1 ]
row = tile_base + 8*warp + i; column = h32 + h

H32     A operands: compute-local rmem   metadata                   fsel
first   0,2,4,6 / 1,3,5,7                0x44444444 / 0xEEEEEEEE    0
second  8,10,12,14 / 9,11,13,15          0x44444444 / 0xEEEEEEEE    1
masks cover H4 positions {0,1} / {2,3}; fsel selects metadata lanes
```

### C. Xs → B

```text
H8 group             word0   word1   word2   word3
                  +--------+--------+--------+--------+
token 0           | h0 h1  | h2 h3  | h4 h5  | h6 h7  |
token 1           | h0 h1  | h2 h3  | h4 h5  | h6 h7  |
  ...             |        |        |        |        |
token 7           | h0 h1  | h2 h3  | h4 h5  | h6 h7  |
                  +--------+--------+--------+--------+
                    ↓ next H8 group; pairs shown low → high

H32 × token8 tile; a = local BF16 address
     h4 h3 h2 h1 h0 t2 t1 t0
 a7 [ 1  0  0  0  0  0  0  0 ]
 a6 [ 0  1  0  0  0  0  0  0 ]
 a5 [ 0  0  0  0  0  1  0  0 ]
 a4 [ 0  0  0  0  0  0  1  0 ]
 a3 [ 0  0  0  0  0  0  0  1 ]
 a2 [ 0  0  1  0  0  0  0  0 ]
 a1 [ 0  0  0  1  0  0  0  0 ]
 a0 [ 0  0  0  0  1  0  0  0 ]
word = e*(4*H) + 4*h32 + (a>>1); half = a0
allocation = 4*E*H words

ldmatrix.x4 → B register r=0..3, half v, lane l
     l4 l3 l2 l1 l0 r1 r0  v
 h4 [ 0  0  0  0  0  1  0  0 ]
 h3 [ 0  0  0  0  0  0  1  0 ]
 h2 [ 0  0  0  1  0  0  0  0 ]
 h1 [ 0  0  0  0  1  0  0  0 ]
 h0 [ 0  0  0  0  0  0  0  1 ]
 t2 [ 1  0  0  0  0  0  0  0 ]
 t1 [ 0  1  0  0  0  0  0  0 ]
 t0 [ 0  0  1  0  0  0  0  0 ]
```

### D. C → packed Y

```text
m16n8 fragment       t0 t1 t2 t3 t4 t5 t6 t7
                    +------------------------+
rows  0..7: Gate    | I8 × token8            |
rows 8..15: Up      | same I8 × token8       |
                    +------------------------+

C[0,1] = Gate; C[2,3] = Up
C register c=0..3, lane l; b=0 Gate, b=1 Up
     l4 l3 l2 l1 l0 c1 c0
  b [ 0  0  0  0  0  1  0 ]
 i2 [ 1  0  0  0  0  0  0 ]
 i1 [ 0  1  0  0  0  0  0 ]
 i0 [ 0  0  1  0  0  0  0 ]
 t2 [ 0  0  0  1  0  0  0 ]
 t1 [ 0  0  0  0  1  0  0 ]
 t0 [ 0  0  0  0  0  0  1 ]

XOR 4: l2 flips → adjacent I row
writer l2 supplies token parity; packed half v supplies I parity

I8 × token8 output; a = local BF16 address
     l4 l3 l2 l1 l0  v
 a5 [ 0  0  0  1  0  0 ]
 a4 [ 0  0  0  0  1  0 ]
 a3 [ 0  0  1  0  0  0 ]
 a2 [ 1  0  0  0  0  0 ]
 a1 [ 0  1  0  0  0  0 ]
 a0 [ 0  0  0  0  0  1 ]
I_tile = tile_base + 8*warp
store = payload + 32*(I_tile/8) + (a>>1)
low/high BF16 = even/odd I; 32 words/warp
```

### E. Y → K3

```text
Y: | expert 0 slab | expert 1 slab | ... |

expert slab
+------------------------+------------------------+-----+
| count + 7 zero words   | batch 0                | ... |
+------------------------+------------------------+-----+

batch record
+-------------+-----------------------------------+
| 8 token IDs | payload: ceil(I/8) × token8 × 4   |
+-------------+-----------------------------------+

Ib = ceil(I/8); B = ceil(N/8)
S = 8+32*Ib; D = 8+B*S; allocate E*D words
record  = Y + e*D + 8 + batch*S
payload = record + 8
count = valid batches; padding = 7 zero words
```

Payload uses appendix C's memory map with I replacing H. Tail IDs = `0xffffffff`; padded values = 0. Read only `count` batches. Fixed capacity reserves N rounded to eight slots/expert; token IDs drive K3 scatter.

### F. PTX helper contracts

- `swiglu_topkw_f32(x,y,w)`: x = Gate, y = Up (FP32 bits); w = `[BF16 weight | token ID]`. `exp(-abs(x))` keeps the exponent nonpositive.
- `cvt_bf16x2_f32(x,y)`: low = x; high = y; one BF16 rounding after SwiGLU × topkW.
- `ldcg_b16x8(src,dst,size)`: size = 0..8 valid BF16 elements; zero-fill the remainder.
