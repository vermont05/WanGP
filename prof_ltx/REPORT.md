# LTX-2.3 (22B GGUF Q4_K_M) profiling report — WanGP on gfx1102

Machine: AMD Radeon RX 7600M XT (8 GB, ~2 GB/s host→GPU link), 32 GB RAM, 8 GB swap.
Run: `--profile 4` (weights pinned in host RAM, streamed to GPU), `--attention sdpa`, 1280x720, 121 frames, 8 steps.
Model: `ltx2_22B_distilled_gguf_q4_k_m` (12.07 GB GGUF) + Gemma-3-12B int8 text encoder. Two clips, one session (clip 2 reuses loaded models).
GGUF runtime: `[llama.cpp CUDA] kernels unavailable, using fallback` → PyTorch dense dequant + `F.linear` (per the code path in `shared/qtypes/gguf.py`).

## REPEATJSON (authoritative per-phase durations)
```
clip1 seed42 total=1021.9s Loading=103.2 EncTextPrompt=312.5 ProcTextEmb=58.2 Denoise1=215.1 VAEDec=37.4 Denoise2=283.2 AudioVAE=4.2 Save=6.7
clip2 seed7  total=655.0s  EncTextPrompt=122.8 ProcTextEmb=101.1 Denoise1=213.9 VAEDec=7.6 Denoise2=202.3 AudioVAE=4.4 Save=2.3
```

## Per-phase system metrics (from samples.csv; windows are label-lagged approximations)
format: gpu_busy mean/max %, cpu mean %, disk_read total GB (mean MB/s), max swap GiB, peak RAM GiB

| phase | clip | dur(s) | gpu% m/x | cpu% | disk GB(mB/s) | swap GiB | RAM GiB |
|---|---|---|---|---|---|---|---|
| Encoding Text Prompt | 1 (cold) | 312.5 | 3.8 / 91 | 106 | 0.951 (3.47) | 4.37 | 16.60 |
| Encoding Text Prompt | 2 (warm) | 122.8 | 7.2 / 73 | 109 | 0.083 (0.75) | 4.07 | 18.11 |
| Processing Text Emb | 1 | 58.2 | 14.9 / 98 | 123 | 0.011 (0.23) | 4.00 | 16.35 |
| Processing Text Emb | 2 | 101.1 | 6.9 / 73 | 130 | 0.056 (0.68) | 4.11 | 18.08 |
| Denoising First | 1 | 215.1 | 99.2 / 100 | 200 | 0.001 (0.00) | 3.99 | 16.66 |
| Denoising First | 2 | 213.9 | 99.6 / 100 | 200 | 0.000 (0.00) | 4.00 | 17.40 |
| Denoising Second | 1 | 283.2 | 88.3 / 100 | 200 | 0.032 (0.16) | 4.01 | 17.49 |
| Denoising Second | 2 | 202.3 | 99.5 / 100 | 200 | 0.000 (0.00) | 4.00 | 17.68 |
| VAE Decoding | 1 | 37.4 | 24.2 / 100 | 176 | 0.023 (0.20) | 4.02 | 17.79 |
| VAE Decoding | 2 | 7.6 | 65.1 / 100 | 180 | 0.000 (0.00) | 4.00 | 17.67 |

Session peak RAM 18.3 GiB; swap stayed ~4.0 GiB (no swap thrash). Disk reads are negligible everywhere except clip-1 cold text encoding (0.95 GB).

## Kernel micro-benchmark (linear_bench.py) — 3 largest 2-D Q*_K weights (all 16384×4096, 67.1M elem)
Tensors: `ff.net.0.proj` Q5_K, `ff.net.2` Q6_K, `ff.net.0.proj` Q5_K. Median of 5 after 2 warmups.

| op | Q5_K 16k×4k | Q6_K 4k×16k | Q5_K 16k×4k |
|---|---|---|---|
| (a) H2D copy raw bytes | 23.5 ms (1.96 GB/s) | 28.1 ms (1.96 GB/s) | 23.6 ms (1.96 GB/s) |
| (b) GPU dequant→bf16 | 13.3 ms (13.6 GB/s) | 10.4 ms (18.2 GB/s) | 12.6 ms (14.3 GB/s) |
| (c) linear 3520 tok | 20.8 ms (22.7 TF) | 19.2 ms (24.6 TF) | 18.9 ms (25.0 TF) |
| (c) linear 14080 tok | 77.8 ms (24.3 TF) | 76.0 ms (24.9 TF) | 74.0 ms (25.5 TF) |

File census (all quantized 2-D tensors): Q4_K 1188 (11.65B elem, 6.55 GB), Q6_K 336 (5.84B, 4.79 GB), Q5_K 108 (1.06B, 0.73 GB). **Total 1632 tensors, 18.55B elements, 12.07 GB.**

Per-denoising-step estimate (scale largest-tensor rates to all 12.07 GB / 18.55B elem; dequant+transfer are token-independent, matmul ∝ tokens):

| tokens | dequant ms | matmul ms | transfer ms |
|---|---|---|---|
| 3520 | 3680 | 5746 | 6159 |
| 14080 | 3680 | 21516 | 6159 |

## Q1 — Where does "Encoding Text Prompt" / "Processing Text Embeddings" time go?
**Not GPU compute, not disk. It is host→GPU weight streaming + CPU work.**
- GPU is nearly idle: mean busy 3.8% (cold) / 7.2% (warm) for Encoding, 14.9% / 6.9% for Processing — the 8 GB GPU spends >90% of the phase starved, waiting for the Gemma-12B / connector weights to arrive from pinned RAM over the ~2 GB/s link.
- Disk is not the cause: only 0.95 GB read (cold) / 0.08 GB (warm) over the whole phase; at any realistic NVMe rate that is <10 s, not 300 s.
- CPU runs ~106–130% (≈1 core) doing tokenization, int8 handling and the per-layer shuttle.
- **Lower when already loaded: yes.** Encoding Text Prompt drops 312.5 s (cold) → 122.8 s (warm), a 61% cut; the cold delta is first-time weight population (0.95 GB vs 0.08 GB disk + page faults). Even warm it stays transfer/CPU-bound (GPU 7.2% busy), so a new prompt still costs ~123 s.

## Q2 — In one denoising step: dequant vs matmul vs H2D transfer?
Measured GPU is ~99% busy during denoising (compute-bound, transfers overlapped), so the step time is set by GPU dequant+matmul with the H2D weight copy pipelined behind it. Scaling the micro-benchmark to all 12.07 GB of weights:
- **The GGUF overhead (dequant + H2D transfer ≈ 3.7 s + 6.2 s ≈ 9.9 s/step) is token-count-independent and large.**
- At the low-token phase (3520): useful matmul is only 5.7 s — so **~64% of the step (dequant 3.7 s + transfer 6.2 s) is weight materialization/movement, not matmul FLOPs.** Dequantization alone (3.7 s) rivals the matmul.
- At the high-token phase (14080): matmul grows to 21.5 s and dominates (≈2/3), with dequant (3.7 s) + transfer (6.2 s) the remaining ≈1/3.
- The estimate is a lower bound: real steps also pay for the many small Q4_K tensors (lower GEMM efficiency than the 16k×4k probe) and SDPA attention, which is why measured ≈27 s/step (phase 1) exceeds the 9–16 s from the largest-tensor scaling. The robust conclusion: **dequantization and host→GPU weight transfer are first-order costs that rival the matmuls at low token counts and remain a sizable fraction at high token counts** — the dense-dequant fallback is the dominant lever, not the GEMMs.
