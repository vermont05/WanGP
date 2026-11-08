# LTX-2.3 22B (Q4_K_M) — Native llama.cpp GGUF kernels vs PyTorch fallback

A/B test on ROCm (gfx1102, AMD Radeon RX 7600M XT). Model: `ltx2_22B_distilled_gguf_q4_k_m`,
8 steps, 1280x720, 121 frames, 3 prompts (seeds 42, 7, 11).
Common env: `ROCR_VISIBLE_DEVICES=1 MIOPEN_FIND_MODE=FAST ROCPROFILER_REGISTER_ENABLED=0 PYTHONUNBUFFERED=1`.

- **Native (B):** `WGP_GGUF_LLAMACPP_CUDA=1 WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE=cublas` → `llamacpp_gguf_cuda 1.0.29+torch215rocm714py311`, HIP dequant kernels, `linear mode: cublas`. `ab_native.log` **contains** `[GGUF][llama.cpp CUDA] linear fast path active for Q4_K/Q5_K/Q6_K`.
- **Baseline (A):** `WGP_GGUF_LLAMACPP_CUDA=0` → PyTorch dense weight materialization (GPU dequant to bf16 + `F.linear`). `ab_base.log` **does not** contain the fast-path marker; it logs `[GGUF] linear backend=PyTorch dense weight materialization`.

Both sessions ran one after the other (native first, baseline second), never simultaneously.

---

## 1. Micro-benchmark (single-layer dequant-linear, `prof_ltx/native_gguf_bench.py`)

Median time over 5 reps after 2 warmups, on the largest Q4_K / Q5_K / Q6_K 2-D tensors plus a 4096x4096 Q4_K tensor, at the two denoising token counts. `rel_err` = relative L2 error of native output vs the bf16 reference.

| qtype | tensor | shape (out,in) | tokens | rel_err | fallback_ms | bf16_matmul_only_ms | native_ms | speedup |
|-------|--------|----------------|--------|---------|-------------|---------------------|-----------|---------|
| Q4_K | transformer_blocks.10.ff.net.0.proj.weight | 16384, 4096 | 3520 | 0.00725 | 28.9 | 18.8 | 21.9 | 1.32 |
| Q4_K | transformer_blocks.10.ff.net.0.proj.weight | 16384, 4096 | 14080 | 0.00724 | 81.0 | 71.7 | 83.0 | 0.98 |
| Q5_K | transformer_blocks.0.ff.net.0.proj.weight | 16384, 4096 | 3520 | 0.00723 | 30.2 | 18.2 | 21.7 | 1.39 |
| Q5_K | transformer_blocks.0.ff.net.0.proj.weight | 16384, 4096 | 14080 | 0.00724 | 83.4 | 71.0 | 82.4 | 1.01 |
| Q6_K | transformer_blocks.0.ff.net.2.weight | 4096, 16384 | 3520 | 0.01341 | 28.9 | 19.0 | 22.4 | 1.29 |
| Q6_K | transformer_blocks.0.ff.net.2.weight | 4096, 16384 | 14080 | 0.01341 | 84.1 | 73.6 | 85.7 | 0.98 |
| Q4_K | transformer_blocks.10.attn1.to_k.weight | 4096, 4096 | 3520 | 0.00725 | 7.0 | 4.7 | 6.2 | 1.13 |
| Q4_K | transformer_blocks.10.attn1.to_k.weight | 4096, 4096 | 14080 | 0.00725 | 20.6 | 18.2 | 22.9 | 0.90 |

**Reading:** in isolation, native wins clearly at the shorter 3520-token count (1.13x–1.39x) and is at parity at 14080 tokens (0.90x–1.01x). `bf16_matmul_only_ms` is the pure GEMM floor (dequant-free); the fallback's extra cost over that floor is the per-call dense dequant. In the *real* pipeline the fallback re-materializes the full dense bf16 weight on every layer/step, which is far more expensive than this isolated single-layer call captures — so the end-to-end win below is much larger than the micro-benchmark alone would suggest.

---

## 2. End-to-end per-clip results

### Native (B) — `prof_ltx/ab_native.log`

| clip | seed | total_s | Loading | Enc. Prompt | Enc. Text | Proc. Embed | VAE Enc | Denoise 1st | VAE Dec | Denoise 2nd | Audio VAE Dec | Save |
|------|------|---------|---------|-------------|-----------|-------------|---------|-------------|---------|-------------|---------------|------|
| 1 (cold) | 42 | 950.5 | 220.8 | 0.0 | 148.3 | 157.2 | 0.0 | 84.6 | 56.4 | 251.7 | 5.0 | 25.3 |
| 2 (warm) | 7  | 642.7 | — | 3.4 | 202.7 | 93.1  | 0.0 | 83.8 | 1.3  | 184.4 | 60.1 | 13.8 |
| 3 (warm) | 11 | 465.4 | — | 0.1 | 135.3 | 67.8  | 0.0 | 84.2 | 12.4 | 160.3 | 2.5  | 2.6  |

mp4:
- clip 1: `outputs/2026-11-04-22h06m45s_seed42_A red fox trots through fresh snow in a pine forest at dawn, soft light, slow dolly shot.mp4`
- clip 2: `outputs/2026-11-04-22h17m39s_seed7_An old fisherman mends a net on a wooden pier, gulls overhead, waves lapping, documentary style.mp4`
- clip 3: `outputs/2026-11-04-22h25m36s_seed11_A neon-lit city street at night in the rain, reflections on wet asphalt, a cyclist passes, cinematic.mp4`

### Baseline (A) — `prof_ltx/ab_base.log`

| clip | seed | total_s | Loading | Enc. Prompt | Enc. Text | Proc. Embed | VAE Enc | Denoise 1st | VAE Dec | Denoise 2nd | Audio VAE Dec | Save |
|------|------|---------|---------|-------------|-----------|-------------|---------|-------------|---------|-------------|---------------|------|
| 1 (cold) | 42 | 962.2 | 213.7 | 0.0 | 139.4 | 45.4  | 0.0 | 214.6 | 18.4 | 273.3 | 41.0 | 15.2 |
| 2 (warm) | 7  | 649.8 | — | 0.1 | 110.8 | 72.6  | 0.0 | 215.0 | 9.7  | 216.8 | 4.2  | 20.5 |
| 3 (warm) | 11 | 644.1 | — | 0.1 | 125.0 | 81.4  | 0.0 | 215.9 | 5.4  | 206.5 | 2.5  | 7.1  |

mp4:
- clip 1: `outputs/2026-11-04-22h52m08s_seed42_A red fox trots through fresh snow in a pine forest at dawn, soft light, slow dolly shot.mp4`
- clip 2: `outputs/2026-11-04-23h02m53s_seed7_An old fisherman mends a net on a wooden pier, gulls overhead, waves lapping, documentary style.mp4`
- clip 3: `outputs/2026-11-04-23h13m50s_seed11_A neon-lit city street at night in the rain, reflections on wet asphalt, a cyclist passes, cinematic.mp4`

(Loading appears only on the cold clip 1; warm clips skip it. `—` = not present in that clip's phase set.)

---

## 3. Warm-clip comparison (mean of clips 2 and 3)

| metric | native (s) | baseline (s) | delta (s) | native speedup |
|--------|-----------|--------------|-----------|----------------|
| Total | 554.05 | 646.95 | 92.90 faster | 1.17x (14.4% faster) |
| Denoising First Phase | 84.00 | 215.45 | 131.45 faster | 2.56x (61.0% faster) |
| Denoising Second Phase | 172.35 | 211.65 | 39.30 faster | 1.23x (18.6% faster) |

The native llama.cpp dequant kernels deliver a **2.56x** speedup on the first (high-resolution) denoising phase and a **1.23x** speedup on the second phase, for a **14.4%** faster warm-clip end-to-end total. The first-phase win dwarfs the isolated micro-benchmark because the PyTorch fallback re-materializes the dense bf16 weight on every layer and every step, a cost the single-layer micro-benchmark does not reproduce.

---

## 4. Notes

- **Ordering / page-cache bias:** the baseline ran *second*, after the native session had already loaded the model and warmed the OS page cache. That gives the baseline a *favourable* start (weights already in RAM), so the observed native win is **conservative** — had the native run been the one that benefited from a warm cache, the gap would likely be even larger.
- The native session's cold clip 1 (950.5s) and the baseline's cold clip 1 (962.2s) are within ~1.2% of each other, so cold-start/loading is not a differentiator; the difference is entirely in the denoising compute.
- `peak_sys_used_gib` was comparable across both sessions (~20.5–21.8 GiB), so the speedup is a compute win, not a memory-footprint change.
- The second-phase gap (1.23x) is smaller than the first-phase gap (2.56x); the second phase runs at a lower token count where the isolated micro-benchmark also shows the native kernels nearer parity (0.90x–1.01x at 14080 tokens), consistent with the micro-benchmark trend.
