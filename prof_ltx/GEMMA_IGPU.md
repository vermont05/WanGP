# LTX-2.3 with Gemma text encoder on the iGPU

Run: `bash prof_ltx/run_igpu_e2e.sh` (log `prof_ltx/ab_igpu_gemma.log`, watchdog `prof_ltx/ab_igpu_gemma.mem`).
Baseline: `prof_ltx/ab_native.log` — same seeds 42, 7, 11 and prompts; Gemma encoded on the eGPU.

## Outcome

- **Finished successfully** — all 3 runs `success: true`, exit 0. **No `MEMABORT`.**
- `MIN_AVAIL_MB=3506` (from the log). Lowest sampled `avail_MB` in the `.mem` file is also **3506 MB** — well above the 1.5 GB (1500 MB) watchdog abort threshold.

## Per-run phase_seconds and total_s (iGPU Gemma vs native/eGPU Gemma)

Phase names differ between the two logs: the iGPU run reports the Gemma text-encoder step as `Encoding Prompt`, whereas the native baseline splits that same work across `Encoding Prompt` + `Encoding Text Prompt` + `Processing Text Embeddings`. Values below are reported verbatim from each log's `REPEATJSON` line (`–` = phase not present in that run).

| Phase | iGPU r1 | native r1 | iGPU r2 | native r2 | iGPU r3 | native r3 | iGPU mean(2-3) | native mean(2-3) |
|---|---|---|---|---|---|---|---|---|
| Loading | 317.5 | 220.8 | – | – | – | – | – | – |
| Encoding Prompt | 168.0 | 0.0 | 79.8 | 3.4 | 78.5 | 0.1 | 79.2 | 1.8 |
| Encoding Text Prompt | – | 148.3 | – | 202.7 | – | 135.3 | – | 169.0 |
| Processing Text Embeddings | – | 157.2 | – | 93.1 | – | 67.8 | – | 80.5 |
| VAE Encoding | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Denoising First Phase | 84.4 | 84.6 | 84.0 | 83.8 | 84.6 | 84.2 | 84.3 | 84.0 |
| VAE Decoding | 4.6 | 56.4 | 16.8 | 1.3 | 1.4 | 12.4 | 9.1 | 6.9 |
| Denoising Second Phase | 284.6 | 251.7 | 157.2 | 184.4 | 160.9 | 160.3 | 159.1 | 172.4 |
| Audio VAE Decoding | 3.3 | 5.0 | 2.5 | 60.1 | 2.5 | 2.5 | 2.5 | 31.3 |
| Saving File | 2.9 | 25.3 | 3.5 | 13.8 | 10.2 | 2.6 | 6.9 | 8.2 |
| **total_s** | **866.4** | **950.5** | **343.8** | **642.7** | **337.9** | **465.4** | **340.9** | **554.1** |

Like-for-like text-encoding cost (iGPU `Encoding Prompt` vs native sum of the three text phases): r1 168.0 vs 305.5; r2 79.8 vs 299.2; r3 78.5 vs 203.2; mean(2-3) 79.2 vs 251.2. Denoising phases are comparable between the two.

## `[Gemma iGPU]` lines (from the log)

```
[Gemma iGPU] gemma-3-12b-it-qat-Q4_0.gguf ready in 26.2s (48 layers, ctx 256)   # run 1
[Gemma iGPU] 21 tokens encoded in 1.1s
[Gemma iGPU] idle: process stopped, RAM freed
[Gemma iGPU] gemma-3-12b-it-qat-Q4_0.gguf ready in 27.9s (48 layers, ctx 256)   # run 2
[Gemma iGPU] 23 tokens encoded in 0.5s
[Gemma iGPU] idle: process stopped, RAM freed
[Gemma iGPU] gemma-3-12b-it-qat-Q4_0.gguf ready in 26.0s (48 layers, ctx 256)   # run 3
[Gemma iGPU] 23 tokens encoded in 0.9s
[Gemma iGPU] idle: process stopped, RAM freed
```

## RAM watchdog (`ab_igpu_gemma.mem`, 255 samples @ 5 s)

- **Lowest `avail_MB`: 3506** (at 06:36:40, during run 2's encoding window — eGPU model + iGPU Gemma both resident).
- **Gemma was down during denoising:** `gemma=up` in only 15 of 255 samples, in three ~25 s windows (06:27:27–52, 06:36:35–06:37:03, 06:42:21–44) — one per run's text-encoding step. The other 240 samples are `gemma=down`, i.e. the Gemma process is stopped (RAM freed) throughout all denoising phases.

## Output videos

- Run 1 (seed 42): `outputs/2026-11-05-06h36m30s_seed42_A red fox trots through fresh snow in a pine forest at dawn, soft light, slow dolly shot.mp4`
- Run 2 (seed 7): `outputs/2026-11-05-06h42m13s_seed7_An old fisherman mends a net on a wooden pier, gulls overhead, waves lapping, documentary style.mp4`
- Run 3 (seed 11): `outputs/2026-11-05-06h47m44s_seed11_A neon-lit city street at night in the rain, reflections on wet asphalt, a cyclist passes, cinematic.mp4`
