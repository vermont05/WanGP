# LTX-2.3 22B distilled Q4_K_M GGUF — bench summary (GPU 1, gfx1102, WanGP v19.14)
Cmd: `ROCR_VISIBLE_DEVICES=1 MIOPEN_FIND_MODE=FAST PYTHONUNBUFFERED=1 .venv/bin/python bench_wan.py ltx2_22B_distilled_gguf_q4_k_m 8 1280x720 121`

BENCH model=ltx2_22B_distilled_gguf_q4_k_m steps=8 res=1280x720 frames=121 total=966s
BENCHJSON {"model": "ltx2_22B_distilled_gguf_q4_k_m", "steps": 8, "res": "1280x720", "frames": 121, "init_s": 9.3, "gen_s": 956.7, "total_s": 965.9, "peak_vram_gib": 4.3, "base_vram_gib": 0.03, "torch_max_reserved_gib": 3.97, "peak_rss_gib": 25.84, "peak_sys_used_gib": 19.84, "success": true, "files": ["/home/vermont/Wan2GP/outputs/2026-11-04-09h57m50s_seed42_A golden retriever runs along a sunny beach at sunset, waves rolling in, slow tracking shot, cinemat.mp4"]}

First marks — Encoding: 43.20s ("Encoding Prompt") | Denoising: 296.43s ("Denoising First Phase") | VAE: 296.43s ("VAE Encoding")

Distinct phase labels, first-seen seconds:
| Phase | s |
|---|---|
| Encoding Prompt | 43.20 |
| Encoding Text Prompt | 43.20 |
| Processing Text Embeddings | 172.25 |
| VAE Encoding | 296.43 |
| Denoising First Phase | 296.43 |
| VAE Decoding | 589.94 |
| Denoising Second Phase | 621.77 |
| Audio VAE Decoding | 909.94 |
| Saving File 2026-11-04-0...in, slow tracking shot, cinemat.mp4 | 944.22 |
`Loading - *` sub-phases (57, first-seen s):
- Preparing Models 1.14 | Reading Weights Transformer Model 1.51 | Preparing Weights Transformer Model 2.00 | Applying Weights Transformer Model 2.05
- Weights Loaded Transformer Model 4.17 | Reading Weights 4.27 | Preparing Weights 4.27 | Applying Weights 4.27
- Weights Loaded 4.27 | Reading Configuration Text Encoder Model 4.53 | Building Text Encoder Model 4.53 | Reading Weights Text Encoder Model 5.14
- Preparing Weights Text Encoder Model 5.30 | Applying Weights Text Encoder Model 5.32 | Weights Loaded Text Encoder Model 5.63 | Preparing Transformer Model 7.13
- Scanning Sizes Transformer Model 7.30 | Preparing Text Encoder Model 7.41 | Scanning Sizes Text Encoder Model 7.44 | Preparing Text Embedding Projection Model 7.48
- Scanning Sizes Text Embedding Projection Model 7.48 | Preparing VAE Model 7.48 | Scanning Sizes VAE Model 7.48 | Preparing Video Encoder Model 7.48
- Scanning Sizes Video Encoder Model 7.48 | Preparing Audio Encoder Model 7.48 | Scanning Sizes Audio Encoder Model 7.49 | Preparing Audio Decoder Model 7.49
- Scanning Sizes Audio Decoder Model 7.49 | Preparing Vocoder Model 7.49 | Scanning Sizes Vocoder Model 7.50 | Preparing Spatial Upsampler Model 7.51
- Scanning Sizes Spatial Upsampler Model 7.51 | Preparing Text Embeddings Connector Model 7.51 | Scanning Sizes Text Embeddings Connector Model 7.53 | Hooks and LoRA Slots Transformer Model 7.57
- Finalization Transformer Model 7.78 | Hooks and LoRA Slots Text Encoder Model 7.85 | Finalization Text Encoder Model 7.90 | Hooks and LoRA Slots Text Embedding Projection Model 7.91
- Finalization Text Embedding Projection Model 7.91 | Hooks and LoRA Slots VAE Model 7.91 | Finalization VAE Model 7.92 | Hooks and LoRA Slots Video Encoder Model 7.92
- Finalization Video Encoder Model 7.92 | Hooks and LoRA Slots Audio Encoder Model 7.93 | Finalization Audio Encoder Model 7.94 | Hooks and LoRA Slots Audio Decoder Model 7.94
- Finalization Audio Decoder Model 7.94 | Hooks and LoRA Slots Vocoder Model 7.94 | Finalization Vocoder Model 7.99 | Hooks and LoRA Slots Spatial Upsampler Model 7.99
- Finalization Spatial Upsampler Model 8.00 | Hooks and LoRA Slots Text Embeddings Connector Model 8.00 | Finalization Text Embeddings Connector Model 8.01 | Pinning Model Transformer to Reserved RAM 8.96
- Finalization 42.69

tqdm bars that reached 100% (final frame of each bar):
- `100%|██████████| 8/8 [04:53<00:00, 36.67s/it]`
- `100%|██████████| 3/3 [04:11<00:00, 83.69s/it]`
- note: the "Encoding Text Prompt" bar's last logged frame was 47/48 (98%); no 100% frame was written to the log

Keyword lines ([Kernels]/GGUF/gguf/Traceback/Error/OOM/Killed/fallback):
- Loading Model 'ckpts/ltx-2.3-22b-distilled-Q4_K_M_light.gguf' ...
- [GGUF][llama.cpp CUDA] kernels unavailable, using fallback
- [Kernels] Segmented Norm Modulation: pytorch (fast).
- 0%|          | 0/8 [00:03<?, ?it/s][GGUF] linear backend=PyTorch dense weight materialization for Q5_K.
- [GGUF] linear backend=PyTorch dense weight materialization for Q6_K.
- [Kernels] Compact Split RoPE: pytorch (fast).
- [Kernels] Frame Scale/Shift: pytorch (fast).
- 0%|          | 0/8 [00:03<?, ?it/s][GGUF] linear backend=PyTorch dense weight materialization for Q4_K.
- [Kernels] AdaLN: pytorch (fast).
- [Kernels] Scale/Shift: pytorch (fast).
- [Kernels] Streamed Conv3D: pytorch (fast).
- [Kernels] PixelNorm + SiLU: pytorch (fast).
- BENCH model=ltx2_22B_distilled_gguf_q4_k_m steps=8 res=1280x720 frames=121 total=966s
- (no Traceback / Error / out of memory / OOM / Killed lines in log)

Last 15 lines of log:
`[bench  883.9s] VAE Decoding 1/16`
`[Kernels] PixelNorm + SiLU: pytorch (fast).`
`[bench  884.4s] VAE Decoding 10/16`
`[bench  884.7s] VAE Decoding 11/16`
`[bench  885.4s] VAE Decoding 12/16`
`[bench  886.0s] VAE Decoding 13/16`
`[bench  887.3s] VAE Decoding 14/16`
`[bench  888.7s] VAE Decoding 15/16`
`[bench  918.7s] VAE Decoding 16/16`
`[bench  919.2s] Audio VAE Decoding`
`[bench  953.5s] Saving File 2026-11-04-0...in, slow tracking shot, cinemat.mp4`
`Video file saved to Path: outputs/2026-11-04-09h57m50s_seed42_A golden retriever runs along a sunny beach at sunset, waves rolling in, slow tracking shot, cinemat.mp4`
`success True [] ['/home/vermont/Wan2GP/outputs/2026-11-04-09h57m50s_seed42_A golden retriever runs along a sunny beach at sunset, waves rolling in, slow tracking shot, cinemat.mp4']`
BENCHJSON {"model": "ltx2_22B_distilled_gguf_q4_k_m", "steps": 8, "res": "1280x720", "frames": 121, "init_s": 9.3, "gen_s": 956.7, "total_s": 965.9, "peak_vram_gib": 4.3, "base_vram_gib": 0.03, "torch_max_reserved_gib": 3.97, "peak_rss_gib": 25.84, "peak_sys_used_gib": 19.84, "success": true, "files": ["/home/vermont/Wan2GP/outputs/2026-11-04-09h57m50s_seed42_A golden retriever runs along a sunny beach at sunset, waves rolling in, slow tracking shot, cinemat.mp4"], "marks": [[1.1386640071868896, "Loading - Preparing Models", null], [1.5052037239074707, "Loading - Reading Weights Transformer Model", null], [2.002474308013916, "Loading - Preparing Weights Transformer Model", null], [2.0537447929382324, "Loading - Applying Weights Transformer Model", null], [4.167571067810059, "Loading - Weights Loaded Transformer Model", null], [4.265199184417725, "Loading - Reading Weights", null], [4.265331983566284, "Loading - Preparing Weights", null], [4.265445947647095, "Loading - Applying Weights", null], [4.2655251026153564, "Loading - Weights Loaded", null], [4.287583589553833, "Loading - Reading Weights", null], [4.2886962890625, "Loading - Preparing Weights", null], [4.288761854171753, "Loading - Applying Weights", null], [4.28882098197937, "Loading - Weights Loaded", null], [4.399250030517578, "Loading - Reading Weights", null], [4.429454565048218, "Loading - Preparing Weights", null], [4.430178165435791, "Loading - Applying Weights", null], [4.472684144973755, "Loading - Weights Loaded", null], [4.482905864715576, "Loading - Reading Weights", null], [4.482984781265259, "Loading - Preparing Weights", null], [4.483362674713135, "Loading - Applying Weights", null], [4.483561277389526, "Loading - Weights Loaded", null], [4.5223143100738525, "Loading - Reading Weights", null], [4.522392272949219, "Loading - Preparing Weights", null], [4.522442579269409, "Loading - Applying Weights", null], [4.522493839263916, "Loading - Weights Loaded", null], [4.533566236495972, "Loading - Reading Configuration Text Encoder Model", null], [4.533623695373535, "Loading - Building Text Encoder Model", null], [5.135613918304443, "Loading - Reading Weights Text Encoder Model", null], [5.298810243606567, "Loading - Preparing Weights Text Encoder Model", null], [5.319322109222412, "Loading - Applying Weights Text Encoder Model", null], [5.631232976913452, "Loading - Weights Loaded Text Encoder Model", null], [7.101258754730225, "Loading - Reading Weights", null], [7.110389471054077, "Loading - Preparing Weights", null], [7.115635871887207, "Loading - Applying Weights", null], [7.126487970352173, "Loading - Weights Loaded", null], [7.1276936531066895, "Loading - Preparing Transformer Model", null], [7.301226615905762, "Loading - Scanning Sizes Transformer Model", null], [7.410819053649902, "Loading - Preparing Text Encoder Model", null], [7.44242787361145, "Loading - Scanning Sizes Text Encoder Model", null], [7.4760119915008545, "Loading - Preparing Text Embedding Projection Model", null], [7.476227283477783, "Loading - Scanning Sizes Text Embedding Projection Model", null], [7.476513147354126, "Loading - Preparing VAE Model", null], [7.47656774520874, "Loading - Scanning Sizes VAE Model", null], [7.476683616638184, "Loading - Preparing Video Encoder Model", null], [7.476890563964844, "Loading - Scanning Sizes Video Encoder Model", null], [7.476983070373535, "Loading - Preparing Audio Encoder Model", null], [7.488464593887329, "Loading - Scanning Sizes Audio Encoder Model", null], [7.488530158996582, "Loading - Preparing Audio Decoder Model", null], [7.489615201950073, "Loading - Scanning Sizes Audio Decoder Model", null], [7.489650249481201, "Loading - Preparing Vocoder Model", null], [7.500668048858643, "Loading - Scanning Sizes Vocoder Model", null], [7.514501094818115, "Loading - Preparing Spatial Upsampler Model", null], [7.514614820480347, "Loading - Scanning Sizes Spatial Upsampler Model", null], [7.5148069858551025, "Loading - Preparing Text Embeddings Connector Model", null], [7.526848793029785, "Loading - Scanning Sizes Text Embeddings Connector Model", null], [7.566911935806274, "Loading - Hooks and LoRA Slots Transformer Model", null], [7.779693841934204, "Loading - Finalization Transformer Model", null], [7.846112012863159, "Loading - Hooks and LoRA Slots Text Encoder Model", null], [7.897625684738159, "Loading - Finalization Text Encoder Model", null], [7.909632205963135, "Loading - Hooks and LoRA Slots Text Embedding Projection Model", null], [7.909801959991455, "Loading - Finalization Text Embedding Projection Model", null], [7.914754152297974, "Loading - Hooks and LoRA Slots VAE Model", null], [7.919039011001587, "Loading - Finalization VAE Model", null], [7.920476913452148, "Loading - Hooks and LoRA Slots Video Encoder Model", null], [7.924257516860962, "Loading - Finalization Video Encoder Model", null], [7.932859182357788, "Loading - Hooks and LoRA Slots Audio Encoder Model", null], [7.939958333969116, "Loading - Finalization Audio Encoder Model", null], [7.939996957778931, "Loading - Hooks and LoRA Slots Audio Decoder Model", null], [7.940202236175537, "Loading - Finalization Audio Decoder Model", null], [7.940229654312134, "Loading - Hooks and LoRA Slots Vocoder Model", null], [7.985159158706665, "Loading - Finalization Vocoder Model", null], [7.987316370010376, "Loading - Hooks and LoRA Slots Spatial Upsampler Model", null], [7.99521803855896, "Loading - Finalization Spatial Upsampler Model", null], [7.99639630317688, "Loading - Hooks and LoRA Slots Text Embeddings Connector Model", null], [8.006042003631592, "Loading - Finalization Text Embeddings Connector Model", null], [8.95633339881897, "Loading - Pinning Model Transformer to Reserved RAM", null], [42.68884348869324, "Loading - Finalization", null], [43.20203375816345, "Encoding Prompt", null], [43.202231884002686, "Encoding Text Prompt", 0], [46.31337380409241, "Encoding Text Prompt", 1], [48.41073203086853, "Encoding Text Prompt", 5], [49.706201791763306, "Encoding Text Prompt", 7], [51.575530767440796, "Encoding Text Prompt", 8], [53.44215655326843, "Encoding Text Prompt", 11], [55.86370134353638, "Encoding Text Prompt", 13], [56.808645725250244, "Encoding Text Prompt", 14], [57.31613254547119, "Encoding Text Prompt", 16], [58.87539887428284, "Encoding Text Prompt", 18], [60.43032717704773, "Encoding Text Prompt", 20], [61.458587884902954, "Encoding Text Prompt", 22], [62.75849652290344, "Encoding Text Prompt", 23], [63.539825201034546, "Encoding Text Prompt", 25], [64.49839925765991, "Encoding Text Prompt", 28], [64.90179204940796, "Encoding Text Prompt", 31], [65.87502837181091, "Encoding Text Prompt", 32], [67.55520796775818, "Encoding Text Prompt", 33], [70.30929064750671, "Encoding Text Prompt", 34], [83.05084109306335, "Encoding Text Prompt", 35], [97.37984466552734, "Encoding Text Prompt", 36], [123.92305731773376, "Encoding Text Prompt", 37], [164.38909769058228, "Encoding Text Prompt", 38], [164.7439305782318, "Encoding Text Prompt", 40], [169.537043094635, "Encoding Text Prompt", 42], [169.9947805404663, "Encoding Text Prompt", 45], [171.9134006500244, "Encoding Text Prompt", 46], [172.25339698791504, "Encoding Text Prompt", 48], [172.2544960975647, "Processing Text Embeddings", null], [296.425256729126, "VAE Encoding", null], [296.4253990650177, "Denoising First Phase", 0], [330.71631717681885, "Denoising First Phase", 1], [381.87929582595825, "Denoising First Phase", 2], [418.79798555374146, "Denoising First Phase", 3], [476.52165365219116, "Denoising First Phase", 4], [503.00041341781616, "Denoising First Phase", 5], [529.823972940445, "Denoising First Phase", 6], [556.3788249492645, "Denoising First Phase", 7], [589.9436202049255, "VAE Decoding", 8], [621.7734627723694, "VAE Encoding", null], [621.7736215591431, "Denoising Second Phase", 0], [680.5478994846344, "Denoising Second Phase", 1], [815.6077833175659, "Denoising Second Phase", 2], [873.578565120697, "VAE Decoding", 3], [873.5930986404419, "VAE Decoding", 0], [874.6598155498505, "VAE Decoding", 1], [875.1339883804321, "VAE Decoding", 10], [875.4722270965576, "VAE Decoding", 11], [876.135537147522, "VAE Decoding", 12], [876.7598159313202, "VAE Decoding", 13], [878.0174102783203, "VAE Decoding", 14], [879.4743633270264, "VAE Decoding", 15], [909.4432411193848, "VAE Decoding", 16], [909.9415380954742, "Audio VAE Decoding", null], [944.2204024791718, "Saving File 2026-11-04-0...in, slow tracking shot, cinemat.mp4", null]]}
`BENCH model=ltx2_22B_distilled_gguf_q4_k_m steps=8 res=1280x720 frames=121 total=966s`

df -h ~ at end:
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  468G  384G   60G  87% /
