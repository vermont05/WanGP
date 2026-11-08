# FastWan 3-step 720p 121f benchmark — optimized (fastwan_opt)

Date: 2026-11-04 · GPU: ROCR_VISIBLE_DEVICES=1 · MIOPEN_FIND_MODE=FAST

## BENCHJSON (marks list omitted)
```
BENCHJSON {"model": "ti2v_2_2_fastwan", "steps": 3, "res": "1280x720", "frames": 121, "init_s": 8.0, "gen_s": 395.0, "total_s": 403.0, "peak_vram_gib": 6.19, "base_vram_gib": 0.03, "torch_max_reserved_gib": 5.7, "peak_rss_gib": 20.46, "peak_sys_used_gib": 13.05, "success": true, "files": ["/home/vermont/Wan2GP/outputs/2026-11-04-06h49m26s_seed42_A golden retriever runs along a sunny beach at sunset, waves rolling in, slow tracking shot, cinemat.mp4"]}
```

## Phase start marks (seconds, from marks list)
- first "Encoding" mark (Encoding Prompt): 11.275866985321045 s
- first "Denoising" mark (Denoising 0/3): 15.744635343551636 s
- first "VAE" mark (VAE Decoding): 180.2225615978241 s

## Last denoise progress bar (tqdm line containing "s/it]")
```
100%|██████████| 3/3 [02:44<00:00, 54.77s/it]100%|██████████| 3/3 [02:44<00:00, 54.83s/it]
```

## [Kernels] lines
```
[Kernels] Segmented Norm Modulation: pytorch (fast).
```

## Traceback / Error / out of memory
(none found in log)

## compare.py output
```
shapes (121, 704, 1280, 3) (121, 704, 1280, 3) mean_abs_diff 1.645 PSNR_dB 40.89
```
