# VAE Profiling Summary — bench_vae.py (Wan2.2 VAE decode, VAE_PROF=1, GPU 1)

## Job 1 — tile=256 (latent 8x44x80)

`VAEBENCH tile=256 latent=8x44x80 out=(3, 29, 704, 1280) decode=58.8s peak_alloc=2.13GiB peak_reserved=2.64GiB`

Top 12 kernels by device time (name, Self CUDA, Self CUDA %, # calls, CUDA time avg):

| # | Kernel name (as printed) | Device time | % total | Calls | Avg |
|---|---|---|---|---|---|
| 1 | _ZN2ck16tensor_operation6device12_GLOBAL__N_140kernel_grouped_conv_... | 34.975s | 67.69% | 7112 | 4.918ms |
| 2 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::nativ... | 3.112s | 6.02% | 29620 | 105.047us |
| 3 | miopenSp3AsmConv_v30_3_1_gfx11_fp16_dot2_f2x3_stride1.kd | 3.050s | 5.90% | 672 | 4.538ms |
| 4 | batched_transpose_32x64_pack_2x4_ediv_1x2_half.kd | 1.705s | 3.30% | 6496 | 262.428us |
| 5 | batched_transpose_32x16_half.kd | 1.164s | 2.25% | 1556 | 748.337us |
| 6 | batched_transpose_64x64_pack_4x4_ediv_4x4_half.kd | 1.003s | 1.94% | 14168 | 70.826us |
| 7 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::nativ... | 917.826ms | 1.78% | 9935 | 92.383us |
| 8 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::nativ... | 683.711ms | 1.32% | 6748 | 101.320us |
| 9 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::nativ... | 634.690ms | 1.23% | 7440 | 85.308us |
| 10 | batched_transpose_32x64_pack_2x4_ediv_2x4_half.kd | 489.985ms | 0.95% | 2008 | 244.017us |
| 11 | void at::native::vectorized_elementwise_kernel<8, at::native::(anon... | 473.318ms | 0.92% | 6495 | 72.874us |
| 12 | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAF... | 441.296ms | 0.85% | 6728 | 65.591us |

Im3d2Col: not present in this run.
Errors/tracebacks: none. (Warnings only: MIOpen `gfx1102_16.HIP.fdb.txt` unreadable; mem-efficient-attention experimental `UserWarning`.)

## Job 2 — tile=512 (latent 8x44x80)

`VAEBENCH tile=512 latent=8x44x80 out=(3, 29, 704, 1280) decode=49.4s peak_alloc=4.22GiB peak_reserved=5.40GiB`

Top 12 kernels by device time (name, Self CUDA, Self CUDA %, # calls, CUDA time avg):

| # | Kernel name (as printed) | Device time | % total | Calls | Avg |
|---|---|---|---|---|---|
| 1 | _ZN2ck16tensor_operation6device12_GLOBAL__N_140kernel_grouped_conv_... | 31.701s | 66.58% | 2032 | 15.601ms |
| 2 | miopenSp3AsmConv_v30_3_1_gfx11_fp16_dot2_f2x3_stride1.kd | 3.009s | 6.32% | 192 | 15.672ms |
| 3 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::nativ... | 2.627s | 5.52% | 8432 | 311.605us |
| 4 | batched_transpose_32x16_half.kd | 2.363s | 4.96% | 562 | 4.204ms |
| 5 | batched_transpose_64x64_pack_4x4_ediv_4x4_half.kd | 1.098s | 2.31% | 3949 | 278.148us |
| 6 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::nativ... | 795.901ms | 1.67% | 2800 | 284.250us |
| 7 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::nativ... | 572.495ms | 1.20% | 1928 | 296.937us |
| 8 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::nativ... | 544.012ms | 1.14% | 2080 | 261.544us |
| 9 | batched_transpose_32x64_pack_2x4_ediv_1x2_half.kd | 526.059ms | 1.10% | 1856 | 283.437us |
| 10 | void at::native::vectorized_elementwise_kernel<8, at::native::(anon... | 472.144ms | 0.99% | 1856 | 254.388us |
| 11 | void at::native::vectorized_elementwise_kernel<8, at::native::CUDAF... | 462.823ms | 0.97% | 1928 | 240.053us |
| 12 | void at::native::vectorized_elementwise_kernel<8, at::native::AUnar... | 442.572ms | 0.93% | 1928 | 229.550us |

Im3d2Col: not present in this run.
Errors/tracebacks: none. (Warnings only: MIOpen `gfx1102_16.HIP.fdb.txt` unreadable; mem-efficient-attention experimental `UserWarning`.)
