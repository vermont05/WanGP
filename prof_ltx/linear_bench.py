# Part B: Kernel micro-benchmark for WanGP's GGUF Q4_K/Q5_K/Q6_K dequant+linear path.
#
# Opens the LTX-2.3 22B GGUF with gguf.GGUFReader, picks the 3 largest 2-D Q4_K/Q5_K/Q6_K
# tensors, and times on the GPU (torch.cuda.synchronize, median of 5 after 2 warmups):
#   (a) host->GPU copy of the raw quantized bytes (from pinned memory)
#   (b) GPU dequantization to bfloat16 via shared.qtypes.gguf._gguf_dequantize_tensor
#   (c) torch.nn.functional.linear of a bf16 input (3520 and 14080 tokens) vs the bf16 weight
# Then it counts all quantized 2-D tensors in the file and estimates per-denoising-step totals
# by scaling the measured per-element / per-byte / per-FLOP rates.
#
# Run with the GPU env (AFTER Part A has finished):
#   ROCR_VISIBLE_DEVICES=1 MIOPEN_FIND_MODE=FAST PYTHONUNBUFFERED=1 .venv/bin/python prof_ltx/linear_bench.py
import os
import sys
import time
import statistics
from pathlib import Path

import numpy as np
import torch
import gguf

WANGP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WANGP))
os.chdir(WANGP)

GGUF_FILE = str(WANGP / "ckpts" / "ltx-2.3-22b-distilled-Q4_K_M_light.gguf")
DEVICE = "cuda"
TOKEN_COUNTS = [3520, 14080]  # token counts of the two denoising phases
WARMUPS = 2
REPS = 5

# WanGP's GGUF dequantizer (populates _DEQUANTIZE_FUNCTIONS at import).
import shared.qtypes.gguf as wgguf
from shared.qtypes.gguf import _gguf_dequantize_tensor


def qname(t):
    return getattr(t, "name", None) or str(t)


def is_2d(t):
    return len(t.shape) == 2


def median_ms(fn, warmups=WARMUPS, reps=REPS):
    """Run fn() on the GPU `warmups` times, then time `reps` runs; return median ms."""
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.median(times)


def main():
    assert torch.cuda.is_available(), "CUDA/ROCm not available"
    torch.cuda.synchronize()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"file:   {GGUF_FILE}")
    reader = gguf.GGUFReader(GGUF_FILE)
    tensors = reader.tensors
    print(f"total tensors in file: {len(tensors)}")

    K_TYPES = {
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q6_K,
    }
    NON_QUANT = {
        gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F64,
        gguf.GGMLQuantizationType.I8, gguf.GGMLQuantizationType.I16,
        gguf.GGMLQuantizationType.I32, gguf.GGMLQuantizationType.I64,
    }

    # --- pick the 3 largest 2-D Q4_K/Q5_K/Q6_K tensors ---------------------
    kk = [t for t in tensors if is_2d(t) and t.tensor_type in K_TYPES]
    kk.sort(key=lambda t: t.n_elements, reverse=True)
    top3 = kk[:3]
    print("\n=== 3 largest 2-D Q4_K/Q5_K/Q6_K tensors ===")
    for t in top3:
        print(f"  {t.name}  type={qname(t.tensor_type)}  shape(ggml)={[int(x) for x in t.shape]}  "
              f"n_elements={t.n_elements}  n_bytes={t.n_bytes}")

    results = []
    for t in top3:
        # ggml dims are [in_features, out_features]; logical weight is (out, in)
        in_features = int(t.shape[0])
        out_features = int(t.shape[1])
        oshape = (out_features, in_features)
        qtype = t.tensor_type
        n_bytes = t.n_bytes
        n_elem = t.n_elements
        byte_shape = tuple(int(x) for x in np.shape(t.data))

        # raw quantized bytes as a pinned host tensor (byte shape), plus a flat view for the copy
        raw_np = np.ascontiguousarray(t.data, dtype=np.uint8)
        host = torch.from_numpy(raw_np).pin_memory()
        host_flat = host.reshape(-1).contiguous()

        # (a) host->GPU copy of the raw bytes
        gpu_raw = torch.empty_like(host_flat, device=DEVICE)
        med_a = median_ms(lambda: gpu_raw.copy_(host_flat, non_blocking=True))
        gbs_a = n_bytes / (med_a / 1000.0) / 1e9

        # (b) GPU dequantization to bfloat16 (input must be the byte-shaped tensor)
        raw_shaped = gpu_raw.view(byte_shape)

        def deq():
            return _gguf_dequantize_tensor(raw_shaped, qtype, oshape, dtype=torch.bfloat16)

        w_bf16 = deq()  # one warm dequant to keep a valid weight for the linears
        med_b = median_ms(deq)
        gbs_b = (n_bytes + n_elem * 2) / (med_b / 1000.0) / 1e9  # read raw + write bf16

        # (c) bf16 linear at the two token counts
        lin = {}
        for tokens in TOKEN_COUNTS:
            inp = torch.randn(tokens, in_features, dtype=torch.bfloat16, device=DEVICE)
            med_c = median_ms(lambda: torch.nn.functional.linear(inp, w_bf16))
            flops = 2 * tokens * in_features * out_features
            tflops = flops / (med_c / 1000.0) / 1e12
            lin[tokens] = (med_c, tflops)
            del inp
            torch.cuda.empty_cache()

        results.append({
            "name": t.name, "type": qname(t), "in": in_features, "out": out_features,
            "n_bytes": n_bytes, "n_elem": n_elem,
            "copy_ms": med_a, "copy_gbs": gbs_a,
            "deq_ms": med_b, "deq_gbs": gbs_b, "lin": lin,
        })
        print(f"\n--- {t.name}  [{qname(t)}]  weight {out_features}x{in_features}  "
              f"(n_bytes={n_bytes}, n_elem={n_elem}) ---")
        print(f"  (a) H2D copy raw bytes : {med_a:9.3f} ms   {gbs_a:7.2f} GB/s")
        print(f"  (b) GPU dequant ->bf16 : {med_b:9.3f} ms   {gbs_b:7.2f} GB/s (raw read + bf16 write)")
        for tokens in TOKEN_COUNTS:
            med_c, tflops = lin[tokens]
            print(f"  (c) linear {tokens:6d} tok   : {med_c:9.3f} ms   {tflops:7.2f} TFLOPS")

        del host, host_flat, gpu_raw, raw_shaped, w_bf16
        torch.cuda.empty_cache()

    # --- census of all quantized 2-D tensors --------------------------------
    print("\n=== census: all quantized 2-D tensors in the file ===")
    cats = {}
    total_quant_elem = 0
    total_quant_bytes = 0
    total_flops_unit = 0  # sum of in*out over quantized 2-D (GEMM FLOP scale factor)
    for t in tensors:
        if not is_2d(t):
            continue
        qt = t.tensor_type
        if qt in NON_QUANT:
            continue
        in_f, out_f = int(t.shape[0]), int(t.shape[1])
        c = cats.setdefault(qname(qt), {"count": 0, "elem": 0, "bytes": 0, "flops_unit": 0})
        c["count"] += 1
        c["elem"] += t.n_elements
        c["bytes"] += t.n_bytes
        c["flops_unit"] += in_f * out_f
        total_quant_elem += t.n_elements
        total_quant_bytes += t.n_bytes
        total_flops_unit += in_f * out_f
    print(f"  {'type':8s} {'count':>6s} {'elements':>16s} {'bytes(GB)':>10s}")
    for name, c in sorted(cats.items(), key=lambda kv: -kv[1]["elem"]):
        print(f"  {name:8s} {c['count']:6d} {c['elem']:16d} {c['bytes']/1e9:10.2f}")
    print(f"  TOTAL quantized 2-D tensors: {sum(c['count'] for c in cats.values())}  "
          f"elements={total_quant_elem}  bytes={total_quant_bytes/1e9:.2f} GB  "
          f"sum(in*out)={total_flops_unit}")

    # --- per-denoising-step estimate (scale the largest tensor's rates) -----
    largest = max(results, key=lambda r: r["n_elem"])
    deq_rate = largest["deq_ms"] / largest["n_elem"]      # ms per element
    copy_rate = largest["copy_ms"] / largest["n_bytes"]   # ms per byte
    print("\n=== per-denoising-step estimate (one forward through ALL quantized 2-D tensors) ===")
    print(f"  scaling rates from largest tensor: {largest['name']} [{largest['type']}] "
          f"({largest['out']}x{largest['in']})")
    print(f"  dequant rate = {deq_rate*1e6:.2f} us/Melem ; transfer rate = {copy_rate*1e9:.2f} ns/byte")
    est_deq = deq_rate * total_quant_elem
    est_copy = copy_rate * total_quant_bytes
    print(f"  totals: elements={total_quant_elem}  bytes={total_quant_bytes/1e9:.2f} GB  "
          f"sum(in*out)={total_flops_unit}")
    print(f"  {'tokens':>8s}  {'dequant ms':>12s}  {'matmul ms':>12s}  {'transfer ms':>12s}")
    for tokens in TOKEN_COUNTS:
        tflops = largest["lin"][tokens][1]
        total_flops = 2 * tokens * total_flops_unit
        est_mm = total_flops / (tflops * 1e12) * 1000.0
        print(f"  {tokens:8d}  {est_deq:12.1f}  {est_mm:12.1f}  {est_copy:12.1f}")
    print("\nDONE linear_bench")


if __name__ == "__main__":
    main()
