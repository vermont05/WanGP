# Native llama.cpp GGUF HIP kernels (llamacpp_gguf_cuda, built for gfx1102) vs WanGP's fallback
# (GPU dequant to bf16 + F.linear) on real LTX-2.3 Q4_K_M weights.
# For the largest Q4_K, Q5_K and Q6_K tensor and a typical 4096x4096 Q4_K tensor, at the two
# denoising token counts: relative error of native vs fallback, and median time of each.
# usage: ROCR_VISIBLE_DEVICES=1 ROCPROFILER_REGISTER_ENABLED=0 .venv/bin/python prof_ltx/native_gguf_bench.py
import os, sys, time, statistics, json
from pathlib import Path
import numpy as np, torch, gguf
WANGP = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(WANGP)); os.chdir(WANGP)
import shared.qtypes.gguf as wg
from shared.qtypes.gguf import _gguf_dequantize_tensor

GGUF_FILE = str(WANGP / "ckpts" / "ltx-2.3-22b-distilled-Q4_K_M_light.gguf")
TOKENS = [int(x) for x in os.environ.get("TOKENS", "3520,14080").split(",")]
assert wg._probe_gguf_cuda_runtime(force=True), wg._GGUF_CUDA_LOAD_ERROR
mod = wg._GGUF_CUDA_MODULE
print("llamacpp_gguf_cuda", mod.__version__, "linear mode:", mod._linear_mode(), flush=True)


def med_ms(fn, warm=2, reps=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 1)


reader = gguf.GGUFReader(GGUF_FILE)
T = gguf.GGMLQuantizationType
picked = []
for qt in (T.Q4_K, T.Q5_K, T.Q6_K):
    c = sorted((t for t in reader.tensors if len(t.shape) == 2 and t.tensor_type == qt), key=lambda t: -t.n_elements)
    if c: picked.append(c[0])
sq = [t for t in reader.tensors if len(t.shape) == 2 and t.tensor_type == T.Q4_K and int(t.shape[0]) == int(t.shape[1]) == 4096]
if sq: picked.append(sq[0])

rows = []
for t in picked:
    inf, outf = int(t.shape[0]), int(t.shape[1]); shape = (outf, inf); qn = t.tensor_type.name
    raw = torch.from_numpy(np.ascontiguousarray(t.data, dtype=np.uint8)).cuda()
    raw_flat = raw.reshape(-1).contiguous()
    w = _gguf_dequantize_tensor(raw, t.tensor_type, shape, dtype=torch.bfloat16)
    for n in TOKENS:
        x = torch.randn(n, inf, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        nat = wg._llamacpp_cuda_linear(mod, raw_flat, qn, shape, x, None, torch.bfloat16)
        rel = float((nat.float() - ref.float()).norm() / ref.float().norm())
        t_fb = med_ms(lambda: torch.nn.functional.linear(x, _gguf_dequantize_tensor(raw, t.tensor_type, shape, dtype=torch.bfloat16)))
        t_mm = med_ms(lambda: torch.nn.functional.linear(x, w))
        t_nat = med_ms(lambda: wg._llamacpp_cuda_linear(mod, raw_flat, qn, shape, x, None, torch.bfloat16))
        r = dict(tensor=t.name, qtype=qn, shape=list(shape), tokens=n, rel_err=round(rel, 5),
                 fallback_ms=t_fb, bf16_matmul_only_ms=t_mm, native_ms=t_nat, speedup=round(t_fb / t_nat, 2))
        rows.append(r); print("NATIVEJSON " + json.dumps(r), flush=True)
        del x, ref, nat
    del raw, raw_flat, w; torch.cuda.empty_cache()
