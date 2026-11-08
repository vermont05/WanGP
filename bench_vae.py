# Standalone Wan2.2 VAE decode benchmark: python bench_vae.py <tile_size> [frames_latent=31] [h=44] [w=80]
import os, sys, time
from pathlib import Path
WANGP = Path(__file__).resolve().parent; sys.path.insert(0, str(WANGP)); os.chdir(WANGP)
import torch
tile = int(sys.argv[1]) if len(sys.argv) > 1 else 256
T, H, W = (int(a) for a in sys.argv[2:5]) if len(sys.argv) > 4 else (31, 44, 80)
from models.wan.modules.vae2_2 import Wan2_2_VAE
vae = Wan2_2_VAE(vae_pth="ckpts/Wan2.2_VAE.safetensors", dtype=torch.float16, device="cuda")
torch.manual_seed(0)
z = torch.randn(48, T, H, W, dtype=torch.float16, device="cuda")
prof = os.environ.get("VAE_PROF") == "1"
if os.environ.get("VAE_DET") == "1": torch.backends.cudnn.deterministic = True
with torch.inference_mode():
    vae.decode_to_cpu_uint8([z[:, :2]], tile)  # warm-up (kernel selection)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    if prof:
        from torch.profiler import profile, ProfilerActivity
        p = profile(activities=[ProfilerActivity.CUDA], record_shapes=False); p.__enter__()
    t = time.time()
    out = vae.decode_to_cpu_uint8([z], tile)[0]
    torch.cuda.synchronize(); dt = time.time() - t
    if prof:
        p.__exit__(None, None, None)
        print(p.key_averages().table(sort_by="device_time_total", row_limit=12, max_name_column_width=70))
print(f"VAEBENCH tile={tile} latent={T}x{H}x{W} out={tuple(out.shape)} decode={dt:.1f}s peak_alloc={torch.cuda.max_memory_allocated()/2**30:.2f}GiB peak_reserved={torch.cuda.max_memory_reserved()/2**30:.2f}GiB")
if os.environ.get("VAE_SAVE"):
    torch.save(out, os.environ["VAE_SAVE"])
