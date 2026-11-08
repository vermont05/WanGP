# Validates the iGPU Gemma encoder (LTX_GEMMA_IGPU) against WanGP's own int8 Gemma on the eGPU.
# Phase "ref":  one WanGP session with the normal encoder, a tiny LTX job per prompt; saves the 49 hidden
#               states and token ids that encode_raw produced (prof_ltx/gemma_ref/*.pt).
# Phase "igpu": encodes the same tokens with gemma-layers on the iGPU and compares per layer
#               (cosine similarity on the real tokens, after LTX's own per-token RMS normalisation too).
# usage: ROCR_VISIBLE_DEVICES=1 ROCPROFILER_REGISTER_ENABLED=0 .venv/bin/python prof_ltx/gemma_igpu_check.py ref
#        LTX_GEMMA_IGPU=~/models/gemma-3-12b-qat/gemma-3-12b-it-qat-Q4_0.gguf .venv/bin/python prof_ltx/gemma_igpu_check.py igpu
import json, os, sys, time
from pathlib import Path
import torch
WANGP = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(WANGP)); os.chdir(WANGP)
OUT = WANGP / "prof_ltx" / "gemma_ref"; OUT.mkdir(exist_ok=True)
PROMPTS = [
    "A red fox trots through fresh snow in a pine forest at dawn, soft light, slow dolly shot",
    "Anime style, flat cel shading: a girl with short blue hair sings into a microphone on a small stage, "
    "spotlight, cheering crowd silhouettes, the camera slowly pushes in. She sings: \"Tonight we shine!\"",
    "A neon-lit city street at night in the rain, reflections on wet asphalt, a cyclist passes, cinematic",
]

if sys.argv[1] == "ref":
    os.environ.pop("LTX_GEMMA_IGPU", None)
    from models.ltx2.ltx_core.text_encoders.gemma.encoders import base_encoder as be
    orig = be.GemmaTextEncoderModelBase.encode_raw
    count = {"n": 0}
    def capture(self, text, padding_side="left"):
        t = time.time(); raw = orig(self, text, padding_side=padding_side); torch.cuda.synchronize()
        ids = [int(p[0]) for p in self.tokenizer.tokenize_with_weights(text)["gemma"]]
        mask = raw.attention_mask[0].cpu()
        n = int(mask.sum()); hs = torch.stack([h[0, -n:].float().cpu() for h in raw.hidden_states])
        torch.save({"text": text, "ids": [i for i, m in zip(ids, mask.tolist()) if m], "hs": hs, "seconds": time.time() - t}, OUT / f"ref{count['n']}.pt")
        print(f"REF saved prompt {count['n']}: {n} tokens, {time.time() - t:.1f}s", flush=True); count["n"] += 1
        return raw
    be.GemmaTextEncoderModelBase.encode_raw = capture
    from shared.api import init
    s = init(root=WANGP, cli_args=["--attention", "sdpa", "--profile", "4"], console_output=False)
    for i, p in enumerate(PROMPTS):
        st = s.get_default_settings("ltx2_22B_distilled_gguf_q4_k_m")
        st.update({"model_type": "ltx2_22B_distilled_gguf_q4_k_m", "prompt": p, "resolution": "512x320", "video_length": 17, "num_inference_steps": 8, "seed": 1})
        r = s.submit_task(st).result(); print("job", i, r.success, flush=True)
else:
    from shared.gemma_igpu import GemmaIGPU
    g = GemmaIGPU.get(); rows = []
    for f in sorted(OUT.glob("ref*.pt")):
        ref = torch.load(f); t = time.time(); hs = g.hidden_states(ref["ids"]); dt = time.time() - t
        a, b = ref["hs"], hs
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [L, n]
        rms = lambda x: x * torch.rsqrt((x ** 2).mean(-1, keepdim=True) + 1e-6)
        relerr = ((rms(a) - rms(b)).norm(dim=-1) / rms(a).norm(dim=-1))  # [L, n]
        r = {"file": f.name, "tokens": len(ref["ids"]), "ref_s": round(ref["seconds"], 1), "igpu_s": round(dt, 1),
             "cos_mean": round(float(cos.mean()), 4), "cos_min_layer": round(float(cos.mean(1).min()), 4),
             "worst_layer": int(cos.mean(1).argmin()), "cos_per_layer": [round(float(x), 3) for x in cos.mean(1)],
             "rms_relerr_mean": round(float(relerr.mean()), 4)}
        rows.append(r); print("GEMMACHECK " + json.dumps(r), flush=True)
