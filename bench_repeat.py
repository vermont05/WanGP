# One WanGP session, several generations back to back: shows cold (first) vs warm (later) timings per phase.
# usage: python bench_repeat.py <model_type> <steps> <WxH> <frames> <seed1,seed2,...>
import json, os, sys, time, threading, psutil
from pathlib import Path
WANGP = Path(__file__).resolve().parent; sys.path.insert(0, str(WANGP)); os.chdir(WANGP)
model, steps, res_, frames = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
seeds = [int(x) for x in sys.argv[5].split(",")]
PROMPT = "A golden retriever runs along a sunny beach at sunset, waves rolling in, slow tracking shot, cinematic"
# BENCH_PROMPTS="p1|p2|...": a different prompt per generation (otherwise the same prompt, whose text encoding is cached)
PROMPTS = [x.strip() for x in os.environ.get("BENCH_PROMPTS", "").split("|") if x.strip()]

peak = {"sys": 0}
def sampler():
    while True:
        vm = psutil.virtual_memory(); peak["sys"] = max(peak["sys"], vm.total - vm.available); time.sleep(0.5)
threading.Thread(target=sampler, daemon=True).start()

from shared.api import init, extract_status_phase_label
t0 = time.time()
session = init(root=WANGP, cli_args=["--attention", "sdpa", "--profile", "4"], console_output=True)
print(f"REPEAT init {time.time()-t0:.1f}s", flush=True)
for n, seed in enumerate(seeds, 1):
    s = session.get_default_settings(model)
    s.update({"model_type": model, "prompt": PROMPTS[(n - 1) % len(PROMPTS)] if PROMPTS else PROMPT, "resolution": res_, "video_length": frames, "num_inference_steps": steps, "seed": seed})
    peak["sys"] = 0
    ts = time.time(); job = session.submit_task(s); phases = {}; order = []
    for ev in job.events.iter(timeout=0.5):
        if ev.kind != "progress": continue
        p = ev.data
        label = extract_status_phase_label(p.status) or p.raw_phase or p.phase or ""
        key = "Loading" if label.startswith("Loading") else label.split(" 2026")[0]
        if key not in phases: phases[key] = time.time() - ts; order.append(key); print(f"[run {n} {time.time()-ts:6.1f}s] {key}", flush=True)
    r = job.result(); total = time.time() - ts
    seq = [(k, phases[k]) for k in order] + [("end", total)]
    durations = {seq[i][0]: round(seq[i + 1][1] - seq[i][1], 1) for i in range(len(seq) - 1)}
    print("REPEATJSON " + json.dumps({"run": n, "seed": seed, "success": r.success, "total_s": round(total, 1),
                                      "peak_sys_used_gib": round(peak["sys"] / 2**30, 2), "phase_seconds": durations,
                                      "files": r.generated_files}), flush=True)
