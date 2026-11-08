import os, sys, time, threading, json, psutil
from pathlib import Path
WANGP = Path.home() / "Wan2GP"; sys.path.insert(0, str(WANGP)); os.chdir(WANGP)
model = sys.argv[1] if len(sys.argv) > 1 else "ti2v_2_2"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
res_ = sys.argv[3] if len(sys.argv) > 3 else "1280x720"
frames = int(sys.argv[4]) if len(sys.argv) > 4 else 121

# peak sampler: eGPU VRAM from amdgpu sysfs (card1 = 0000:37:00.0), process RSS, system used RAM
VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"
base_vram = int(open(VRAM).read())
peak = {"vram": 0, "rss": 0, "sys_used": 0}
me = psutil.Process()
def sampler():
    while True:
        try:
            peak["vram"] = max(peak["vram"], int(open(VRAM).read()))
            peak["rss"] = max(peak["rss"], me.memory_info().rss)
            vm = psutil.virtual_memory(); peak["sys_used"] = max(peak["sys_used"], vm.total - vm.available)
        except Exception: pass
        time.sleep(0.2)
threading.Thread(target=sampler, daemon=True).start()

from shared.api import init, extract_status_phase_label
t0 = time.time()
session = init(root=WANGP, cli_args=["--attention", "sdpa", "--profile", "4"], console_output=True)
t_init = time.time() - t0
s = session.get_default_settings(model)
s.update({"model_type": model, "prompt": "A golden retriever runs along a sunny beach at sunset, waves rolling in, slow tracking shot, cinematic",
          "resolution": res_, "video_length": frames, "num_inference_steps": steps, "seed": 42})
t_submit = time.time()
job = session.submit_task(s); shown = None
marks = []  # (time since submit, label, step)
for ev in job.events.iter(timeout=0.5):
    if ev.kind != "progress": continue
    p = ev.data
    t = extract_status_phase_label(p.status) or p.raw_phase or p.phase
    txt = f"{t} {p.current_step}/{p.total_steps}" if p.current_step is not None and p.total_steps else t
    if txt != shown:
        shown = txt; now = time.time()
        marks.append((now - t_submit, str(t), p.current_step))
        print(f"[bench {now-t0:6.1f}s] {txt}", flush=True)
r = job.result()
t_end = time.time()
print("success", r.success, [e.message for e in r.errors], r.generated_files)
try:
    import torch; tpeak = torch.cuda.max_memory_reserved() / 2**30
except Exception: tpeak = -1
out = dict(model=model, steps=steps, res=res_, frames=frames, init_s=round(t_init, 1),
           gen_s=round(t_end - t_submit, 1), total_s=round(t_end - t0, 1),
           peak_vram_gib=round(peak["vram"] / 2**30, 2), base_vram_gib=round(base_vram / 2**30, 2),
           torch_max_reserved_gib=round(tpeak, 2), peak_rss_gib=round(peak["rss"] / 2**30, 2),
           peak_sys_used_gib=round(peak["sys_used"] / 2**30, 2), success=r.success, files=r.generated_files, marks=marks)
print("BENCHJSON " + json.dumps(out))
print(f"BENCH model={model} steps={steps} res={res_} frames={frames} total={t_end-t0:.0f}s")
