# Part A: One WanGP session generating TWO LTX-2.3 clips back to back, with a 1 Hz
# background sampler recording system metrics (eGPU busy/VRAM, process CPU, RAM/swap,
# disk r/w, major page faults) to samples.csv. Prints REPEATJSON per-phase durations
# exactly like ../bench_repeat.py.
#
# Code is COPIED from ../bench_repeat.py (not imported). Run with the GPU env:
#   ROCR_VISIBLE_DEVICES=1 MIOPEN_FIND_MODE=FAST PYTHONUNBUFFERED=1 .venv/bin/python prof_ltx/sample_run.py
import csv
import json
import os
import sys
import time
import threading
import psutil
from pathlib import Path

WANGP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WANGP))
os.chdir(WANGP)

MODEL = "ltx2_22B_distilled_gguf_q4_k_m"
STEPS = 8
RES = "1280x720"
FRAMES = 121
# (prompt, seed) for the two clips. Clip 2 uses a NEW prompt so the text encoder
# must run again with the models already loaded.
CLIPS = [
    ("A golden retriever runs along a sunny beach at sunset, waves rolling in, slow tracking shot, cinematic", 42),
    ("A red fox walks through fresh snow in a pine forest at dawn, soft light, slow dolly shot", 7),
]

PROF = Path(__file__).resolve().parent
CSV_PATH = PROF / "samples.csv"

# ---- system metric readers -------------------------------------------------
GPU_BUSY = "/sys/class/drm/card1/device/gpu_busy_percent"
GPU_VRAM = "/sys/class/drm/card1/device/mem_info_vram_used"  # bytes
DISKSTATS = "/proc/diskstats"
SELF_STAT = "/proc/self/stat"
DISK = "nvme0n1"
SECTOR = 512  # bytes per sector


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return -1


def _disk_sectors():
    try:
        with open(DISKSTATS) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 8 and parts[2] == DISK:
                    return int(parts[4]), int(parts[7])  # sectors read, sectors written
    except Exception:
        pass
    return None, None


def _majflt():
    try:
        with open(SELF_STAT) as f:
            data = f.read()
        rest = data[data.rfind(")") + 2:]
        return int(rest.split()[12 - 3])  # field 12 = majflt
    except Exception:
        return None


def _meminfo_kb(key):
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


CSV_COLS = [
    "elapsed_s", "clip", "phase",
    "gpu_busy_pct", "vram_mib", "proc_cpu_pct",
    "sys_ram_gib", "swap_gib",
    "disk_read_mbs", "disk_write_mbs", "majflt_per_s",
]

state = {"clip": 0, "phase": "startup", "done": False}
proc = psutil.Process()
proc.cpu_percent(None)  # prime so the first interval is meaningful


def sampler():
    csv_out = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_out)
    writer.writerow(CSV_COLS)
    csv_out.flush()
    t0 = _START
    prev_t = time.time()
    prev_rd, prev_wr = _disk_sectors()
    prev_maj = _majflt()
    while not state["done"]:
        time.sleep(1.0)
        now = time.time()
        dt = max(now - prev_t, 1e-6)

        gpu_busy = _read_int(GPU_BUSY)
        vram_mib = (_read_int(GPU_VRAM) / (1024 * 1024)) if _read_int(GPU_VRAM) >= 0 else -1
        proc_cpu = proc.cpu_percent(None)
        ram_gib = (_meminfo_kb("MemTotal:") - _meminfo_kb("MemAvailable:")) / (1024 * 1024)
        swap_gib = (_meminfo_kb("SwapTotal:") - _meminfo_kb("SwapFree:")) / (1024 * 1024)

        cur_rd, cur_wr = _disk_sectors()
        if prev_rd is not None and cur_rd is not None:
            read_mbs = (cur_rd - prev_rd) * SECTOR / (1024 * 1024) / dt
            write_mbs = (cur_wr - prev_wr) * SECTOR / (1024 * 1024) / dt
        else:
            read_mbs = write_mbs = 0.0
        prev_rd, prev_wr = cur_rd, cur_wr

        cur_maj = _majflt()
        majflt_ps = (cur_maj - prev_maj) / dt if (prev_maj is not None and cur_maj is not None) else 0.0
        prev_maj = cur_maj

        writer.writerow([
            round(now - t0, 1), state["clip"], state["phase"],
            gpu_busy, round(vram_mib, 1), round(proc_cpu, 1),
            round(ram_gib, 3), round(swap_gib, 3),
            round(read_mbs, 2), round(write_mbs, 2), round(majflt_ps, 1),
        ])
        csv_out.flush()
        prev_t = now
    state["done"] = True
    csv_out.close()


_START = time.time()
threading.Thread(target=sampler, daemon=True).start()

# ---- WanGP session (copied from bench_repeat.py) ---------------------------
from shared.api import init, extract_status_phase_label
t0 = time.time()
session = init(root=WANGP, cli_args=["--attention", "sdpa", "--profile", "4"], console_output=True)
state["phase"] = "init"
print(f"REPEAT init {time.time()-t0:.1f}s", flush=True)

for n, (prompt, seed) in enumerate(CLIPS, 1):
    state["clip"] = n
    s = session.get_default_settings(MODEL)
    s.update({
        "model_type": MODEL, "prompt": prompt, "resolution": RES,
        "video_length": FRAMES, "num_inference_steps": STEPS, "seed": seed,
    })
    ts = time.time()
    state["phase"] = "submit"
    job = session.submit_task(s)
    phases = {}
    order = []
    for ev in job.events.iter(timeout=0.5):
        if ev.kind != "progress":
            continue
        p = ev.data
        label = extract_status_phase_label(p.status) or p.raw_phase or p.phase or ""
        state["phase"] = label
        key = "Loading" if label.startswith("Loading") else label.split(" 2026")[0]
        if key not in phases:
            phases[key] = time.time() - ts
            order.append(key)
            print(f"[run {n} {time.time()-ts:6.1f}s] {key}", flush=True)
    r = job.result()
    total = time.time() - ts
    seq = [(k, phases[k]) for k in order] + [("end", total)]
    durations = {seq[i][0]: round(seq[i + 1][1] - seq[i][1], 1) for i in range(len(seq) - 1)}
    print("REPEATJSON " + json.dumps({
        "run": n, "seed": seed, "success": r.success, "total_s": round(total, 1),
        "phase_seconds": durations, "files": r.generated_files,
    }), flush=True)

state["clip"] = 0
state["phase"] = "done"
state["done"] = True
print(f"REPEAT total session {time.time()-_START:.1f}s", flush=True)
print(f"SAMPLES {CSV_PATH}", flush=True)
time.sleep(2)  # let the sampler flush its final state
