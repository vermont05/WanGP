# Part C helper: summarize prof_ltx/samples.csv per clip and per phase, and echo the
# REPEATJSON lines from sample_run.log plus the linear_bench.log highlights. No GPU needed.
#   .venv/bin/python prof_ltx/analyze.py
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

PROF = Path(__file__).resolve().parent
CSV_PATH = PROF / "samples.csv"
LOG_PATH = PROF / "sample_run.log"


def collapse(label):
    label = label or ""
    return "Loading" if label.startswith("Loading") else label.split(" 2026")[0]


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    print(f"rows in samples.csv: {len(rows)}")

    # group by (clip, collapsed_phase)
    groups = defaultdict(list)
    for r in rows:
        clip = r["clip"]
        if clip == "0":
            continue  # startup/loading before any clip is attributed to the session, not a clip
        ph = collapse(r["phase"])
        groups[(clip, ph)].append(r)

    def f(r, k):
        try:
            return float(r[k])
        except Exception:
            return 0.0

    print("\n=== per-clip / per-phase summary (from samples.csv) ===")
    for (clip, ph), rs in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        rs = sorted(rs, key=lambda r: f(r, "elapsed_s"))
        dur = f(rs[-1], "elapsed_s") - f(rs[0], "elapsed_s")
        gpu = [f(r, "gpu_busy_pct") for r in rs]
        cpu = [f(r, "proc_cpu_pct") for r in rs]
        rd = [f(r, "disk_read_mbs") for r in rs]
        wr = [f(r, "disk_write_mbs") for r in rs]
        sw = [f(r, "swap_gib") for r in rs]
        ram = [f(r, "sys_ram_gib") for r in rs]
        vram = [f(r, "vram_mib") for r in rs]
        # total disk read GB: each row's MB/s is over ~1 s interval
        total_read_gb = sum(rd) / 1000.0
        total_write_gb = sum(wr) / 1000.0
        mean = lambda x: (sum(x) / len(x)) if x else 0.0
        print(f"\nclip {clip} | {ph}  (n={len(rs)}, dur={dur:.1f}s)")
        print(f"   gpu_busy  mean={mean(gpu):5.1f}%  max={max(gpu) if gpu else 0:5.1f}%   "
              f"vram_mib peak={max(vram) if vram else 0:8.1f}")
        print(f"   cpu       mean={mean(cpu):5.1f}%")
        print(f"   disk_read total={total_read_gb:7.3f} GB  mean={mean(rd):7.2f} MB/s   "
              f"disk_write total={total_write_gb:7.3f} GB  mean={mean(wr):7.2f} MB/s")
        print(f"   swap_gib  max={max(sw) if sw else 0:6.3f}   ram_gib peak={max(ram) if ram else 0:6.3f}")

    # session-level (clip 0 = init/loading) peak
    c0 = [r for r in rows if r["clip"] == "0"]
    if c0:
        ram0 = [f(r, "sys_ram_gib") for r in c0]
        sw0 = [f(r, "swap_gib") for r in c0]
        vram0 = [f(r, "vram_mib") for r in c0]
        print(f"\nsession init/loading (clip 0): n={len(c0)}  ram_gib peak={max(ram0):.3f}  "
              f"swap_gib max={max(sw0):.3f}  vram_mib peak={max(vram0):.1f}")

    # REPEATJSON lines from the log
    print("\n=== REPEATJSON lines (from sample_run.log) ===")
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            if line.startswith("REPEATJSON"):
                print(line)
            elif line.startswith("REPEAT "):
                print(line)

    # linear_bench highlights
    bench = PROF / "linear_bench.log"
    if bench.exists():
        print("\n=== linear_bench.log (key lines) ===")
        for line in bench.read_text().splitlines():
            if any(s in line for s in
                   ["largest 2-D", "(a) ", "(b) ", "(c) ", "TOTAL quantized",
                    "per-denoising-step", "tokens", "dequant rate", "scaling rates",
                    "device:", "type=", "Q4_K", "Q5_K", "Q6_K"]):
                print(line)


if __name__ == "__main__":
    main()
