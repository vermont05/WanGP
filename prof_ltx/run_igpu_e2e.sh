#!/bin/bash
# End-to-end LTX-2.3 run with Gemma on the iGPU (same seeds/prompts as ab_native.log), with a RAM watchdog:
# if MemAvailable drops below 1.5 GB the run is stopped (logged as MEMABORT), and peak RAM use is logged every 5 s.
cd ~/Wan2GP
export ROCR_VISIBLE_DEVICES=1 MIOPEN_FIND_MODE=FAST ROCPROFILER_REGISTER_ENABLED=0 PYTHONUNBUFFERED=1
export WGP_GGUF_LLAMACPP_CUDA=1 WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE=cublas
export LTX_GEMMA_IGPU=$HOME/models/gemma-3-12b-qat/gemma-3-12b-it-qat-Q4_0.gguf
unset HSA_OVERRIDE_GFX_VERSION
P="A red fox trots through fresh snow in a pine forest at dawn, soft light, slow dolly shot|An old fisherman mends a net on a wooden pier, gulls overhead, waves lapping, documentary style|A neon-lit city street at night in the rain, reflections on wet asphalt, a cyclist passes, cinematic"
LOG=prof_ltx/ab_igpu_gemma.log; MEM=prof_ltx/ab_igpu_gemma.mem
: > $MEM
BENCH_PROMPTS="$P" .venv/bin/python bench_repeat.py ltx2_22B_distilled_gguf_q4_k_m 8 1280x720 121 42,7,11 > $LOG 2>&1 &
PID=$!; minavail=999999
while kill -0 $PID 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ $avail -lt $minavail ] && minavail=$avail
  echo "$(date +%T) avail_MB=$avail gemma=$(pgrep -x gemma-layers >/dev/null && echo up || echo down)" >> $MEM
  if [ $avail -lt 1500 ]; then echo "MEMABORT avail_MB=$avail" | tee -a $LOG; pkill -x gemma-layers; kill $PID; sleep 5; kill -9 $PID 2>/dev/null; break; fi
  sleep 5
done
wait $PID 2>/dev/null
echo "MIN_AVAIL_MB=$minavail" | tee -a $LOG
