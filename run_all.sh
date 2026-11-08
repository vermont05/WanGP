#!/usr/bin/env bash
cd ~/Wan2GP
export ROCR_VISIBLE_DEVICES=1 MIOPEN_FIND_MODE=FAST PYTHONUNBUFFERED=1
run() { tag=$1; shift; echo "=== START $tag $(date +%T)"; .venv/bin/python bench_wan.py "$@" > benchlogs/$tag.log 2>&1; rc=$?; ok=$(grep -c '^success True' benchlogs/$tag.log); echo "=== END $tag rc=$rc ok=$ok $(date +%T)"; [ "$ok" = 1 ]; }
if run A ti2v_2_2 20 1280x720 121; then AR="1280x720 121"
elif run A_61f ti2v_2_2 20 1280x720 61; then AR="1280x720 61"
elif run A_480p ti2v_2_2 20 832x480 121; then AR="832x480 121"
else AR=""; fi
echo "=== A config: $AR"
[ -n "$AR" ] && run B ti2v_2_2 25 $AR
[ -n "$AR" ] && run FastWan ti2v_2_2_fastwan 3 $AR
echo "=== ALL DONE"
