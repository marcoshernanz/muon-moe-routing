#!/usr/bin/env bash
# The pilot's run of (d), Muon on the routed experts alone, moved from host A's list to host B's and stopped at step 3,000 to keep the pilot inside its budget: its role is the first mixture's readings, which separate the arms by then. Waits for pilot.sh's list to end, trains (d) with the pilot's settings (so its schedule is the full run's), stops it once its routing log passes step 3,000, and touches logs/done again. Run from ~/code, detached (setsid -f), as `experts.sh`.
set -u
export PATH="$HOME/.local/bin:$PATH"
until [ -f logs/done ]; do sleep 10; done
rm -f logs/done
name=mechanism-muon_experts-s0
stamp() { echo "$(date -u '+%F %T') $*" >> logs/phases.txt; }
nvidia-smi --query-gpu=timestamp,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader -l 30 >> logs/gpu.csv &
watch=$!
stamp "start $name"
uv run python -m tfg.train --tier mechanism --steps 7629 --compiled --recompute 1 --resume --run-dir runs/pilot/$name --partition muon_experts --adamw-lr 0.002 --muon-lr 0.004 >> logs/pilot.log 2>&1 &
until [ "$(tail -1 runs/pilot/$name/routing.jsonl 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0)" -ge 3000 ]; do
  kill -0 $! 2>/dev/null || break
  sleep 15
done
pkill -f "runs/pilot/$name"; wait
stamp "end $name stopped at step $(tail -1 runs/pilot/$name/routing.jsonl | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])')"
kill $watch 2>/dev/null
touch logs/done
