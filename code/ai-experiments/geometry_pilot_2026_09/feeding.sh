#!/usr/bin/env bash
# Host C's last two steps, added once (h)'s first 860 steps showed that the dense block alone leaves the router's inputs as dissimilar as all-AdamW: stops (h) at step 3,000, as (d) was, then trains (i), Muon on attention and the dense block together (every matrix that feeds the first mixture), and stops it at step 3,000 too. Unlike experts.sh, it waits for the training process alone, so the card logger cannot hold it. Run from ~/code, detached (setsid -f), as `feeding.sh`.
set -u
export PATH="$HOME/.local/bin:$PATH"
stamp() { echo "$(date -u '+%F %T') $*" >> logs/phases.txt; }
step_of() { tail -1 runs/pilot/$1/routing.jsonl 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])' 2>/dev/null || echo 0; }
until [ "$(step_of mechanism-muon_dense-s0)" -ge 3000 ]; do sleep 15; done
pkill -f "runs/pilot/mechanism-muon_dense-s0"
stamp "stopped mechanism-muon_dense-s0 at step $(step_of mechanism-muon_dense-s0)"
until [ -f logs/done ]; do sleep 10; done
rm -f logs/done
name=mechanism-muon_attention_dense-s0
nvidia-smi --query-gpu=timestamp,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader -l 30 >> logs/gpu.csv &
logger=$!
stamp "start $name"
uv run python -m tfg.train --tier mechanism --steps 7629 --compiled --recompute 1 --resume --run-dir runs/pilot/$name --partition muon_attention_dense --adamw-lr 0.002 --muon-lr 0.004 >> logs/pilot.log 2>&1 &
train=$!
until [ "$(step_of $name)" -ge 3000 ]; do
  kill -0 $train 2>/dev/null || break
  sleep 15
done
pkill -f "runs/pilot/$name"
wait $train
stamp "end $name stopped at step $(step_of $name)"
kill $logger 2>/dev/null
touch logs/done
