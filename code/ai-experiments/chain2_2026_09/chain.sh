#!/usr/bin/env bash
# The RUNBOOK's second chain (section 5): the RQ3 block with four workers, one per card; when every worker has exited and a run of the block is neither finished nor diverged (a worker that died leaves its run), the same loop again, which resumes it, up to three rounds; then the block's evaluation with four workers. Beside it, the cards' power, use, memory, temperature and clock every 30 s, and the host's memory and disk every minute. Run from ~/code, detached (setsid -f); relaunched as it is after an interruption, since the block resumes and the evaluation skips what it has done. Appends to logs/second-gpu<n>.log, logs/eval-gpu<n>.log, logs/gpus.csv, logs/host.txt and logs/phases.txt, writes the live workers' process ids to logs/workers.pid and the stage to logs/phase, and touches logs/done at the end. If the instance's own Vast key is in the container's environment, it stops the instance (disk kept) four hours after the end unless logs/fetched exists by then, so that a host nobody fetches does not bill its cards.
set -u
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
rm -f logs/done
stamp() { echo "$(date -u '+%F %T') $*" >> logs/phases.txt; }
nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader -l 30 >> logs/gpus.csv &
gpus=$!
(while true; do echo "$(date -u '+%F %T') used_mb=$(free -m | awk '/Mem:/{print $3}') disk_used=$(df -BG ~ | awk 'NR==2{print $3}') disk_avail=$(df -BG ~ | awk 'NR==2{print $4}')" >> logs/host.txt; sleep 60; done) &
host=$!
P="uv run python -m tfg.program"
open_runs() { uv run python -c "
from pathlib import Path
from tfg.analysis import status_of
runs = [d for d in Path('runs/rq3').iterdir() if (d / 'settings.json').exists()] if Path('runs/rq3').exists() else []
print(33 - sum(status_of(d) in ('finished', 'diverged') for d in runs))"; }
for round in 1 2 3; do
  echo training > logs/phase
  stamp "second chain, training round $round"
  pids=()
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$P rq3 --recompute 1" >> logs/second-gpu$gpu.log 2>&1 &
    pids+=($!)
  done
  echo "${pids[@]}" > logs/workers.pid
  codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
  left=$(open_runs)
  stamp "training round $round exits:$codes, runs open: $left"
  [ "$left" = "0" ] && break
done
echo evaluating > logs/phase
stamp "evaluation start"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$P evaluate --block rq3 --batch-size 32" >> logs/eval-gpu$gpu.log 2>&1 &
  pids+=($!)
done
echo "${pids[@]}" > logs/workers.pid
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
stamp "evaluation exits:$codes, $(ls runs/rq3/*/specialization.json 2>/dev/null | wc -l) panels, $(ls runs/rq3/*/annex.json 2>/dev/null | wc -l) annexes"
kill $gpus $host 2>/dev/null
echo done > logs/phase
touch logs/done
key=${CONTAINER_API_KEY:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_API_KEY=//p')}  # an ssh session lacks the container's variables
id=${CONTAINER_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_ID=//p')}
if [ -n "$key" ] && [ -n "$id" ]; then
  sleep 14400
  if [ ! -f logs/fetched ]; then
    stamp "nobody fetched the results in four hours: stopping the instance"
    curl -s -X PUT "https://console.vast.ai/api/v0/instances/${id}/" -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" -d '{"state": "stopped"}' >> logs/phases.txt 2>&1
  fi
fi
