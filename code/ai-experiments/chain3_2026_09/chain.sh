#!/usr/bin/env bash
# The RUNBOOK's third chain (section 5): on each of the four cards one worker running the confirmation block and then the balancer block; when every worker has exited and a run of either block is neither finished nor diverged (a worker that died leaves its run), the same loop again, which resumes it, up to three rounds; then both blocks' evaluation with four workers. Beside it, the cards' power, use, memory, temperature and clock every 30 s, and the host's memory and disk every minute. Run from ~/code, detached (setsid -f); relaunched as it is after an interruption, since the blocks resume and the evaluation skips what it has done. Appends to logs/third-gpu<n>.log, logs/eval-gpu<n>.log, logs/gpus.csv, logs/host.txt and logs/phases.txt, writes the live workers' process ids to logs/workers.pid and the stage to logs/phase, and touches logs/done at the end. If the instance's own Vast key is in the container's environment, two guards stop the instance (disk kept) so that a host nobody watches does not bill its cards: four hours after the end unless logs/fetched exists by then, and while training, when no run has written its log for three hours.
set -u
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
rm -f logs/done
stamp() { echo "$(date -u '+%F %T') $*" >> logs/phases.txt; }
key=${CONTAINER_API_KEY:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_API_KEY=//p')}  # an ssh session lacks the container's variables
id=${CONTAINER_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_ID=//p')}
stop_instance() {
  stamp "$1: stopping the instance"
  [ -n "$key" ] && [ -n "$id" ] && curl -s -X PUT "https://console.vast.ai/api/v0/instances/${id}/" -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" -d '{"state": "stopped"}' >> logs/phases.txt 2>&1
}
nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader -l 30 >> logs/gpus.csv &
gpus=$!
(while true; do echo "$(date -u '+%F %T') used_mb=$(free -m | awk '/Mem:/{print $3}') disk_used=$(df -BG ~ | awk 'NR==2{print $3}') disk_avail=$(df -BG ~ | awk 'NR==2{print $4}')" >> logs/host.txt; sleep 60; done) &
host=$!
(while sleep 600; do
  [ "$(cat logs/phase 2>/dev/null)" = training ] || continue
  newest=$(find runs/confirmation runs/balancer -name metrics.jsonl -printf '%T@\n' 2>/dev/null | sort -n | tail -1); newest=${newest%.*}
  [ -n "$newest" ] && [ $(( $(date +%s) - newest )) -gt 10800 ] && { stop_instance "no run's log written for three hours"; break; }
done) &
stall=$!
P="uv run python -m tfg.program"
open_runs() { uv run python -c "
from pathlib import Path
from tfg.analysis import status_of
def left(root, expected, nested=False):
    root = Path(root)
    dirs = [s.parent for s in (root.rglob('settings.json') if nested else root.glob('*/settings.json'))] if root.exists() else []
    return max(expected, len(dirs)) - sum(status_of(d) in ('finished', 'diverged') for d in dirs)
print(left('runs/confirmation/recheck', 8, nested=True) + left('runs/confirmation', 9) + left('runs/balancer', 12))"; }
for round in 1 2 3; do
  echo training > logs/phase
  stamp "third chain, training round $round"
  pids=()
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$P confirmation --recompute 9 && $P balancer --recompute 1" >> logs/third-gpu$gpu.log 2>&1 &
    pids+=($!)
  done
  echo "${pids[@]}" > logs/workers.pid
  codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
  left=$(open_runs)
  stamp "training round $round exits:$codes, runs open: $left"
  [ "$left" = "0" ] && break
done
kill $stall 2>/dev/null
echo evaluating > logs/phase
stamp "evaluation start"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$P evaluate --block confirmation --batch-size 16 && $P evaluate --block balancer --batch-size 32" >> logs/eval-gpu$gpu.log 2>&1 &
  pids+=($!)
done
echo "${pids[@]}" > logs/workers.pid
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
stamp "evaluation exits:$codes, confirmation $(ls runs/confirmation/*/specialization.json 2>/dev/null | wc -l) panels and $(ls runs/confirmation/*/annex.json 2>/dev/null | wc -l) annexes, balancer $(ls runs/balancer/*/specialization.json 2>/dev/null | wc -l) panels and $(ls runs/balancer/*/annex.json 2>/dev/null | wc -l) annexes"
kill $gpus $host 2>/dev/null
echo done > logs/phase
touch logs/done
sleep 14400
[ -f logs/fetched ] || stop_instance "nobody fetched the results in four hours"
