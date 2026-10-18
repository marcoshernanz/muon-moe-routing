#!/usr/bin/env bash
# The RUNBOOK's first chain (section 5) with four workers, one per card, and edge.py between the tuning and Gate 1; beside it, the cards' power, use, memory, temperature and clock every 30 s, and the host's memory and disk every minute. Run from ~/code, detached (setsid -f); relaunched as it is after an interruption, since every block resumes. Appends to logs/first-gpu<n>.log, logs/gpus.csv, logs/host.txt and logs/phases.txt, writes the workers' process ids to logs/workers.pid, and touches logs/done when every worker has exited.
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
G="uv run python ai-experiments/chain1_2026_09/edge.py"
stamp "first chain start"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$P tuning --recompute 1 && $G && $P gate1 --recompute 1 && $P rq2 --recompute 1" >> logs/first-gpu$gpu.log 2>&1 &
  pids+=($!)
done
echo "${pids[@]}" > logs/workers.pid
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
stamp "first chain exits:$codes"
kill $gpus $host 2>/dev/null
touch logs/done
