#!/usr/bin/env bash
# The host's side of the balance-loss block (Amendment 5). A guard stops the instance (disk kept) CAP seconds after this script starts, whatever it is doing, so that the host cannot bill past the cap if the laptop loses it; a second one stops it when no run has written its log for an hour while training. Then the cards every minute, the locked environment, every shard of the data (so each run's data order is RQ3's), the check (check.py: RQ3's run of all-Muon at seed 3 trained on this code without the loss for its first 250 steps and compared with RQ3's log), one worker per card on the program's balance_loss block, again while a run is neither finished nor diverged, up to three rounds, and last the specialization panel and the annex over the block's runs, one worker per card, as the third chain did, so that each run carries the files its RQ3 run carries. Writes logs/guard.txt, logs/gpus.csv, logs/setup.txt (the panel built there), logs/check.txt, logs/block-gpu<n>.log, logs/eval-gpu<n>.log, logs/phases.txt and logs/DONE. Run detached from ~/code with the tag's code and this folder unpacked there: setsid -f bash ai-experiments/balance_loss_2026_09/host.sh CAP
set -uo pipefail
cap=$1
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
stamp() { echo "$(date -u '+%F %T') $*" >> logs/phases.txt; }
key=${CONTAINER_API_KEY:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_API_KEY=//p')}  # an ssh session lacks the container's variables
id=${CONTAINER_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_ID=//p')}
stop_instance() {
  echo "$(date -u '+%F %T') $1: stopping the instance" >> logs/guard.txt
  [ -n "$key" ] && [ -n "$id" ] && curl -s -X PUT "https://console.vast.ai/api/v0/instances/${id}/" -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" -d '{"state": "stopped"}' >> logs/guard.txt 2>&1
}
echo "$(date -u '+%F %T') guard armed: stop after $cap s (key $([ -n "$key" ] && echo present || echo absent), id ${id:-absent})" > logs/guard.txt
(sleep "$cap"; stop_instance "cap reached") &
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader -l 60 >> logs/gpus.csv &
stamp "setup start"
{
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
  echo "cpus $(nproc)"; free -g | head -2; df -h ~ | tail -1
  t=$(date +%s); uv sync --frozen 2>&1 | tail -1; echo "uv sync $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run hf download marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full --repo-type dataset --local-dir data/fineweb-edu --max-workers 16 > /dev/null 2>&1; echo "hf download $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run python -c "from tfg.data import download; download()"; echo "download() $(( $(date +%s) - t )) s, $(ls data/fineweb-edu | wc -l) files, $(du -sh data/fineweb-edu | cut -f1)"
  for try in 1 2 3; do  # the Hub answered the third chain's first build with a 429
    t=$(date +%s); uv run python -m tfg.panel build 2>&1 | tail -3; code=${PIPESTATUS[0]}; echo "panel, try $try: exit $code in $(( $(date +%s) - t )) s"
    [ "$code" = 0 ] && break; sleep 120
  done
  echo SETUP_OK
} > logs/setup.txt 2>&1
stamp "check start"
check=runs/check/mechanism-muon-s3
rm -rf "$check"
uv run python -m tfg.train --tier mechanism --partition muon --seed 3 --steps 15259 --adamw-lr 2e-3 --muon-lr 4e-3 --compiled --recompute 1 --run-dir "$check" > logs/check-train.txt 2>&1 &
pid=$!
logged() { grep -q '"step": 250,' "$check/metrics.jsonl" 2>/dev/null && grep -q '"step": 250,' "$check/routing.jsonl" 2>/dev/null; }
while kill -0 $pid 2>/dev/null && ! logged; do sleep 5; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null
uv run python ai-experiments/balance_loss_2026_09/check.py "$check" reference/mechanism-muon-s3 > logs/check.txt 2>&1
stamp "check: $(tail -1 logs/check.txt)"
(while sleep 600; do
  newest=$(find runs/balance_loss -name metrics.jsonl -printf '%T@\n' 2>/dev/null | sort -n | tail -1); newest=${newest%.*}
  [ -n "$newest" ] && [ $(( $(date +%s) - newest )) -gt 3600 ] && { stop_instance "no run's log written for an hour"; break; }
done) &
stall=$!
open_runs() { uv run python -c "
from pathlib import Path
from tfg.analysis import status_of
dirs = [s.parent for s in Path('runs/balance_loss').glob('*/settings.json')]
print(max(9, len(dirs)) - sum(status_of(d) in ('finished', 'diverged') for d in dirs))"; }
gpus=$(nvidia-smi -L | wc -l)
for round in 1 2 3; do
  stamp "training round $round on $gpus cards"
  pids=()
  for ((gpu = 0; gpu < gpus; gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu nohup uv run python -m tfg.program balance_loss --recompute 1 --adamw-lr 2e-3 --muon-lr 4e-3 >> logs/block-gpu$gpu.log 2>&1 &
    pids+=($!)
  done
  codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
  left=$(open_runs)
  stamp "training round $round exits:$codes, runs open: $left"
  [ "$left" = "0" ] && break
done
kill $stall 2>/dev/null
stamp "evaluation start"
pids=()
for ((gpu = 0; gpu < gpus; gpu++)); do
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python -m tfg.program evaluate --block balance_loss --batch-size 32 >> logs/eval-gpu$gpu.log 2>&1 &
  pids+=($!)
done
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
stamp "evaluation exits:$codes, $(ls runs/balance_loss/*/specialization.json 2>/dev/null | wc -l) panels and $(ls runs/balance_loss/*/annex.json 2>/dev/null | wc -l) annexes"
touch logs/DONE
