#!/usr/bin/env bash
# The host's side of the scale probe: the locked environment, every shard of the data (so that each run's data order is the third chain's), then launch.py, which starts no job later than DEADLINE seconds after this script started; writes logs/setup.txt, logs/jobs.txt, logs/gpus.csv and logs/DONE. A guard stops the instance (disk kept) CAP seconds after this script starts, whatever it is doing, so that the host cannot bill past the probe's cap if the laptop loses it. Run detached from ~/code with the tag's code and this folder unpacked there: setsid -f bash ai-experiments/scale_probe_2026_09/host.sh CAP DEADLINE
set -uo pipefail
cap=$1; deadline=$(( $(date +%s) + $2 ))
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
key=${CONTAINER_API_KEY:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_API_KEY=//p')}  # an ssh session lacks the container's variables
id=${CONTAINER_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_ID=//p')}
echo "$(date -u '+%F %T') guard armed: stop after $cap s (key $([ -n "$key" ] && echo present || echo absent), id ${id:-absent})" > logs/guard.txt
(sleep "$cap"; echo "$(date -u '+%F %T') cap reached: stopping the instance" >> logs/guard.txt; [ -n "$key" ] && [ -n "$id" ] && curl -s -X PUT "https://console.vast.ai/api/v0/instances/${id}/" -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" -d '{"state": "stopped"}' >> logs/guard.txt 2>&1) &
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader -l 60 >> logs/gpus.csv &
{
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
  echo "cpus $(nproc)"; free -g | head -2; df -h ~ | tail -1
  t=$(date +%s); uv sync --frozen 2>&1 | tail -1; echo "uv sync $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run hf download marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full --repo-type dataset --local-dir data/fineweb-edu --max-workers 16 > /dev/null 2>&1; echo "hf download $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run python -c "from tfg.data import download; download()"; echo "download() $(( $(date +%s) - t )) s, $(ls data/fineweb-edu | wc -l) files, $(du -sh data/fineweb-edu | cut -f1)"
  echo SETUP_OK
} > logs/setup.txt 2>&1
uv run python ai-experiments/scale_probe_2026_09/launch.py "$deadline" > logs/launch.txt 2>&1
touch logs/DONE
