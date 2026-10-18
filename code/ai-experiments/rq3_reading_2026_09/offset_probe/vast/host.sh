#!/usr/bin/env bash
# The host's side of the offset probe at the mechanism tier: the locked environment, every shard of the data (so that each run's data order is the second chain's), then launch.py; writes logs/setup.txt, logs/jobs.txt and logs/DONE. Run detached from ~/code with the tag's code and this folder unpacked there: setsid -f bash ai-experiments/rq3_reading_2026_09/offset_probe/vast/host.sh
set -uo pipefail
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
{
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
  t=$(date +%s); uv sync --frozen 2>&1 | tail -1; echo "uv sync $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run hf download marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full --repo-type dataset --local-dir data/fineweb-edu --max-workers 16 > /dev/null 2>&1; echo "hf download $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run python -c "from tfg.data import download; download()"; echo "download() $(( $(date +%s) - t )) s, $(ls data/fineweb-edu | wc -l) files"
  echo SETUP_OK
} > logs/setup.txt 2>&1
uv run python ai-experiments/rq3_reading_2026_09/offset_probe/vast/launch.py > logs/launch.txt 2>&1
touch logs/DONE
