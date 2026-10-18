#!/usr/bin/env bash
# The RUNBOOK's sections 1 and 2 on a rented one-card host, without the panel, which the pilot does not use: the card, the CPU, memory and disk, the locked environment, and every shard of the data (the Hub's client, sixteen at a time, then download()'s check), each timed. Run from ~/code with the working tree's code and this folder unpacked there; writes logs/setup.txt.
set -euo pipefail
command -v uv > /dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
{
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,power.limit,power.max_limit --format=csv
  echo "cpus $(nproc)"; lscpu | grep -E "Model name"; free -g; df -h ~ | tail -1
  t=$(date +%s); uv sync --frozen 2>&1 | tail -1; echo "uv sync $(( $(date +%s) - t )) s"
  uv run python -c "import torch; print(torch.__version__, torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
  t=$(date +%s); uv run hf download marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full --repo-type dataset --local-dir data/fineweb-edu --max-workers 16 > /dev/null 2>&1; echo "hf download $(( $(date +%s) - t )) s"
  t=$(date +%s); uv run python -c "from tfg.data import download; download()"; echo "download() $(( $(date +%s) - t )) s, $(ls data/fineweb-edu | wc -l) files, $(du -sh data/fineweb-edu | cut -f1)"
} > logs/setup.txt 2>&1
echo SETUP_OK
