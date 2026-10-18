#!/usr/bin/env bash
# The laptop's side of the RUNBOOK's section 1 for the third chain: the code of the tag frozen-2026-09-26 (git archive), this folder's host scripts and the host checks' step.py, and the tuning block's rates.json, which the confirmation and balancer blocks read, into ~/code on the host. Usage: upload.sh <ip> <port>
set -euo pipefail
ip=$1; port=$2
ssh_() { ssh -p "$port" -o ConnectTimeout=20 -o StrictHostKeyChecking=no "root@$ip" "$@"; }
cd "$(git rev-parse --show-toplevel)"
git archive frozen-2026-09-26:code tfg pyproject.toml uv.lock RUNBOOK.md | ssh_ 'mkdir -p code && tar x -C code'
tar c -C code ai-experiments/chain3_2026_09/setup.sh ai-experiments/chain3_2026_09/checks.sh ai-experiments/chain3_2026_09/chain.sh ai-experiments/chain3_2026_09/probe.sh ai-experiments/chain3_2026_09/status.py ai-experiments/host_checks_2026_09/step.py | ssh_ 'tar x -C code'
ssh_ 'mkdir -p code/runs/tuning' && scp -P "$port" -o StrictHostKeyChecking=no code/runs/tuning/rates.json "root@${ip}:code/runs/tuning/"
ssh_ 'cd code && ls && cat runs/tuning/rates.json && du -sh tfg && grep -c "def balancer" tfg/program.py'
