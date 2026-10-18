#!/usr/bin/env bash
# The laptop's side, first: the code of the tag frozen-2026-09-28 (git archive; the thesis code the block runs), this folder's host scripts, and the check's reference, RQ3's run of all-Muon at seed 3 to step 250 (its settings and its first 25 lines of each log), into ~/code on the host. Usage: upload.sh <ip> <port>
set -euo pipefail
ip=$1; port=$2
ssh_() { ssh -p "$port" -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$ip" "$@"; }
cd "$(git rev-parse --show-toplevel)"
git archive frozen-2026-09-28:code tfg pyproject.toml uv.lock | ssh_ 'mkdir -p code && tar x -C code'
tar c -C code ai-experiments/balance_loss_2026_09/host.sh ai-experiments/balance_loss_2026_09/check.py | ssh_ 'tar x -C code'
reference=$(mktemp -d); trap 'rm -rf "$reference"' EXIT
mkdir -p "$reference/reference/mechanism-muon-s3"
rq3=code/runs/rq3/mechanism-muon-s3
cp "$rq3/settings.json" "$reference/reference/mechanism-muon-s3/"
for name in metrics.jsonl routing.jsonl; do head -25 "$rq3/$name" > "$reference/reference/mechanism-muon-s3/$name"; done
tar c -C "$reference" reference | ssh_ 'tar x -C code'
ssh_ 'cd code && ls && ls ai-experiments/balance_loss_2026_09 reference/mechanism-muon-s3 && grep -c "def sequence_balance" tfg/model.py && du -sh tfg'
