#!/usr/bin/env bash
# The laptop's side, first: the code of the tag frozen-2026-09-26 (git archive; the thesis code the third chain ran) and this folder's probe and host scripts into ~/code on the host. Usage: upload.sh <ip> <port>
set -euo pipefail
ip=$1; port=$2
ssh_() { ssh -p "$port" -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$ip" "$@"; }
cd "$(git rev-parse --show-toplevel)"
git archive frozen-2026-09-26:code tfg pyproject.toml uv.lock | ssh_ 'mkdir -p code && tar x -C code'
tar c -C code ai-experiments/scale_probe_2026_09/probe.py ai-experiments/scale_probe_2026_09/launch.py ai-experiments/scale_probe_2026_09/host.sh | ssh_ 'tar x -C code'
ssh_ 'cd code && ls && ls ai-experiments/scale_probe_2026_09 && du -sh tfg'
