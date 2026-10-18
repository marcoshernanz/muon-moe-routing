#!/usr/bin/env bash
# The laptop's side: the probe's finished run directories (without saves) into ../outputs/mechanism/runs and the host's logs into ../outputs/mechanism/host-logs; with "final", marks the host fetched so its safety stop stands down. Usage: fetch.sh <ip> <port> [final]
set -euo pipefail
ip=$1; port=$2
out="$(dirname "$0")/../outputs/mechanism"
mkdir -p "$out/runs" "$out/host-logs"
ssh_() { ssh -p "$port" -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$ip" "$@"; }
ssh_ 'cd code/runs/offset-probe && tar c --exclude="*.pt" $(for d in */; do [ -f "$d/probe.json" ] && echo "${d%/}"; done)' | tar x -C "$out/runs"
ssh_ 'cd code/logs && tar c --exclude=safety.sh .' | tar x -C "$out/host-logs"
[ "${3:-}" = final ] && ssh_ 'touch code/logs/fetched'
ls "$out/runs"; du -sh "$out/runs"
