#!/usr/bin/env bash
# The laptop's side, last: the probe's finished run directories (without saves) into outputs/runs and the host's logs into outputs/host-logs. Usage: fetch.sh <ip> <port>
set -euo pipefail
ip=$1; port=$2
out="$(cd "$(dirname "$0")" && pwd)/outputs"
mkdir -p "$out/runs" "$out/host-logs"
ssh_() { ssh -p "$port" -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$ip" "$@"; }
ssh_ 'cd code/runs/scale-probe && tar c --exclude="*.pt" $(for d in */; do [ -f "$d/probe.json" ] && echo "${d%/}"; done)' | tar x -C "$out/runs"
ssh_ 'cd code/logs && tar c .' | tar x -C "$out/host-logs"
ls "$out/runs"; du -sh "$out/runs"
