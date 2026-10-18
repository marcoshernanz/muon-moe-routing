#!/usr/bin/env bash
# Brings the block home, as the RUNBOOK's last section does: its run directories without their saves into code/runs/balance_loss, and the host's logs and the check's run into outputs/. Usage: fetch.sh <ip> <port>
set -euo pipefail
ip=$1; port=$2
here=$(cd "$(dirname "$0")" && pwd)
code=$(cd "$here/../.." && pwd)
mkdir -p "$code/runs/balance_loss" "$here/outputs/host-logs" "$here/outputs/check"
rsync -az --exclude '*.pt' --exclude '.*.lock' -e "ssh -p $port -o StrictHostKeyChecking=no" "root@${ip}:code/runs/balance_loss/" "$code/runs/balance_loss/"
rsync -az -e "ssh -p $port -o StrictHostKeyChecking=no" "root@${ip}:code/logs/" "$here/outputs/host-logs/"
rsync -az --exclude '*.pt' -e "ssh -p $port -o StrictHostKeyChecking=no" "root@${ip}:code/runs/check/" "$here/outputs/check/"
echo "balance_loss: $(find "$code/runs/balance_loss" -name settings.json | wc -l | tr -d ' ') runs, $(du -sh "$code/runs/balance_loss" | cut -f1)"
echo "host logs: $(ls "$here/outputs/host-logs" | wc -l | tr -d ' ') files"
