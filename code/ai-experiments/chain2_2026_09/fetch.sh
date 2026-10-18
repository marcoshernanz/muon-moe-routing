#!/usr/bin/env bash
# Brings the second chain home, as the RUNBOOK's last section does: the RQ3 block's run directories without their saves into code/runs/rq3, and the host's logs into outputs/host-logs/. Usage: fetch.sh <ip> <port> [final]; with final, the workers' logs are gzipped at home and logs/fetched is touched on the host, which stands down chain.sh's four-hour safety stop.
set -euo pipefail
ip=$1; port=$2; final=${3:-}
here=$(cd "$(dirname "$0")" && pwd)
code=$(cd "$here/../.." && pwd)
mkdir -p "$code/runs/rq3" "$here/outputs/host-logs"
rsync -az --exclude '*.pt' --exclude '.*.lock' -e "ssh -p $port -o StrictHostKeyChecking=no" "root@${ip}:code/runs/rq3/" "$code/runs/rq3/"
rsync -az --exclude '*.off' -e "ssh -p $port -o StrictHostKeyChecking=no" "root@${ip}:code/logs/" "$here/outputs/host-logs/"
if [ "$final" = final ]; then
  gzip -f "$here"/outputs/host-logs/second-gpu*.log
  ssh -p "$port" -o StrictHostKeyChecking=no "root@${ip}" 'touch ~/code/logs/fetched'
fi
echo "runs: $(ls -d "$code"/runs/rq3/*/ 2>/dev/null | wc -l | tr -d ' ') directories, $(du -sh "$code/runs/rq3" | cut -f1); host logs: $(ls "$here/outputs/host-logs" | wc -l | tr -d ' ') files"
