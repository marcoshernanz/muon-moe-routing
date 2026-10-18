#!/usr/bin/env bash
# Brings the third chain home, as the RUNBOOK's last section does: the confirmation block's run directories (its re-check included) and the balancer block's, without their saves, into code/runs/confirmation and code/runs/balancer, and the host's logs into outputs/host-logs/. Usage: fetch.sh <ip> <port> [final]; with final, the workers' logs are gzipped at home and logs/fetched is touched on the host, which stands down chain.sh's four-hour safety stop.
set -euo pipefail
ip=$1; port=$2; final=${3:-}
here=$(cd "$(dirname "$0")" && pwd)
code=$(cd "$here/../.." && pwd)
for block in confirmation balancer; do
  mkdir -p "$code/runs/$block"
  ssh -p "$port" -o StrictHostKeyChecking=no "root@${ip}" "test -d code/runs/$block" && rsync -az --exclude '*.pt' --exclude '.*.lock' -e "ssh -p $port -o StrictHostKeyChecking=no" "root@${ip}:code/runs/$block/" "$code/runs/$block/"
done
mkdir -p "$here/outputs/host-logs"
rsync -az --exclude '*.off' -e "ssh -p $port -o StrictHostKeyChecking=no" "root@${ip}:code/logs/" "$here/outputs/host-logs/"
if [ "$final" = final ]; then
  gzip -f "$here"/outputs/host-logs/third-gpu*.log
  ssh -p "$port" -o StrictHostKeyChecking=no "root@${ip}" 'touch ~/code/logs/fetched'
fi
for block in confirmation balancer; do
  echo "$block: $(find "$code/runs/$block" -name settings.json | wc -l | tr -d ' ') runs, $(du -sh "$code/runs/$block" | cut -f1)"
done
echo "host logs: $(ls "$here/outputs/host-logs" | wc -l | tr -d ' ') files"
