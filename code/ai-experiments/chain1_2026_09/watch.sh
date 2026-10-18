#!/usr/bin/env bash
# The laptop's watch over the first chain: probe.sh over ssh every five minutes, each look appended to outputs/watch.log; exits, which calls the assistant back, with a line saying why: a PROBLEM, a change in the number of live workers, the chain DONE, the host UNREACHABLE three times running, or the HEARTBEAT after the given seconds (two hours by default). Usage: watch.sh <ip> <port> [seconds]
ip=$1; port=$2; beat=${3:-7200}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/outputs"
start=$(date +%s); fails=0; workers=""
while true; do
  out=$(ssh -p "$port" -o ConnectTimeout=20 -o ServerAliveInterval=15 -o StrictHostKeyChecking=no "root@$ip" 'bash ~/code/ai-experiments/chain1_2026_09/probe.sh' 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    fails=$((fails + 1)); echo "$(date '+%F %T') ssh failed ($rc): $out" >> "$here/outputs/watch.log"
    [ $fails -ge 3 ] && { echo "UNREACHABLE: $out"; exit 2; }
    sleep 60; continue
  fi
  fails=0
  { echo "$(date '+%F %T')"; echo "$out"; } >> "$here/outputs/watch.log"
  now=$(echo "$out" | awk '/^WORKERS/{print $2}')
  if echo "$out" | grep -q "^PROBLEM"; then echo "$out"; exit 1; fi
  if echo "$out" | grep -q "^DONE"; then echo "$out"; exit 0; fi
  if [ -n "$workers" ] && [ "$now" != "$workers" ]; then echo "WORKERS $workers -> $now"; echo "$out"; exit 0; fi
  workers=$now
  if [ $(( $(date +%s) - start )) -ge "$beat" ]; then echo "HEARTBEAT"; echo "$out"; exit 0; fi
  sleep 300
done
