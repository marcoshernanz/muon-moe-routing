#!/usr/bin/env bash
# The laptop's watch over the block: every five minutes the host's phases, the step each run has reached, and the cards, appended to outputs/watch.log; exits with a line saying why: SETUP (setup ended), CHECK (the check ended), RUN (a run finished), DONE, ERROR (a traceback in a worker's log), UNREACHABLE three times running, or HEARTBEAT after the given seconds. Usage: watch.sh <ip> <port> [seconds]
ip=$1; port=$2; beat=${3:-3600}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/outputs"
start=$(date +%s); fails=0; finished=""; setup=""; checked=""
while true; do
  out=$(ssh -p "$port" -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$ip" 'cd code; tail -2 logs/setup.txt 2>/dev/null; tail -3 logs/phases.txt 2>/dev/null; echo "CHECKED $(grep -c "IDENTICAL\|DIFFERENT" logs/check.txt 2>/dev/null)"; echo "FINISHED $(grep -l "\"val_loss\"" runs/balance_loss/*/metrics.jsonl 2>/dev/null | while read f; do tail -1 "$f" | grep -q "\"step\": 15259," && echo x; done | wc -l)"; for d in runs/balance_loss/*/; do [ -f "$d/metrics.jsonl" ] && echo "$(basename "$d") $(tail -1 "$d/metrics.jsonl" | grep -o "\"step\": [0-9]*")"; done; tail -4 logs/gpus.csv 2>/dev/null; grep -l "Traceback" logs/block-gpu*.log 2>/dev/null | sed "s/^/ERRORLOG /"; [ -f logs/DONE ] && echo DONE; df -h ~ | tail -1' 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    fails=$((fails + 1)); echo "$(date -u '+%F %T') ssh failed ($rc): $out" >> "$here/outputs/watch.log"
    [ $fails -ge 3 ] && { echo "UNREACHABLE: $out"; exit 2; }
    sleep 60; continue
  fi
  fails=0
  { echo "$(date -u '+%F %T')"; echo "$out"; } >> "$here/outputs/watch.log"
  ok=$(echo "$out" | grep -c SETUP_OK); now=$(echo "$out" | awk '/^FINISHED/{print $2}'); check=$(echo "$out" | awk '/^CHECKED/{print $2}')
  if echo "$out" | grep -q "^DONE"; then echo "DONE"; echo "$out"; exit 0; fi
  if echo "$out" | grep -q "^ERRORLOG"; then echo "ERROR"; echo "$out"; exit 1; fi
  if [ -n "$setup" ] && [ "$ok" != "$setup" ]; then echo "SETUP"; echo "$out"; exit 0; fi
  if [ -n "$checked" ] && [ "$check" != "$checked" ]; then echo "CHECK"; echo "$out"; exit 0; fi
  if [ -n "$finished" ] && [ "$now" != "$finished" ]; then echo "RUN"; echo "$out"; exit 0; fi
  finished=$now; setup=$ok; checked=$check
  if [ $(( $(date +%s) - start )) -ge "$beat" ]; then echo "HEARTBEAT"; echo "$out"; exit 0; fi
  sleep 300
done
