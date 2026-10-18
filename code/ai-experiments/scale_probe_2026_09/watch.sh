#!/usr/bin/env bash
# The laptop's watch over the scale probe: every five minutes the host's setup, jobs and cards appended to outputs/watch.log; exits with a line saying why: SETUP (setup ended), JOB (a job ended), DONE, UNREACHABLE three times running, or HEARTBEAT after the given seconds. Usage: watch.sh <ip> <port> [seconds]
ip=$1; port=$2; beat=${3:-3600}
here=$(cd "$(dirname "$0")" && pwd)
start=$(date +%s); fails=0; jobs=""; setup=""
while true; do
  out=$(ssh -p "$port" -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$ip" 'cd code; tail -3 logs/setup.txt 2>/dev/null; echo "JOBS $(grep -c exit logs/jobs.txt 2>/dev/null)"; tail -4 logs/jobs.txt 2>/dev/null; tail -2 logs/gpus.csv 2>/dev/null; for f in logs/probe-*.txt; do [ -f "$f" ] && echo "$f $(grep -o "\"step\": [0-9]*" "$f" | tail -1)"; done; grep -il "error\|traceback" logs/probe-*.txt 2>/dev/null | sed "s/^/ERRORLOG /"; [ -f logs/DONE ] && echo DONE; df -h ~ | tail -1' 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    fails=$((fails + 1)); echo "$(date -u '+%F %T') ssh failed ($rc): $out" >> "$here/outputs/watch.log"
    [ $fails -ge 3 ] && { echo "UNREACHABLE: $out"; exit 2; }
    sleep 60; continue
  fi
  fails=0
  { echo "$(date -u '+%F %T')"; echo "$out"; } >> "$here/outputs/watch.log"
  now=$(echo "$out" | awk '/^JOBS/{print $2}'); ok=$(echo "$out" | grep -c SETUP_OK)
  if echo "$out" | grep -q "^DONE"; then echo "DONE"; echo "$out"; exit 0; fi
  if echo "$out" | grep -q "^ERRORLOG"; then echo "ERROR"; echo "$out"; exit 1; fi
  if [ -n "$setup" ] && [ "$ok" != "$setup" ]; then echo "SETUP"; echo "$out"; exit 0; fi
  if [ -n "$jobs" ] && [ "$now" != "$jobs" ]; then echo "JOB"; echo "$out"; exit 0; fi
  jobs=$now; setup=$ok
  if [ $(( $(date +%s) - start )) -ge "$beat" ]; then echo "HEARTBEAT"; echo "$out"; exit 0; fi
  sleep 300
done
