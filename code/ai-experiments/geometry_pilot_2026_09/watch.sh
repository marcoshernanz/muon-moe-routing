#!/usr/bin/env bash
# The laptop's watch over the two pilot hosts: probe.sh over ssh every five minutes, appended to outputs/watch.log; exits on a problem, a new phase on either host (a run started or ended), both hosts done, a host unreachable three times running, the first run of a host passing step 1,000 once (for the early bit-for-bit check), or a two-hour heartbeat. Usage: watch.sh ip:port ip:port. Keeps its state per host in outputs/.watch-<ip>-* (the laptop's bash has no associative arrays).
cd "$(dirname "$0")"
start=$(date +%s)
rm -f outputs/.watch-*-misses
while true; do
  done_count=0
  for hp in "$@"; do
    ip=${hp%:*}; port=${hp#*:}; state=outputs/.watch-$ip
    out=$(ssh -o ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -p "$port" "root@$ip" 'bash ~/code/ai-experiments/geometry_pilot_2026_09/probe.sh' < /dev/null 2>/dev/null)
    if [ -z "$out" ]; then
      misses=$(( $(cat $state-misses 2>/dev/null || echo 0) + 1 )); echo $misses > $state-misses
      echo "$(date -u '+%F %T') $hp unreachable ($misses)" >> outputs/watch.log
      [ "$misses" -ge 3 ] && { echo "EXIT unreachable $hp"; exit 0; }
      continue
    fi
    echo 0 > $state-misses
    { echo "$(date -u '+%F %T') $hp"; echo "$out"; } >> outputs/watch.log
    echo "$out" | grep -q "^PROBLEM" && { echo "EXIT problem $hp"; echo "$out"; exit 0; }
    echo "$out" | grep -q "^DONE" && done_count=$(( done_count + 1 ))
    n=$(echo "$out" | awk '/^phases/{print $2}')
    if [ -f $state-phases ] && [ "$n" != "$(cat $state-phases)" ]; then echo "$n" > $state-phases; echo "EXIT phase $hp"; echo "$out"; exit 0; fi
    echo "$n" > $state-phases
    first=$(echo "$out" | grep -E "^mechanism-(adamw|muon)-s0 step" | awk '{print $3}')
    if [ ! -f $state-early ] && [ -n "$first" ] && [ "$first" -ge 1000 ]; then touch $state-early; echo "EXIT early $hp"; echo "$out"; exit 0; fi
  done
  [ "$done_count" -eq "$#" ] && { echo "EXIT all done"; exit 0; }
  [ $(( $(date +%s) - start )) -ge 7200 ] && { echo "EXIT heartbeat"; exit 0; }
  sleep 300
done
