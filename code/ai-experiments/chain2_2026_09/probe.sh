#!/usr/bin/env bash
# One look at the second chain for the laptop's watch.sh. PROBLEM lines: the workers' new log lines that name a problem since the last look, a run that newly diverged, a disk under 25 GB free, a card whose power limit is under the host's own (logs/power.floor, written at the start; 575 W without it) or whose temperature is over 88 C, and, while training, no run's log written for 20 minutes with workers alive. Then the stage (PHASE), the live workers (WORKERS), the block's runs by status (RUNS), the runs under way (status.py), the disk, the cards, and DONE once chain.sh has touched logs/done. Run from anywhere on the host.
cd ~/code
export PATH="$HOME/.local/bin:$PATH"
for f in logs/second-gpu*.log logs/eval-gpu*.log; do
  [ -f "$f" ] || continue
  o=$(cat "$f.off" 2>/dev/null || echo 0); n=$(stat -c %s "$f")
  tail -c +$((o + 1)) "$f" | head -c $((n - o)) | grep -E "Traceback|Error|error:|refused|Killed|out of memory|relaunch|No space" | cut -c1-300 | sed "s|^|PROBLEM $f: |"
  echo "$n" > "$f.off"
done
phase=$(cat logs/phase 2>/dev/null || echo none)
echo "PHASE $phase"
alive=0; for pid in $(cat logs/workers.pid 2>/dev/null); do kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); done
echo "WORKERS $alive"
status=$(PYTHONPATH=. uv run python ai-experiments/chain2_2026_09/status.py 2>&1)
echo "$status" | sed 's/^/  /'
diverged=$(echo "$status" | grep -oE "'diverged': [0-9]+" | grep -oE "[0-9]+" || echo 0); diverged=${diverged:-0}
before=$(cat logs/diverged.count 2>/dev/null || echo 0)
[ "$diverged" -gt "$before" ] && echo "PROBLEM $((diverged - before)) run(s) newly diverged"
echo "$diverged" > logs/diverged.count
if [ "$phase" = training ] && [ "$alive" -gt 0 ]; then
  newest=$(find runs/rq3 -name metrics.jsonl -printf '%T@\n' 2>/dev/null | sort -n | tail -1); newest=${newest%.*}
  [ -n "$newest" ] && [ $(( $(date +%s) - newest )) -gt 1200 ] && echo "PROBLEM no run's log written for $(( ($(date +%s) - newest) / 60 )) minutes"
fi
avail=$(df -BG ~ | awk 'NR==2{gsub("G","",$4); print $4}')
echo "DISK ${avail}G free"
[ "$avail" -lt 25 ] && echo "PROBLEM disk: ${avail}G free"
floor=$(cat logs/power.floor 2>/dev/null || echo 575)
tail -n 4 logs/gpus.csv 2>/dev/null | awk -F', ' -v floor="$floor" '{split($4,l," "); split($7,t," "); printf "GPU %s %s %s %s %s\n", $2, $3, $5, $7, $4; if (l[1] + 0 < floor) print "PROBLEM power limit " $4 " on GPU " $2; if (t[1] + 0 > 88) print "PROBLEM temperature " $7 " on GPU " $2}'
[ -f logs/done ] && echo "DONE $(tail -n 1 logs/phases.txt)"
exit 0
