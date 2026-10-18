#!/usr/bin/env bash
# One look at the chain for the laptop's watch.sh: the workers' new log lines that name a problem since the last look (PROBLEM), how many workers are alive (WORKERS), the disk (a PROBLEM under 40 GB free), each card's last power limit and temperature (a PROBLEM under 575 W or over 88 C), and DONE once chain.sh has touched logs/done. Run from anywhere on the host.
cd ~/code
for f in logs/first-gpu*.log; do
  [ -f "$f" ] || continue
  o=$(cat "$f.off" 2>/dev/null || echo 0); n=$(stat -c %s "$f")
  tail -c +$((o + 1)) "$f" | head -c $((n - o)) | grep -E "Traceback|Error|error:|refused|Killed|out of memory|still an edge|stops before|widen" | cut -c1-300 | sed "s|^|PROBLEM $f: |"
  echo "$n" > "$f.off"
done
alive=0; for pid in $(cat logs/workers.pid 2>/dev/null); do kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); done
echo "WORKERS $alive"
avail=$(df -BG ~ | awk 'NR==2{gsub("G","",$4); print $4}')
echo "DISK ${avail}G free"
[ "$avail" -lt 40 ] && echo "PROBLEM disk: ${avail}G free"
tail -n 4 logs/gpus.csv 2>/dev/null | awk -F', ' '{split($4,l," "); split($7,t," "); printf "GPU %s %s %s %s %s\n", $2, $3, $5, $7, $4; if (l[1] + 0 < 575) print "PROBLEM power limit " $4 " on GPU " $2; if (t[1] + 0 > 88) print "PROBLEM temperature " $7 " on GPU " $2}'
[ -f logs/done ] && echo "DONE $(tail -n 1 logs/phases.txt)"
exit 0
