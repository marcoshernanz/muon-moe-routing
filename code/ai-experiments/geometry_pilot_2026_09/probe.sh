#!/usr/bin/env bash
# One look at a pilot host for watch.sh, run there from ~/code: the phases so far, whether the list is over, the run under way with its last logged step and loss, new log lines naming a problem, the card's power, temperature and memory, and the free disk.
cd ~/code
echo "phases $(wc -l < logs/phases.txt 2>/dev/null || echo 0)"
tail -2 logs/phases.txt 2>/dev/null
[ -f logs/done ] && echo DONE
[ -f logs-setup.out ] && grep -q "SETUP_OK" logs-setup.out 2>/dev/null || { [ -f logs/setup.txt ] && echo "setting up: $(tail -1 logs/setup.txt)"; }
for m in runs/pilot/*/metrics.jsonl; do
  [ -f "$m" ] || continue
  echo "$(basename $(dirname $m)) $(tail -1 $m | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("step", d["step"], "loss", round(d["loss"], 4), "val", d.get("val_loss"))' 2>/dev/null)"
done
grep -n -i -E "traceback|error|nan|out of memory|killed" logs/pilot.log logs-setup.out 2>/dev/null | grep -v -i "expandable_segments" | tail -3 | sed 's/^/PROBLEM /'
nvidia-smi --query-gpu=power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader
df -BG ~ | awk 'NR==2{print "disk free", $4}'
