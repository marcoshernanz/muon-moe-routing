#!/usr/bin/env bash
# The block end to end on the laptop before the host, at the laptop tier on the CPU, 200 steps a run (102,400 tokens), uncompiled: RQ3's three partitions at seeds 3 to 5 trained by hand as the references, the program's balance_loss block, the references linked in as link.sh links RQ3's, and the analysis, whose claims table must carry the five claims of Amendment 5 decided (passed or failed, not incomplete). The numbers decide nothing; the check is that every step runs and the analysis pairs each run with its reference. Usage, from code/: bash ai-experiments/balance_loss_2026_09/e2e.sh DIR; writes outputs/e2e.txt.
set -euo pipefail
dir=$1
here=$(cd "$(dirname "$0")" && pwd)
common=(--adamw-lr 2e-3 --muon-lr 4e-3 --warmup-steps 20 --eval-every 50 --checkpoint-every 100 --no-compiled)
rm -rf "$dir"; mkdir -p "$dir/rq3"
for seed in 3 4 5; do
  for partition in adamw muon muon_except_router; do
    uv run python -m tfg.train --tier laptop --partition $partition --seed $seed --steps 200 "${common[@]}" --run-dir "$dir/rq3/laptop-$partition-s$seed" > /dev/null
  done
done
uv run python -m tfg.program balance_loss --tier laptop --tokens 102400 --runs-dir "$dir" "${common[@]}" > "$dir/block.log" 2>&1
for partition in adamw muon muon_except_router; do
  for seed in 3 4 5; do ln -s "../rq3/laptop-$partition-s$seed" "$dir/balance_loss/laptop-$partition-s$seed"; done
done
uv run python -m tfg.analysis --runs-dir "$dir/balance_loss" --reference adamw > "$dir/analysis.log" 2>&1
{
  echo "runs: $(ls "$dir/balance_loss" | grep -c laptop), of which linked: $(find "$dir/balance_loss" -maxdepth 1 -type l | wc -l | tr -d ' ')"
  for run in "$dir"/balance_loss/laptop-*balanceloss*; do
    echo "$(basename "$run"): balance_loss $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['balance_loss'])" "$run/settings.json"), last sequence_balance $(tail -1 "$run/metrics.jsonl" | python3 -c "import json,sys; print(round(json.load(sys.stdin)['sequence_balance'], 4))")"
  done
  python3 - "$dir/balance_loss/analysis/claims.csv" <<'PY'
import csv, sys
for row in csv.DictReader(open(sys.argv[1])):
    if row["claim"] in ("S-loss", "G-loss", "L-loss", "O-loss", "L-loss-router"):
        print({k: row[k] for k in row})
PY
} | tee "$here/outputs/e2e.txt"
