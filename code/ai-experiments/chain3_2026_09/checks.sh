#!/usr/bin/env bash
# The host's checks before the third chain. First the step at the confirmation tier, where the chain spends most of its card-hours, on all four cards at once (host_checks_2026_09/step.py, 200 steps, compiled, K = 9): Muon on cards 0 and 1, AdamW on 2 and 3, each with its peak memory, against the 746 and 655 ms the cost model takes (chain3_plan_2026_09) and the peaks of RUNBOOK section 4. Then, at once, the suite on card 0 (RUNBOOK section 3) and, on cards 1 to 3, the third chain end to end with every run cut to 20 steps, compiled as the chain trains, seed 0, into smoke/: each worker runs the confirmation block (the re-check of four rates per optimizer, then (a), (b), (c) at 0.7 times the winners) and then the balancer block, as chain.sh does; then both blocks evaluated on the panel and the annex by three workers (16 sequences a batch at the confirmation tier, 32 at the mechanism tier), and both read by tfg.analysis, the confirmation block with the balancer block as its other tier; last, the runs' settings against what the plan fixes and the claims each block's analysis reads. Run from ~/code after setup.sh; writes logs/step-gpu<n>.txt, logs/pytest.txt, logs/smoke-gpu<n>.log, logs/smoke-eval-gpu<n>.log, logs/smoke-analysis-<block>.log, logs/smoke-settings.txt and logs/checks.txt (one line per stage with its exit codes), then removes checks/ and smoke/.
set -u
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs checks
note() { echo "$(date -u '+%F %T') $*" >> logs/checks.txt; }
pids=()
for gpu in 0 1 2 3; do
  partition=$([ $gpu -lt 2 ] && echo muon || echo adamw)
  CUDA_VISIBLE_DEVICES=$gpu env PYTHONPATH=. uv run python ai-experiments/host_checks_2026_09/step.py $partition confirmation 9 200 checks/step-gpu$gpu > logs/step-gpu$gpu.txt 2>&1 &
  pids+=($!)
done
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
note "step exits:$codes"
rm -rf checks
P="uv run python -m tfg.program"
COMMON="--runs-dir smoke --seeds 0 --tokens 1310720 --warmup-steps 5 --log-every 10 --eval-every 20 --eval-batches 8 --checkpoint-every 10 --adamw-lr 0.002 --muon-lr 0.004"
t=$(date +%s); (CUDA_VISIBLE_DEVICES=0 uv run pytest -q > logs/pytest.txt 2>&1; echo "exit $? in $(( $(date +%s) - t )) s" >> logs/pytest.txt) &
suite=$!
pids=()
for gpu in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$P confirmation $COMMON --recheck-tokens 1310720 --recompute 9 && $P balancer $COMMON --recompute 1" > logs/smoke-gpu$gpu.log 2>&1 &
  pids+=($!)
done
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
note "smoke training exits:$codes, $(find smoke/confirmation/recheck -name settings.json 2>/dev/null | wc -l) re-check runs, $(ls smoke/confirmation/*/settings.json 2>/dev/null | wc -l) confirmation runs, $(ls smoke/balancer/*/settings.json 2>/dev/null | wc -l) balancer runs, re-check rates $(tr -d ' \n' < smoke/confirmation/recheck/rates.json 2>/dev/null)"
pids=()
for gpu in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$P evaluate --runs-dir smoke --block confirmation --batch-size 16 && $P evaluate --runs-dir smoke --block balancer --batch-size 32" > logs/smoke-eval-gpu$gpu.log 2>&1 &
  pids+=($!)
done
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
note "smoke evaluation exits:$codes, confirmation $(ls smoke/confirmation/*/specialization.json 2>/dev/null | wc -l) panels and $(ls smoke/confirmation/*/annex.json 2>/dev/null | wc -l) annexes, balancer $(ls smoke/balancer/*/specialization.json 2>/dev/null | wc -l) panels and $(ls smoke/balancer/*/annex.json 2>/dev/null | wc -l) annexes"
uv run python -m tfg.analysis --runs-dir smoke/confirmation --reference adamw --other-tier smoke/balancer > logs/smoke-analysis-confirmation.log 2>&1
note "smoke confirmation analysis exit $?, $(ls smoke/confirmation/analysis 2>/dev/null | wc -l) files"
uv run python -m tfg.analysis --runs-dir smoke/balancer --reference adamw > logs/smoke-analysis-balancer.log 2>&1
note "smoke balancer analysis exit $?, $(ls smoke/balancer/analysis 2>/dev/null | wc -l) files"
uv run python - > logs/smoke-settings.txt 2>&1 <<'EOF'
import json
from pathlib import Path

import pandas as pd

rates = json.loads(Path("smoke/confirmation/recheck/rates.json").read_text())
bad = []
for block, expected in (("confirmation", 3), ("balancer", 4)):
    runs = sorted(Path(f"smoke/{block}").glob("*/settings.json"))
    if len(runs) != expected:
        bad.append(f"{block}: {len(runs)} runs, not {expected}")
    for path in runs:
        s = json.loads(path.read_text())
        print(block, path.parent.name, {k: s[k] for k in ("tier", "partition", "seed", "adamw_lr", "muon_lr", "balancer_multiplier", "recompute", "steps", "compiled")})
        if block == "confirmation":
            want = {"tier": "confirmation", "recompute": 9, "adamw_lr": 0.7 * rates["adamw"], "muon_lr": 0.7 * rates["muon"], "balancer_multiplier": 1.0}
        else:
            want = {"tier": "mechanism", "recompute": 1, "adamw_lr": 0.002, "muon_lr": 0.004, "balancer_multiplier": 4.0 if "balancer4x" in path.parent.name else 1.0}
        bad += [f"{path.parent.name}: {k} {s[k]} != {v}" for k, v in want.items() if (abs(s[k] - v) > 1e-12 if isinstance(v, float) else s[k] != v)]
for block, want, absent in (("confirmation", {"O", "O-router", "U", "S", "G", "C4", "C5", "C6"}, {"S-fast", "L", "O-fast"}), ("balancer", {"S-fast", "L", "O-fast", "S", "G", "R", "C4", "C5"}, {"O", "O-router", "U"})):
    claims = pd.read_csv(f"smoke/{block}/analysis/claims.csv")
    contrasts = pd.read_csv(f"smoke/{block}/analysis/contrasts.csv")
    print(block, "claims:", dict(contrasts.groupby("claim").size()))
    bad += [f"{block}: claim {c} missing" for c in want - set(claims["claim"])] + [f"{block}: claim {c} read" for c in absent & set(claims["claim"])]
print("SETTINGS_OK" if not bad else "SETTINGS_BAD " + "; ".join(bad))
EOF
note "smoke settings and claims: $(tail -n 1 logs/smoke-settings.txt)"
wait $suite
note "suite: $(tail -n 2 logs/pytest.txt | tr '\n' ' ')"
rm -rf smoke
echo CHECKS_OK
