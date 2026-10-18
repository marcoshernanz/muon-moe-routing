#!/usr/bin/env bash
# The host's checks before the second chain. First Muon's step at the mechanism tier on all four cards at once (host_checks_2026_09/step.py, 300 steps, compiled, K = 1), against the 363.7 to 374.7 ms of the first chain's host (EVIDENCE-LOG section 16). Then, at once, the suite on card 0 (RUNBOOK section 3) and, on cards 1 to 3, the RQ3 block end to end at the mechanism tier cut to 20 steps, compiled as the chain trains (eager does not fit the mechanism tier in 32 GB), seed 0, into smoke/: its 11 arms trained by three workers, evaluated on the panel and the annex at batch 32 by three workers, and read by tfg.analysis. Run from ~/code after setup.sh; writes logs/step-gpu<n>.txt, logs/pytest.txt, logs/smoke-gpu<n>.log, logs/smoke-eval-gpu<n>.log, logs/smoke-analysis.log and logs/checks.txt (one line per stage with its exit codes), then removes checks/ and smoke/.
set -u
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs checks
note() { echo "$(date -u '+%F %T') $*" >> logs/checks.txt; }
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu env PYTHONPATH=. uv run python ai-experiments/host_checks_2026_09/step.py muon mechanism 1 300 checks/step-gpu$gpu > logs/step-gpu$gpu.txt 2>&1 &
  pids+=($!)
done
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
note "step exits:$codes"
rm -rf checks
P="uv run python -m tfg.program"
SMOKE="--runs-dir smoke --seeds 0 --tokens 1310720 --warmup-steps 5 --log-every 10 --eval-every 20 --eval-batches 8 --checkpoint-every 10 --recompute 1 --adamw-lr 0.002 --muon-lr 0.004"
t=$(date +%s); (CUDA_VISIBLE_DEVICES=0 uv run pytest -q > logs/pytest.txt 2>&1; echo "exit $? in $(( $(date +%s) - t )) s" >> logs/pytest.txt) &
suite=$!
pids=()
for gpu in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu $P rq3 $SMOKE > logs/smoke-gpu$gpu.log 2>&1 &
  pids+=($!)
done
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
note "smoke training exits:$codes, $(ls -d smoke/rq3/*/ 2>/dev/null | wc -l) run directories"
pids=()
for gpu in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu $P evaluate --runs-dir smoke --block rq3 --batch-size 32 > logs/smoke-eval-gpu$gpu.log 2>&1 &
  pids+=($!)
done
codes=""; for pid in "${pids[@]}"; do wait $pid; codes="$codes $?"; done
note "smoke evaluation exits:$codes, $(ls smoke/rq3/*/specialization.json 2>/dev/null | wc -l) panels, $(ls smoke/rq3/*/annex.json 2>/dev/null | wc -l) annexes"
uv run python -m tfg.analysis --runs-dir smoke/rq3 --reference adamw > logs/smoke-analysis.log 2>&1
note "smoke analysis exit $?, $(ls smoke/rq3/analysis 2>/dev/null | wc -l) files"
wait $suite
note "suite: $(tail -n 2 logs/pytest.txt | tr '\n' ' ')"
rm -rf smoke
echo CHECKS_OK
