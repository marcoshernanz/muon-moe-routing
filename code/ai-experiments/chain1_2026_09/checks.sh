#!/usr/bin/env bash
# The host's checks before the chain: the step of the mechanism tier under Muon on all four cards at once (step.py of the host checks, 300 steps, compiled, K = 1), to set each card against the 365.6 ms of EVIDENCE-LOG section 15, then the suite on card 0 (RUNBOOK section 3). Run from ~/code after setup.sh; writes logs/step-gpu<n>.txt and logs/pytest.txt, and removes the step runs' directories.
set -u
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs checks
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu env PYTHONPATH=. uv run python ai-experiments/host_checks_2026_09/step.py muon mechanism 1 300 checks/step-gpu$gpu > logs/step-gpu$gpu.txt 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid; done
rm -rf checks
t=$(date +%s); CUDA_VISIBLE_DEVICES=0 uv run pytest -q > logs/pytest.txt 2>&1; echo "exit $? in $(( $(date +%s) - t )) s" >> logs/pytest.txt
echo CHECKS_OK
