#!/usr/bin/env bash
# The pilot's runs on one host, one after another on its one card: the tuning block's settings (the mechanism tier, 0.5B tokens in 7,629 steps, seed 0, a cosine schedule, compiled, one recomputed block) at the tuned rates, AdamW 2e-3 and Muon 4e-3. Host A first runs the suite on the card. The first run of each host repeats a tuning run of the first chain with the new readings, which compare.py checks bit for bit against it. Run from ~/code, detached (setsid -f), as `pilot.sh A` or `pilot.sh B`; a relaunch resumes. Appends to logs/pilot.log, logs/gpu.csv and logs/phases.txt, and touches logs/done when the list is over.
set -u
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs runs/pilot
rm -f logs/done
stamp() { echo "$(date -u '+%F %T') $*" >> logs/phases.txt; }
nvidia-smi --query-gpu=timestamp,power.draw,power.limit,utilization.gpu,memory.used,temperature.gpu,clocks.sm --format=csv,noheader -l 30 >> logs/gpu.csv &
watch=$!
T="uv run python -m tfg.train --tier mechanism --steps 7629 --compiled --recompute 1 --resume"
run() {  # name, then the flags that set the arm
  local name=$1; shift
  stamp "start $name"
  $T --run-dir runs/pilot/$name "$@" >> logs/pilot.log 2>&1
  stamp "end $name exit $?"
}
case "$1" in
  A)
    if [ ! -f logs/pytest.txt ]; then
      t=$(date +%s); uv run pytest -q > logs/pytest.txt 2>&1; echo "exit $? in $(( $(date +%s) - t )) s" >> logs/pytest.txt
      stamp "pytest $(tail -1 logs/pytest.txt)"
    fi
    run mechanism-adamw-s0 --partition adamw --adamw-lr 0.002 --muon-lr 0.001
    run mechanism-muon_except_router-s0 --partition muon_except_router --adamw-lr 0.002 --muon-lr 0.004
    run mechanism-muon_dense-s0 --partition muon_dense --adamw-lr 0.002 --muon-lr 0.004
    run mechanism-muon_experts-s0 --partition muon_experts --adamw-lr 0.002 --muon-lr 0.004
    run mechanism-adamw-unbalanced-s0 --partition adamw --no-balanced --adamw-lr 0.002 --muon-lr 0.004
    ;;
  B)
    run mechanism-muon-s0 --partition muon --adamw-lr 0.002 --muon-lr 0.004
    run mechanism-muon_router-s0 --partition muon_router --adamw-lr 0.002 --muon-lr 0.004
    run mechanism-muon-unbalanced-s0 --partition muon --no-balanced --adamw-lr 0.002 --muon-lr 0.004
    run mechanism-muon_attention-s0 --partition muon_attention --adamw-lr 0.002 --muon-lr 0.004
    ;;
esac
kill $watch 2>/dev/null
touch logs/done
