#!/usr/bin/env bash
# Whether the change leaves a run without the loss as it was, bit for bit: the code of the last commit (git archive HEAD, whose training path is the tag frozen-2026-09-25's) and the working tree train the same laptop-tier run of all-Muon at seed 3, 40 steps, with the first mixture block recomputed, eager and compiled, on the laptop's CPU; baseline.py compares the two runs' logs (the time per step aside) and saved weights. Usage, from code/: bash ai-experiments/balance_loss_2026_09/baseline.sh DIR; writes outputs/baseline.txt.
set -euo pipefail
dir=$1
here=$(cd "$(dirname "$0")" && pwd)
code=$(cd "$here/../.." && pwd)
rm -rf "$dir"; mkdir -p "$dir/head"
git -C "$code/.." archive HEAD:code tfg | tar x -C "$dir/head"
ln -s "$code/data" "$dir/head/data"
args=(--tier laptop --partition muon --seed 3 --steps 40 --warmup-steps 10 --log-every 10 --eval-every 20 --checkpoint-every 20 --recompute 1)
for mode in eager compiled; do
  flag=--no-compiled; [ $mode = compiled ] && flag=--compiled
  (cd "$dir/head" && uv run --project "$code" python -m tfg.train "${args[@]}" $flag --run-dir "$dir/before-$mode" > "$dir/before-$mode.log" 2>&1)
  (cd "$code" && uv run python -m tfg.train "${args[@]}" $flag --run-dir "$dir/after-$mode" > "$dir/after-$mode.log" 2>&1)
done
(cd "$code" && uv run python "$here/baseline.py" "$dir") | tee "$here/outputs/baseline.txt"
