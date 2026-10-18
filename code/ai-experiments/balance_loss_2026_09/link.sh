#!/usr/bin/env bash
# Links RQ3's runs of (a), (b) and (c) at seeds 3 to 5 into code/runs/balance_loss beside the block's own, so that the analysis reads each run with its RQ3 run, as the plan's Amendment 5 registers; the links are relative, and a run already there is left as it is. Usage: link.sh
set -euo pipefail
code=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$code/runs/balance_loss"
for partition in adamw muon muon_except_router; do
  for seed in 3 4 5; do
    name=mechanism-$partition-s$seed
    [ -e "$code/runs/balance_loss/$name" ] || ln -s "../rq3/$name" "$code/runs/balance_loss/$name"
  done
done
ls -l "$code/runs/balance_loss" | grep -c -- '->' | sed 's/$/ links/'
