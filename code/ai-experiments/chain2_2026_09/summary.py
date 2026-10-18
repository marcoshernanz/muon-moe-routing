"""The second chain's numbers for the evidence log, from the analysis's tables and the host's logs.

Per run, the registered endpoints; per arm, the step time and the first mixture's span early and at the end; the exploratory pairs no claim orders ((b) against (c), and the parts no claim names against (a)); Muon at half its rate against Muon at its tuned rate; the cards and the disk. Run from code/ after tfg.analysis: uv run python ai-experiments/chain2_2026_09/summary.py > ai-experiments/chain2_2026_09/outputs/summary.txt
"""

import csv
import glob
import json
import os
import re
from statistics import mean, median

import numpy as np

ARMS = ["adamw", "muon", "muon_except_router", "muon_router", "muon_experts", "muon_attention", "muon_dense", "muon_attention_dense", "muon-muonlr0.5x", "adamw-unbalanced", "muon-unbalanced"]
SEEDS = (3, 4, 5)
TOKENS_PER_STEP = 65536
LOGS = "ai-experiments/chain2_2026_09/outputs/host-logs"

rows = {(r["arm"], int(r["seed"])): r for r in csv.DictReader(open("runs/rq3/analysis/endpoints.csv"))}


def value(arm: str, seed: int, name: str) -> float:
    return float(rows[(arm, seed)][name])


print("Per run: validation loss, first mixture's span (S), cosine (G), span over all mixtures (R), warmup-window churn (M), final usage perplexity (C3), specialization (C4), update-load correlation (C6), experts collapsed at the end")
for arm in ARMS:
    for seed in SEEDS:
        names = ["val_loss", "bias_span_first", "router_input_similarity_first", "bias_span", "load_churn_warmup", "usage_perplexity_final", "specialization", "update_load_correlation", "collapsed_experts_final"]
        print(f"{arm:22} {seed} " + " ".join(f"{value(arm, seed, n):8.4f}" for n in names))

print("\nPer arm: step time (ms, median of each run after its first 500 steps), the first mixture's span early (the 5 percent of steps after the warmup) and at the last logged step, spikes under the plan's rule")
for arm in ARMS:
    times = []
    for seed in SEEDS:
        run = f"runs/rq3/mechanism-{arm}-s{seed}"
        rates = [json.loads(line).get("tokens_per_s") for line in open(f"{run}/metrics.jsonl")]
        times.append(TOKENS_PER_STEP / median([r for r in rates if r][50:]) * 1000)
    early = [value(arm, s, "bias_span_first_early") for s in SEEDS]
    final = [value(arm, s, "bias_span_first_final") for s in SEEDS]
    spikes = [value(arm, s, "spikes") for s in SEEDS]
    print(f"{arm:22} {min(times):.0f} to {max(times):.0f} ms, {mean(times) * 15259 / 3.6e6:.2f} h a run; early {' '.join(f'{x:.3f}' for x in early)}; final {' '.join(f'{x:.3f}' for x in final)}; spikes {sum(spikes):.0f}")

print("\nPairs no claim orders, on the first mixture's integrated span (first minus second, per seed)")
for a, b in [("muon", "muon_except_router"), ("muon_attention", "adamw"), ("muon_dense", "adamw"), ("muon_router", "adamw")]:
    d = [value(a, s, "bias_span_first") - value(b, s, "bias_span_first") for s in SEEDS]
    print(f"{a} - {b}: {' '.join(f'{x:+.4f}' for x in d)}, mean {mean(d):+.4f}")

print("\nMuon at half its rate minus Muon at its tuned rate, per seed")
for name in ["val_loss", "bias_span_first", "router_input_similarity_first", "bias_span"]:
    d = [value("muon-muonlr0.5x", s, name) - value("muon", s, name) for s in SEEDS]
    print(f"{name}: {' '.join(f'{x:+.4f}' for x in d)}, mean {mean(d):+.4f}")

cards = list(csv.reader(open(f"{LOGS}/gpus.csv")))
busy = [r for r in cards if float(r[4].strip(" %")) >= 90]
watts = np.array([float(r[2].strip(" W")) for r in busy])
print(f"\nCards: {len(cards)} readings, {len(busy)} at 90 percent or more; there {np.median(watts):.0f} W median, {np.percentile(watts, 5):.0f} W at the 5th percentile, {median(float(r[7].strip(' MHz')) for r in busy):.0f} MHz median; at most {max(float(r[6]) for r in cards):.0f} C; limits {sorted({r[3].strip() for r in cards})}")
free = [int(m.group(1)) for m in (re.search(r"disk_avail=(\d+)G", line) for line in open(f"{LOGS}/host.txt")) if m]
print(f"Free disk: {free[0]} GB at the start, {min(free)} GB at the least")
print(f"Run directories: {len(glob.glob('runs/rq3/mechanism-*'))}, panels {len(glob.glob('runs/rq3/*/specialization.json'))}, annexes {len(glob.glob('runs/rq3/*/annex.json'))} at batch sizes {sorted({json.load(open(f))['batch_size'] for f in glob.glob('runs/rq3/*/annex.json')})}")
print(f"Phases: {open(f'{LOGS}/phases.txt').read().strip()}")
print(f"Diverged: {open(f'{LOGS}/diverged.count').read().strip()}; workers' logs {sorted(os.path.basename(f) for f in glob.glob(f'{LOGS}/second-gpu*'))}")

print("\nWhether a rise sits on one expert: the first mixture's biases over steps 2,001 to 3,000 and over the last 1,000 steps, mean over the readings; the lowest bias, the others' mean, the span, and the span without the lowest expert")
for arm in ["adamw", "muon", "muon_except_router", "muon_experts", "muon_attention", "muon_attention_dense"]:
    for seed in SEEDS:
        records = [json.loads(line) for line in open(f"runs/rq3/mechanism-{arm}-s{seed}/routing.jsonl")]
        parts = []
        for low, high in [(2000, 3000), (14259, 15259)]:
            biases = np.array([r["mixtures"][0]["bias"] for r in records if low < r["step"] <= high])  # [T, E]
            lowest = biases.argmin(axis=1)  # [T]
            rest = np.array([np.delete(b, i) for b, i in zip(biases, lowest)])  # [T, E - 1]
            parts.append(f"{biases.min(axis=1).mean():+.3f} {rest.mean():+.3f} {(biases.max(axis=1) - biases.min(axis=1)).mean():.3f} {(rest.max(axis=1) - rest.min(axis=1)).mean():.3f} (expert {np.bincount(lowest).argmax()} lowest in {np.bincount(lowest).max() / len(lowest):.0%})")
        print(f"{arm:22} {seed}  2,001-3,000: {parts[0]};  last 1,000: {parts[1]}")
