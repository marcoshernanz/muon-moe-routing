"""An independent check of the second chain's runs and of the analysis's primary readings, without tfg.

Every run's settings against its arm of the plan's section 7.1, one data order for all (train_windows), a routing reading at every tenth step to 15,259 and no step logged twice, no loss that is not a number, and the first mixture's integrated bias span and router-input cosine recomputed from routing.jsonl (the span as the largest bias minus the smallest, the mean over the logged steps after the warmup) against runs/rq3/analysis/endpoints.csv. Run from code/: uv run python ai-experiments/chain2_2026_09/crosscheck.py > ai-experiments/chain2_2026_09/outputs/crosscheck.txt
"""

import csv
import glob
import json
import os
from statistics import mean

ARMS = {  # arm: (partition, muon_lr_multiplier, balanced)
    "adamw": ("adamw", 1.0, True),
    "muon": ("muon", 1.0, True),
    "muon_except_router": ("muon_except_router", 1.0, True),
    "muon_router": ("muon_router", 1.0, True),
    "muon_experts": ("muon_experts", 1.0, True),
    "muon_attention": ("muon_attention", 1.0, True),
    "muon_dense": ("muon_dense", 1.0, True),
    "muon_attention_dense": ("muon_attention_dense", 1.0, True),
    "muon-muonlr0.5x": ("muon", 0.5, True),
    "adamw-unbalanced": ("adamw", 1.0, False),
    "muon-unbalanced": ("muon", 1.0, False),
}
COMMON = dict(steps=15259, warmup_steps=500, adamw_lr=0.002, muon_lr=0.004, adamw_clip=1.0, muon_clip=1.0, schedule="cosine", compiled=True, recompute=1, lr_multiplier=1.0, router_lr_multiplier=1.0, balancer_multiplier=1.0, batch_size=64, log_every=10, eval_every=250, adamw_beta2=0.95)

endpoints = {(r["arm"], r["seed"]): r for r in csv.DictReader(open("runs/rq3/analysis/endpoints.csv"))}
worst_span = worst_similarity = 0.0
windows, problems, seen = set(), [], set()
for run in sorted(glob.glob("runs/rq3/mechanism-*")):
    settings = json.load(open(f"{run}/settings.json"))
    name = os.path.basename(run)
    arm = name[len("mechanism-") : name.rindex("-s")]
    seed = settings["seed"]
    seen.add((arm, seed))
    partition, multiplier, balanced = ARMS[arm]
    for key, value in dict(COMMON, partition=partition, muon_lr_multiplier=multiplier, balanced=balanced).items():
        if settings[key] != value:
            problems.append((name, key, settings[key], value))
    windows.add(settings["train_windows"])
    steps, spans, similarities = [], [], []
    for line in open(f"{run}/routing.jsonl"):
        record = json.loads(line)
        steps.append(record["step"])
        if record["step"] > settings["warmup_steps"]:
            first = record["mixtures"][0]
            spans.append(max(first["bias"]) - min(first["bias"]))
            similarities.append(first["router_input_similarity"])
    if steps != list(range(10, 15251, 10)) + [15259]:
        problems.append((name, "routing steps", len(steps), len(set(steps))))
    metrics = [json.loads(line) for line in open(f"{run}/metrics.jsonl")]
    if len({m["step"] for m in metrics}) != len(metrics):
        problems.append((name, "a metrics step logged twice"))
    if any(x != x or abs(x) == float("inf") for m in metrics for x in m.get("losses", [])):
        problems.append((name, "a loss that is not a number"))
    evaluated = [m for m in metrics if "val_loss" in m]
    row = endpoints[(arm, str(seed))]
    d_span = abs(mean(spans) - float(row["bias_span_first"]))
    d_similarity = abs(mean(similarities) - float(row["router_input_similarity_first"]))
    worst_span, worst_similarity = max(worst_span, d_span), max(worst_similarity, d_similarity)
    print(f"{name:36} readings {len(steps)} to step {steps[-1]}, span {mean(spans):.4f} (off by {d_span:.1e}), cosine {mean(similarities):.4f} (off by {d_similarity:.1e}), validation loss {evaluated[-1]['val_loss']:.4f} at step {evaluated[-1]['step']}")
missing = sorted({(a, s) for a in ARMS for s in (3, 4, 5)} - seen)
print(f"runs {len(seen)}, missing {missing}; train_windows {sorted(windows)}")
print(f"largest difference from the analysis: span {worst_span:.1e}, cosine {worst_similarity:.1e}")
print(f"problems: {problems or 'none'}")
