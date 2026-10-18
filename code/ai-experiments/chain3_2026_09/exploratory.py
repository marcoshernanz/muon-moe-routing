"""What no claim of the third chain orders, read from its runs' logs: the loss spikes at the confirmation tier and where the first mixture stood when they came; the first mixture's load and span by window through the whole run, the mechanism tier's early window among them (the busiest expert's load is at most 8, every token with the same expert among its four); the experts left with almost no tokens at the end; the balancer's biases in common mode; and, on the balancer block, what the faster bias does to AdamW and to Muon beyond the registered contrasts.

Post hoc and exploratory: none of it changes a claim's verdict. Run from code/ after summary.py: PYTHONPATH=. uv run python ai-experiments/chain3_2026_09/exploratory.py > ai-experiments/chain3_2026_09/outputs/exploratory.txt
"""

import csv
import json
from pathlib import Path
from statistics import mean

from tfg.analysis import spike_table

ARMS = ("adamw", "muon", "muon_except_router")
SEEDS = (0, 1, 2)
WINDOWS = [(301, 500), (501, 1262), (1263, 2789), (501, 2789), (2790, 5000), (5001, 10000), (10001, 35776), (35777, 45776)]  # 501 to 2,789 is the early window at 45,776 steps, 501 to 1,262 the one at 15,259
DEAD = 0.1 / 32  # a tenth of the fair share


def records(arm: str, seed: int) -> list[dict]:
    return [json.loads(line) for line in open(f"runs/confirmation/confirmation-{arm}-s{seed}/routing.jsonl")]


def span(mixture: dict) -> float:
    return max(mixture["bias"]) - min(mixture["bias"])


print("Loss spikes at the confirmation tier after the warmup, the plan's rule and Qwen's raw one; for each spike under the plan's rule, the first mixture at the nearest routing reading")
for arm in ARMS:
    for seed in SEEDS:
        table = spike_table(Path(f"runs/confirmation/confirmation-{arm}-s{seed}"))
        steps = [int(step) for step in table.index[table["spike"]]]
        print(f"  {arm:20} {seed} spikes {len(steps)}, raw {int(table['spike_raw'].sum())}")
        for step in steps:
            near = min(records(arm, seed), key=lambda r: abs(r["step"] - step))["mixtures"][0]
            print(f"      step {step}: span {span(near):.3f}, busiest expert {near['max_load_ratio']:.2f}")

print("\nThe first mixture by window, busiest expert over the fair share / span, and the readings with the span at 0.99 or more (the ceiling is about 1)")
print("  windows " + " ".join(f"{a}-{b}" for a, b in WINDOWS))
for arm in ARMS:
    for seed in SEEDS:
        recs = records(arm, seed)
        parts = []
        for lo, hi in WINDOWS:
            ms = [r["mixtures"][0] for r in recs if lo <= r["step"] <= hi]
            parts.append(f"{mean(m['max_load_ratio'] for m in ms):.2f}/{mean(span(m) for m in ms):.2f}")
        top = [r["step"] for r in recs if span(r["mixtures"][0]) >= 0.99]
        print(f"  {arm:20} {seed} " + "  ".join(parts) + f"  at the ceiling {len(top)} readings, steps {min(top, default=None)} to {max(top, default=None)}")

print("\nThe end: experts under a tenth of their share at the last reading, per mixture; when each of the first mixture's last had one; over the last 10,000 steps the first mixture's usage perplexity with the bias and without it (32 is even) and its busiest expert")
for arm in ARMS:
    for seed in SEEDS:
        recs = records(arm, seed)
        last = recs[-1]["mixtures"]
        dead = [sum(1 for x in m["load"] if x < DEAD) for m in last]
        alive = [max((r["step"] for r in recs if r["mixtures"][0]["load"][e] >= DEAD), default=0) for e in range(32) if last[0]["load"][e] < DEAD]
        late = [r["mixtures"][0] for r in recs if r["step"] > 35776]
        print(f"  {arm:20} {seed} {dead}; last above a tenth at {alive}; perplexity {mean(m['usage_perplexity'] for m in late):.1f}, without the bias {mean(m['raw_usage_perplexity'] for m in late):.1f}, busiest {mean(m['max_load_ratio'] for m in late):.2f}")

print("\nThe biases in common mode: the first mixture's mean bias and its span at the last reading")
for arm in ARMS:
    for seed in SEEDS:
        m = records(arm, seed)[-1]["mixtures"][0]
        print(f"  {arm:20} {seed} mean {mean(m['bias']):+.3f}, lowest {min(m['bias']):+.3f}, span {span(m):.3f}")

print("\nThe balancer block: the faster bias against the usual one, per seed 6 7 8 and the mean, at equal tokens")
rows = {(r["arm"], int(r["seed"])): r for r in csv.DictReader(open("runs/balancer/analysis/endpoints.csv"))}
for base in ("adamw", "muon"):
    for name in ("val_loss", "bias_span_first", "bias_span", "max_load_ratio_first_early", "bias_span_first_early", "bias_span_first_final", "update_load_correlation", "router_input_similarity_first"):
        diffs = [float(rows[(f"{base}-balancer4x", s)][name]) - float(rows[(base, s)][name]) for s in (6, 7, 8)]
        print(f"  {base}-balancer4x - {base:6} {name:30} " + " ".join(f"{d:+.4f}" for d in diffs) + f"  mean {mean(diffs):+.4f}")
