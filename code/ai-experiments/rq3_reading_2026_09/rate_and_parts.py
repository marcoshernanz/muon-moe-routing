"""Muon's rate and the parts of the backbone, on the loss and on the first mixture's span.

The validation loss over training for AdamW, Muon at its tuned rate and at AdamW's rate, and Muon with the router on AdamW, with Muon at half its rate minus Muon at its tuned rate per seed; every arm's loss and span beside what it costs or gains against (a) and (b); and the excess of the span over (a)'s at Muon's two rates, in the first mixture and over all seven. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/rate_and_parts.py > ai-experiments/rq3_reading_2026_09/outputs/rate_and_parts.txt
"""

import numpy as np
from reading import ARMS, BALANCED, LABEL, SEEDS, array, seed_mean

POINTS = [500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 15259]


def validation(arm: str, seed: int, step: int) -> float:
    at, loss = array(arm, seed, "val_steps"), array(arm, seed, "val_loss")
    return float(loss[min(np.searchsorted(at, step), len(at) - 1)])


print("Validation loss at the evaluation at or after steps " + " ".join(map(str, POINTS)) + ", seed mean")
for arm in ["adamw", "muon", "muon-muonlr0.5x", "muon_except_router"]:
    print(f"{LABEL[arm]:10} " + " ".join(f"{np.mean([validation(arm, s, p) for s in SEEDS]):.4f}" for p in POINTS))
print("\nMuon at half its rate minus Muon at its tuned rate, per seed, same steps")
for seed in SEEDS:
    print(f"seed {seed}     " + " ".join(f"{validation('muon-muonlr0.5x', seed, p) - validation('muon', seed, p):+.4f}" for p in POINTS))

final = {arm: np.mean([validation(arm, s, 15259) for s in SEEDS]) for arm in ARMS}
print("\nPer arm, seed mean: final validation loss, its gain over (a) and the share of (b)'s gain it takes; the first mixture's integrated span and its excess over (a)'s; the span over all seven mixtures and its excess")
span_first = {arm: seed_mean(arm, "bias_span") for arm in BALANCED}
span_all = {arm: np.mean([seed_mean(arm, "bias_span", block=b) for b in range(7)]) for arm in BALANCED}
for arm in ARMS:
    gain = final["adamw"] - final[arm]
    share = gain / (final["adamw"] - final["muon"])
    line = f"{LABEL[arm]:10} {final[arm]:.4f} gain {gain:+.4f} ({share:5.0%} of (b)'s)"
    if arm in BALANCED:
        line += f"; first {span_first[arm]:.3f} ({span_first[arm] - span_first['adamw']:+.3f}, {span_first[arm] / span_first['adamw'] - 1:+.0%}); all {span_all[arm]:.3f} ({span_all[arm] - span_all['adamw']:+.3f}, {span_all[arm] / span_all['adamw'] - 1:+.0%})"
    print(line)
parts = sum(final["adamw"] - final[a] for a in ["muon_router", "muon_experts", "muon_attention", "muon_dense"])
print(f"\nThe four single parts' gains summed ((g), (d), (e), (h)): {parts:+.4f}, against (b)'s {final['adamw'] - final['muon']:+.4f}")
