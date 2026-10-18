"""Whether a rise of the first mixture's span sits on one expert, and what that expert does.

Per run, the expert whose bias is most often the lowest after step 1,000, and, per window of steps, its bias against the others' mean, its load over the fair share, its update's norm over the median expert's (the Frobenius norm C6 reads) and the share of readings in which it has the lowest bias. The expert with the lowest bias is the one the router's scores favour most, since the bias lowers the scores of the experts the router overloads. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/one_expert.py > ai-experiments/rq3_reading_2026_09/outputs/one_expert.txt
"""

import numpy as np
from reading import LABEL, SEEDS, array, steps

WINDOWS = [(0, 500), (500, 1262), (1262, 3000), (3000, 6000), (6000, 10000), (10000, 15259)]
print("Per run: the expert most often lowest after step 1,000; per window " + " ".join(f"{a + 1}-{b}" for a, b in WINDOWS) + ": its bias/the others' mean bias, its load over the fair share, its update norm over the median expert's, the share of readings it is lowest in")
for arm in ["muon_experts", "muon_attention_dense", "muon_attention", "muon", "muon_except_router", "muon-muonlr0.5x", "adamw"]:
    for seed in SEEDS:
        at, bias, load, update = steps(arm, seed), array(arm, seed, "bias"), array(arm, seed, "load"), array(arm, seed, "update")
        experts = bias.shape[1]
        low = int(np.bincount(bias[at > 1000].argmin(1), minlength=experts).argmax())
        parts = []
        for a, b in WINDOWS:
            m = (at > a) & (at <= b)
            others = np.delete(bias[m], low, axis=1).mean()
            relative = np.nanmean(update[m][:, low] / np.median(update[m], axis=1))
            parts.append(f"{bias[m][:, low].mean():+.2f}/{others:+.2f} {load[m][:, low].mean() * experts:.2f} {relative:.2f} {(bias[m].argmin(1) == low).mean():4.0%}")
        print(f"{LABEL[arm]:10} {seed} expert {low:2}: " + " | ".join(parts))
