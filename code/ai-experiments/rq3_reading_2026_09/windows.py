"""The second chain's first mixture over time: when the span, its offset and the geometry rise and fall, and when the realized load departs from the fair share.

Per arm, the seed mean of every reading over windows of steps (the warmup in three parts, the 5 percent after it, and on to the end); when each run's readings first pass a threshold and where they peak; and the router's and the gains' weight norms by window. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/windows.py > ai-experiments/rq3_reading_2026_09/outputs/windows.txt
"""

import numpy as np
from reading import ARMS, LABEL, SEEDS, group, scalar, seed_mean, steps

WINDOWS = [(0, 100), (100, 300), (300, 500), (500, 1262), (1262, 3000), (3000, 6000), (6000, 10000), (10000, 14259), (14259, 15259)]
KEYS = ["bias_span", "mean_score_span", "router_input_similarity", "router_logit_norm", "raw_max_load_ratio", "max_load_ratio", "usage_perplexity", "load_churn", "router_entropy", "margin_mean"]

for key in KEYS:
    print(f"\n{key}, the first mixture, seed mean, windows " + " ".join(f"{a + 1}-{b}" for a, b in WINDOWS))
    for arm in ARMS:
        print(f"{LABEL[arm]:10} " + " ".join(f"{seed_mean(arm, key, a, b):8.3f}" for a, b in WINDOWS))

print("\nmax_load_ratio, the mean of the seven mixtures, seed mean, same windows")
for arm in ARMS:
    print(f"{LABEL[arm]:10} " + " ".join(f"{np.mean([seed_mean(arm, 'max_load_ratio', a, b, block) for block in range(7)]):8.3f}" for a, b in WINDOWS))

for name in ["router", "gains", "attention", "experts"]:
    print(f"\nThe {name} group's weight norm, seed mean, same windows")
    for arm in ARMS:
        values = []
        for a, b in WINDOWS:
            values.append(np.mean([np.nanmean(group(arm, s, name)[(steps(arm, s) > a) & (steps(arm, s) <= b)]) for s in SEEDS]))
        print(f"{LABEL[arm]:10} " + " ".join(f"{v:8.3f}" for v in values))


def first_above(at: np.ndarray, values: np.ndarray, threshold: float) -> str:
    above = values > threshold
    return str(at[np.argmax(above)]) if above.any() else "never"


print("\nPer run, the first mixture: the first logged step with the cosine above 0.2 and 0.35 after step 100, the score spread above 0.3, the span above 0.3; the span's and the score spread's peaks and their steps; the logit norm's peak")
for arm in ARMS:
    for seed in SEEDS:
        at = steps(arm, seed)
        late = at > 100
        cosine, spread, span, norm = (scalar(arm, seed, k) for k in ["router_input_similarity", "mean_score_span", "bias_span", "router_logit_norm"])
        print(
            f"{LABEL[arm]:10} {seed} cosine>0.2 {first_above(at[late], cosine[late], 0.2):>5}, >0.35 {first_above(at[late], cosine[late], 0.35):>5}; spread>0.3 {first_above(at, spread, 0.3):>5}; span>0.3 {first_above(at, span, 0.3):>5}; "
            f"span peak {span.max():.3f} at {at[np.argmax(span)]}, spread peak {spread.max():.3f} at {at[np.argmax(spread)]}, logit norm peak {norm.max():.1f} at {at[np.argmax(norm)]}"
        )
