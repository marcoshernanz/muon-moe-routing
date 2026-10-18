"""What the first mixture's span follows: the router's per-expert score offset, and the geometry behind it.

Across the 27 balanced runs, the integrated span against the integrated spread of the router's batch-mean score across experts (`mean_score_span`), the router-input cosine and the router's logit norm, on log scales; within every run, the same readings over the logged steps after the warmup; the span over the score spread by window; and, per run, every reading of the router's geometry the telemetry keeps. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/mechanism.py > ai-experiments/rq3_reading_2026_09/outputs/mechanism.txt
"""

import numpy as np
from reading import ARMS, BALANCED, LABEL, SEEDS, WARMUP, scalar, steps, window

GEOMETRY = ["bias_span", "mean_score_span", "router_input_similarity", "router_logit_norm", "raw_max_load_ratio", "raw_usage_perplexity", "router_update_common_share", "router_row_similarity", "router_svd_entropy"]

print("Per run, the first mixture integrated after the warmup: " + ", ".join(GEOMETRY))
for arm in ARMS:
    for seed in SEEDS:
        print(f"{LABEL[arm]:10} {seed} " + " ".join(f"{window(arm, seed, k):8.4f}" for k in GEOMETRY))

runs = [(arm, seed) for arm in BALANCED for seed in SEEDS]
span = np.log([window(a, s, "bias_span") for a, s in runs])
columns = {k: np.log([window(a, s, k) for a, s in runs]) for k in ["mean_score_span", "router_input_similarity", "router_logit_norm"]}


def fit(names: list[str]) -> None:
    design = np.column_stack([columns[n] for n in names] + [np.ones(len(span))])  # [R, K + 1]
    coef = np.linalg.lstsq(design, span, rcond=None)[0]
    fitted = design @ coef
    r2 = 1 - ((span - fitted) ** 2).sum() / ((span - span.mean()) ** 2).sum()
    print(f"  {' + '.join(names):72} slopes {' '.join(f'{c:+.3f}' for c in coef[:-1])}, R2 {r2:.3f}")


print(f"\nAcross the {len(runs)} balanced runs, log of the first mixture's integrated span regressed on the logs of:")
for names in (["mean_score_span"], ["router_input_similarity"], ["router_logit_norm"], ["router_input_similarity", "router_logit_norm"], ["mean_score_span", "router_input_similarity", "router_logit_norm"]):
    fit(names)

print("\nWithin every run, over the logged steps after the warmup, Pearson correlations per seed 3 4 5: span with the score spread | span with the cosine | score spread with the cosine")
for arm in BALANCED:
    parts = []
    for pair in [("bias_span", "mean_score_span"), ("bias_span", "router_input_similarity"), ("mean_score_span", "router_input_similarity")]:
        values = []
        for seed in SEEDS:
            after = steps(arm, seed) > WARMUP
            values.append(np.corrcoef(scalar(arm, seed, pair[0])[after], scalar(arm, seed, pair[1])[after])[0, 1])
        parts.append(" ".join(f"{v:+.2f}" for v in values))
    print(f"{LABEL[arm]:10} " + " | ".join(parts))

WINDOWS = [(500, 1262), (1262, 3000), (3000, 6000), (6000, 10000), (10000, 15259)]
print("\nThe span over the score spread, the first mixture, the mean of the three seeds' ratios, per window " + " ".join(f"{a + 1}-{b}" for a, b in WINDOWS))
for arm in BALANCED:
    print(f"{LABEL[arm]:10} " + " ".join(f"{np.mean([window(arm, s, 'bias_span', a, b) / window(arm, s, 'mean_score_span', a, b) for s in SEEDS]):6.2f}" for a, b in WINDOWS))
