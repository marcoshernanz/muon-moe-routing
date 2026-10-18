"""The second chain mixture by mixture: how the shared direction, the span and the router's structure change with depth, what the unbalanced arms do in each mixture, and the activation outliers per arm.

Per arm, the seed mean of every reading integrated after the warmup in each of the seven mixtures; the unbalanced arms' usage at the last reading per mixture; the router's SVD entropy and row similarity in the first mixture and over all of them (Moonlight and Srivastava et al. read both on Muon routers); and the largest residual, router logit and expert output and the experts' largest kurtosis, which the outlier literature reads (Muon is reported to leave fewer outliers). Run from code/: uv run python ai-experiments/rq3_reading_2026_09/blocks.py > ai-experiments/rq3_reading_2026_09/outputs/blocks.txt
"""

import numpy as np
from reading import ARMS, LABEL, SEEDS, scalar, seed_mean

for key in ["bias_span", "router_input_similarity", "mean_score_span", "router_logit_norm", "raw_max_load_ratio", "max_load_ratio", "load_churn", "router_update_common_share"]:
    print(f"\n{key}, integrated after the warmup, seed mean, per mixture 1 to 7")
    for arm in ARMS:
        print(f"{LABEL[arm]:10} " + " ".join(f"{seed_mean(arm, key, block=b):7.3f}" for b in range(7)))

print("\nThe unbalanced arms at the last reading, per seed and mixture 1 to 7: usage perplexity (32 is even, 4 is every token on four experts) | the busiest expert's load over the fair share (8 is every token)")
for arm in ["adamw-unbalanced", "muon-unbalanced"]:
    for seed in SEEDS:
        print(f"{LABEL[arm]:10} {seed} " + " ".join(f"{scalar(arm, seed, 'usage_perplexity', b)[-1]:6.2f}" for b in range(7)) + " | " + " ".join(f"{scalar(arm, seed, 'max_load_ratio', b)[-1]:5.2f}" for b in range(7)))

print("\nThe router's SVD entropy and row similarity, integrated, seed mean: the first mixture | the mean of the seven | the mean of the seven at the last reading")
for arm in ARMS:
    first = [seed_mean(arm, k) for k in ["router_svd_entropy", "router_row_similarity"]]
    every = [np.mean([seed_mean(arm, k, block=b) for b in range(7)]) for k in ["router_svd_entropy", "router_row_similarity"]]
    last = [np.mean([scalar(arm, s, k, b)[-1] for s in SEEDS for b in range(7)]) for k in ["router_svd_entropy", "router_row_similarity"]]
    print(f"{LABEL[arm]:10} {first[0]:.3f} {first[1]:.3f} | {every[0]:.3f} {every[1]:.3f} | {last[0]:.3f} {last[1]:.3f}")

print("\nOutliers, integrated after the warmup, seed mean: the largest residual, router logit, expert output and expert kurtosis; the first mixture | the largest over the seven mixtures' integrated values")
for arm in ARMS:
    parts = []
    for key in ["residual_max", "router_logit_max", "expert_output_max", "expert_kurtosis_max"]:
        parts.append(f"{seed_mean(arm, key):8.2f} {max(seed_mean(arm, key, block=b) for b in range(7)):8.2f}")
    print(f"{LABEL[arm]:10} " + " | ".join(parts))
