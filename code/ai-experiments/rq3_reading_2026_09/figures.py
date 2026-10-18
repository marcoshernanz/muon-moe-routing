"""The reading's three figures.

`offset.png`: every balanced run's first-mixture integrated span against its score spread (`mean_score_span`) and against its router-input cosine, log scales. `early.png`: the first mixture over the first 3,000 steps for (a), (b), (c) and Muon at half its rate, seed mean, smoothed over 50 steps: the cosine, the score spread, the span, and the busiest expert's load over the fair share. `balancer.png`: the same load reading in the stress grid for Muon at four times its rate with the balancer at its rate and at four times it, beside AdamW. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reading import BALANCED, LABEL, SEEDS, first_mixture, scalar, steps, window

OUT = "ai-experiments/rq3_reading_2026_09/outputs"
COLOURS = {"adamw": "#1f77b4", "muon_router": "#9ecae1", "muon_dense": "#bcbd22", "muon_attention": "#ff7f0e", "muon_attention_dense": "#e377c2", "muon_experts": "#8c564b", "muon_except_router": "#2ca02c", "muon": "#d62728", "muon-muonlr0.5x": "#9467bd"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def smooth(at: np.ndarray, values: np.ndarray, width: int = 5) -> tuple[np.ndarray, np.ndarray]:  # five readings, 50 steps, no padding at the ends
    return at[width // 2 : len(at) - width // 2], np.convolve(values, np.ones(width) / width, mode="valid")


figure, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for arm in BALANCED:
    span = [window(arm, s, "bias_span") for s in SEEDS]
    for ax, key in zip(axes, ["mean_score_span", "router_input_similarity"]):
        ax.scatter([window(arm, s, key) for s in SEEDS], span, color=COLOURS[arm], label=LABEL[arm], s=22)
for ax, name, ticks in zip(axes, ["score spread across experts (mean_score_span)", "router-input cosine"], [[0.1, 0.2, 0.3], [0.03, 0.1, 0.4]]):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_yticks([0.1, 0.2, 0.3, 0.5], ["0.1", "0.2", "0.3", "0.5"])
    ax.set_yticks([], minor=True)
    ax.set_xlabel(f"{name}, integrated")
axes[0].set_ylabel("first mixture's span, integrated")
axes[0].plot([0.07, 0.35], [0.07, 0.35], color="grey", lw=0.8, ls="--")
axes[1].legend(fontsize=7, frameon=False, loc="upper left")
figure.tight_layout()
figure.savefig(f"{OUT}/offset.png", dpi=180)

ARMS = ["adamw", "muon", "muon_except_router", "muon-muonlr0.5x"]
figure, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True)
for arm in ARMS:
    at = steps(arm, SEEDS[0])
    keep = at <= 3000
    for ax, key in zip(axes, ["router_input_similarity", "mean_score_span", "bias_span", "max_load_ratio"]):
        mean = np.mean([scalar(arm, s, key)[keep] for s in SEEDS], axis=0)
        ax.plot(*smooth(at[keep], mean), color=COLOURS[arm], label=LABEL[arm], lw=1.2)
for ax, name in zip(axes, ["router-input cosine", "score spread across experts", "span of the bias", "busiest expert's load / fair share"]):
    ax.set_title(name)
    ax.axvline(500, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("step")
axes[3].axhline(1, color="grey", lw=0.6)
axes[0].legend(fontsize=7, frameon=False)
figure.tight_layout()
figure.savefig(f"{OUT}/early.png", dpi=180)

figure, ax = plt.subplots(figsize=(4.5, 3))
for name, colour, label in [("adamw-lr4x", "#1f77b4", "AdamW at 4x"), ("muon-lr4x", "#d62728", "Muon at 4x"), ("muon-lr4x-balancer4x", "#ff9896", "Muon at 4x, balancer at 4x")]:
    curves = []
    for seed in (0, 1, 2):
        at, values = first_mixture(f"runs/rq2/mechanism-{name}-s{seed}", ["max_load_ratio"])
        curves.append(values[at <= 3000, 0])
    ax.plot(*smooth(at[at <= 3000], np.mean(curves, axis=0)), color=colour, label=label, lw=1.2)
ax.axhline(1, color="grey", lw=0.6)
ax.set_xlabel("step")
ax.set_ylabel("busiest expert's load / fair share")
ax.set_title("stress grid, first mixture, seeds 0 to 2")
ax.legend(fontsize=7, frameon=False)
figure.tight_layout()
figure.savefig(f"{OUT}/balancer.png", dpi=180)
