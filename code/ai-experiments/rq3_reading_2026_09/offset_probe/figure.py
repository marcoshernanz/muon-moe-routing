"""The first 3,000 steps of the first mixture under the backbone on Muon, as the chain ran it and with each of the probe's changes, against all-AdamW: the busiest expert's load over its fair share, the bias span and the score offset, seed means over seeds 3 to 5, smoothed over 50 steps. Left, the router input centred and the bias four times faster, against (b) and (a); right, the router on AdamW against the router on Muon, both with Muon at 2e-3. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/offset_probe/figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROBE = Path("ai-experiments/rq3_reading_2026_09/offset_probe/outputs/mechanism/runs")
CHAIN = Path("runs/rq3")
OUT = Path("ai-experiments/rq3_reading_2026_09/offset_probe/outputs/probe.png")
STOP, SEEDS, SMOOTH = 3000, (3, 4, 5), 5  # SMOOTH logged steps of 10
KEYS = [("max_load_ratio", "busiest expert / fair share"), ("bias_span", "bias span"), ("mean_score_span", "score offset")]
PANELS = [
    [("chain", "muon", "(b) Muon, as run", "C3"), ("probe", "muon-centred", "(b), router input centred", "C2"), ("probe", "muon-balancer4x", "(b), bias 4x faster", "C1"), ("chain", "adamw", "(a) AdamW", "C0")],
    [("probe", "muon_except_router-muonlr0.5x", "(c) router on AdamW, Muon 2e-3", "C4"), ("chain", "muon-muonlr0.5x", "(b) Muon 2e-3", "C5"), ("chain", "adamw", "(a) AdamW", "C0")],
]


def series(source: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    runs = []
    for seed in SEEDS:
        path = (PROBE if source == "probe" else CHAIN) / f"mechanism-{arm}-s{seed}" / "routing.jsonl"
        at, rows = [], []
        with path.open() as log:
            for line in log:
                record = json.loads(line)
                if record["step"] > STOP:
                    break
                at.append(record["step"])
                rows.append([record["mixtures"][0][k] for k, _ in KEYS])
        runs.append(np.array(rows))
    mean = np.mean(runs, axis=0)  # [T, K]
    kernel = np.ones(SMOOTH) / SMOOTH
    smooth = np.stack([np.convolve(mean[:, k], kernel, mode="valid") for k in range(len(KEYS))], 1)
    return np.array(at)[SMOOTH // 2 : SMOOTH // 2 + len(smooth)], smooth


fig, axes = plt.subplots(len(KEYS), 2, figsize=(11, 8.5), sharex=True)
for column, panel in enumerate(PANELS):
    for source, arm, label, colour in panel:
        at, values = series(source, arm)
        for row in range(len(KEYS)):
            axes[row, column].plot(at, values[:, row], color=colour, label=label, lw=1.4)
    for row, (_, name) in enumerate(KEYS):
        axes[row, column].set_ylabel(name)
        axes[row, column].axvline(500, color="0.6", lw=0.8, ls=":")
        axes[row, column].axvspan(500, 1262, color="0.92", zorder=0)
    axes[0, column].legend(fontsize=8, frameon=False)
    axes[-1, column].set_xlabel("step")
axes[0, 0].set_title("Centring the router input and a faster bias", fontsize=10)
axes[0, 1].set_title("The router's optimizer at Muon's better rate", fontsize=10)
fig.suptitle("First mixture, first 3,000 steps, seeds 3 to 5 (the warmup ends at the dotted line; the plan's early window shaded)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(OUT)
