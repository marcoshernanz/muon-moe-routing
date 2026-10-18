"""The draft of the plan's figure 13 (exploratory, Amendment 4): the first mixture's selection margins against the balancer's step.

Six panels. (A) Every balanced run of the program: the first mixture's mean selection margin over the bias's step u against its load churn, both over every step after the warmup, (a), (b) and (c) coloured and every other arm grey, the mechanism tier as circles and the confirmation tier as squares. (B) The confirmation tier per mixture: the mean margin over u for (a), (b) and (c), a line per seed. (C) The confirmation tier's first mixture over training: the busiest expert's load over its share, (a), (b) and (c), each seed smoothed over 50 readings. (D) The scale probe's first 3,000 steps at seed 0: the busiest expert of the first mixture for (b) and (c) as run and with the router input centred, against (a), and (b) with the bias four times faster, each run smoothed over 10 readings. (E) The same runs: the share of tokens whose selection margin is under two of the run's bias steps, so that one step of the balancer can change their expert. (F) The same runs: the share of chosen experts the scores alone would not have chosen. Run from code/: uv run python ai-experiments/scale_probe_2026_09/figure.py (writes outputs/knife-edge.png and .pdf)
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
BLOCKS = ("rq1", "rq2", "rq3", "balancer", "confirmation")
U = 1e-3
COLOURS = {"adamw": "#1f77b4", "muon": "#d62728", "muon_except_router": "#ff7f0e"}
NAMES = {"adamw": "(a) all AdamW", "muon": "(b) all Muon", "muon_except_router": "(c) router on AdamW"}


def rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in open(run / "routing.jsonl")]


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(values, np.ones(width) / width, mode="valid")


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    ax = axes[0, 0]
    for block in BLOCKS:
        for run in sorted(p.parent for p in Path("runs", block).glob("*/routing.jsonl")):
            settings = json.loads((run / "settings.json").read_text())
            if not settings.get("balanced", True):
                continue
            arm = run.name.rsplit("-s", 1)[0].split("-", 1)[1]
            after = [r["mixtures"][0] for r in rows(run) if r["step"] > settings["warmup_steps"]]
            margin = np.mean([m["margin_mean"] for m in after]) / (U * settings.get("balancer_multiplier", 1.0))
            churn = np.mean([m["load_churn"] for m in after])
            colour = COLOURS.get(arm, "#bbbbbb")
            marker = "s" if settings["tier"] == "confirmation" else "o"
            ax.scatter(margin, churn, c=colour, marker=marker, s=40 if colour != "#bbbbbb" else 14, edgecolors="black" if marker == "s" else "none", linewidths=0.6, zorder=3 if colour != "#bbbbbb" else 2)
    ax.axvline(1, color="black", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("first mixture's mean selection margin / bias step u")
    ax.set_ylabel("first mixture's load churn")
    ax.set_title("A. Every balanced run: circles 55M, squares 108M; grey, other arms")

    ax = axes[0, 1]
    for arm, colour in COLOURS.items():
        for seed in (0, 1, 2):
            records = [r for r in rows(Path("runs/confirmation") / f"confirmation-{arm}-s{seed}") if r["step"] > 500]
            count = len(records[0]["mixtures"])
            margins = [np.mean([r["mixtures"][m]["margin_mean"] for r in records]) / U for m in range(count)]
            ax.plot(range(1, count + 1), margins, color=colour, marker="o", ms=3, label=NAMES[arm] if seed == 0 else None)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("mixture (1 is the first, after the dense block)")
    ax.set_ylabel("mean selection margin / u")
    ax.set_title("B. 108M, 3B tokens: the margins per mixture, a line per seed")
    ax.legend()

    ax = axes[1, 0]
    for arm, colour in COLOURS.items():
        for seed in (0, 1, 2):
            records = rows(Path("runs/confirmation") / f"confirmation-{arm}-s{seed}")
            steps = np.array([r["step"] for r in records])
            busiest = np.array([r["mixtures"][0]["max_load_ratio"] for r in records])
            ax.plot(smooth(steps, 50), smooth(busiest, 50), color=colour, lw=0.8, label=NAMES[arm] if seed == 0 else None)
    ax.set_xlabel("step")
    ax.set_ylabel("first mixture's busiest expert / its share (8 is every token)")
    ax.set_title("C. 108M, 3B tokens: the overload over training")
    ax.legend()

    probes = sorted((HERE / "outputs" / "runs").glob("confirmation-*-s0"))
    panels = [  # axis, reading, label, title
        (axes[1, 1], None, "first mixture's busiest expert / its share", "D. Probe, seed 0: as run solid, centred dotted, bias 4× dash-dot"),
        (axes[0, 2], "flippable", "tokens within two bias steps of a switch", "E. The same runs: tokens one bias step can move"),
        (axes[1, 2], "bias_decided", "chosen experts the scores alone would not choose", "F. The same runs: the choices the bias makes"),
    ]
    for ax, reading, ylabel, title in panels:
        for run in probes:
            info = json.loads((run / "probe.json").read_text())
            if reading is None:
                records = rows(run)
                steps = np.array([r["step"] for r in records])
                values = np.array([r["mixtures"][0]["max_load_ratio"] for r in records])
            else:
                readings = np.load(run / "readings.npz")
                steps = readings["steps"]
                values = readings["summary"][:, 0, list(readings["summary_names"]).index(reading)]
            style = "-." if info["balancer_multiplier"] != 1 else ":" if info["centre"] else "-"
            label = NAMES[info["partition"]] + (", centred" if info["centre"] else "") + (", bias 4×" if info["balancer_multiplier"] != 1 else "")
            ax.plot(smooth(steps, 10), smooth(values, 10), color=COLOURS[info["partition"]], ls=style, lw=1.2, label=label)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if probes and reading is None:
            ax.legend(fontsize=8)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / "outputs" / f"knife-edge.{suffix}", dpi=150)


if __name__ == "__main__":
    main()
