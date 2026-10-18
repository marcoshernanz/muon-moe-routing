"""The first chain read again for the rethink of 2026-09-24: where in the network and when in training Muon's larger bias span sits, how it moves with the rate, and what the router's steps look like under the three stressed partitions.

Run from ``code/`` with ``PYTHONPATH=. uv run python ai-experiments/chain1_2026_09/rethink.py > ai-experiments/chain1_2026_09/outputs/rethink.txt``. Reads ``runs/tuning``, ``runs/rq1`` and ``runs/rq2`` (the chain's runs, from the Hub dataset ``marcoshernanz/tfg-runs``) and the T4 tier's arrays of ``update_norm_endpoint_2026_09``. Every reading is averaged over the logged steps after the warmup (500 steps) unless a window is named; "first" is the first mixture (block 1, right after the dense block 0), "rest" the mean of the other six.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfg.analysis import endpoints, routing_frame

WARMUP = 500
WINDOWS = [(500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 15259)]
READINGS = ["bias_span", "max_load_ratio", "load_churn", "router_logit_norm", "router_row_dispersion"]
T4 = Path("ai-experiments/update_norm_endpoint_2026_09/outputs/kaggle_t4")


def per_mixture(run: str, low: float = WARMUP, high: float = np.inf) -> pd.DataFrame:
    """The readings per mixture, averaged over the logged steps in ``(low, high]``."""
    frame = routing_frame(Path(run))
    inside = frame[(frame["step"] > low) & (frame["step"] <= high)]
    return inside.groupby("mixture")[READINGS].mean()


def first_and_rest(run: str) -> dict[str, float]:
    """The integrated readings over all mixtures, and the span of the first mixture and of the rest."""
    mixtures = per_mixture(run)
    return {
        "span": mixtures["bias_span"].mean(),
        "first": mixtures["bias_span"].iloc[0],
        "rest": mixtures["bias_span"].iloc[1:].mean(),
        "maxload": mixtures["max_load_ratio"].mean(),
        "churn": mixtures["load_churn"].mean(),
        "logit_norm": mixtures["router_logit_norm"].mean(),
        "row_disp": mixtures["router_row_dispersion"].mean(),
    }


def tuning() -> None:
    print("## The tuning sweeps: 0.5B tokens, cosine, seed 0\n")
    rows = []
    for optimizer in ("adamw", "muon", "momentum"):
        losses = json.loads(Path(f"runs/tuning/{optimizer}/sweep.json").read_text())["val_loss"]
        for rate, loss in sorted(losses.items(), key=lambda item: float(item[0])):
            rows.append({"optimizer": optimizer, "rate": rate, "val_loss": loss, **first_and_rest(f"runs/tuning/{optimizer}/lr{rate}")})
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")


def gate1_windows() -> None:
    print("## Gate 1 (1B tokens, cosine): the span per mixture by window of training, mean of seeds 0 to 2\n")
    for arm in ("adamw", "muon"):
        for low, high in WINDOWS:
            spans = np.mean([per_mixture(f"runs/rq1/mechanism-{arm}-s{seed}", low, high)["bias_span"].to_numpy() for seed in range(3)], axis=0)
            print(f"{arm:6s} {low:5d}-{high:5d}  " + " ".join(f"{v:.3f}" for v in spans) + f"  | first {spans[0]:.3f}  rest {spans[1:].mean():.3f}")
    print()


def gate1_seeds() -> None:
    print("## Gate 1 per seed: first mixture and the rest, integrated\n")
    rows = [{"arm": arm, "seed": seed, **first_and_rest(f"runs/rq1/mechanism-{arm}-s{seed}")} for arm in ("adamw", "muon") for seed in range(3)]
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")


def anisotropy() -> None:
    print("## Gate 1: the annex's mean hidden-state cosine within a domain (the router's input at the last checkpoint) against the span, per mixture, mean of seeds\n")
    for arm in ("adamw", "muon"):
        within, spans = [], []
        for seed in range(3):
            run = f"runs/rq1/mechanism-{arm}-s{seed}"
            annex = json.loads(Path(run, "annex.json").read_text())["mixtures"]
            within.append([m["hidden_cosine_within"] for m in annex])
            spans.append(per_mixture(run, 8000)["bias_span"].to_numpy())
        print(f"{arm:6s} cosine within  " + " ".join(f"{v:.3f}" for v in np.mean(within, axis=0)))
        print(f"{arm:6s} span, last half " + " ".join(f"{v:.3f}" for v in np.mean(spans, axis=0)))
    print()


def stress() -> None:
    print("## The stress grid (0.5B tokens, constant rate after the warmup): mean of seeds 0 to 2\n")
    rows = []
    for partition in ("adamw", "muon", "muon_except_router"):
        for label, multiplier in (("", 1), ("-lr2x", 2), ("-lr4x", 4), ("-lr8x", 8), ("-lr16x", 16)):
            runs = [f"runs/rq2/mechanism-{partition}{label}-s{seed}" for seed in range(3)]
            readings = pd.DataFrame([first_and_rest(run) for run in runs]).mean()
            ends = pd.DataFrame([endpoints(Path(run)) for run in runs])
            extra = ends[["load_churn_warmup", "load_churn_early", "update_norm_router", "angular_step_router", "weight_norm_router"]].mean()
            rows.append({"partition": partition, "multiplier": multiplier, **readings, **extra, "span_low": min(first_and_rest(run)["span"] for run in runs), "span_high": max(first_and_rest(run)["span"] for run in runs)})
    print(pd.DataFrame(rows).round(4).to_string(index=False), "\n")


def t4() -> None:
    print("## The T4 tier (section 11 of the evidence log): 16 experts, 5 mixtures, batch 16, 1,500 steps, both optimizers at 1e-3; the span per mixture after step 100 and after step 1,000\n")
    for path in sorted(glob.glob(str(T4 / "*.npz"))):
        arrays = np.load(path)
        steps, span = arrays["steps"], arrays["bias_span"]  # [T], [T, M]
        if span.max() == 0:  # a run without the balancer
            continue
        after, late = span[steps > 100].mean(0), span[steps > 1000].mean(0)
        print(f"{Path(path).stem:24s} " + " ".join(f"{v:.3f}" for v in after) + "  | late " + " ".join(f"{v:.3f}" for v in late))
    print()


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    tuning()
    gate1_seeds()
    gate1_windows()
    anisotropy()
    stress()
    t4()
