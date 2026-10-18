"""The first chain's readings seed by seed on the arms the second registration contrasts, to see which of its predictions three fresh seeds could decide: Gate 1's two arms (cosine, 1B tokens) and the stress grid's three partitions at 1x (constant after the warmup, 0.5B tokens), each reading integrated after the warmup, over the warmup, and over the early window, in all the mixtures and in the first alone.

Run from ``code/`` with ``PYTHONPATH=. uv run python ai-experiments/chain1_2026_09/per_seed.py > ai-experiments/chain1_2026_09/outputs/per-seed.txt``. Reads ``runs/rq1`` and ``runs/rq2`` (from the Hub dataset ``marcoshernanz/tfg-runs``).
"""

from pathlib import Path

import pandas as pd
from tfg.analysis import endpoints

READINGS = [
    "val_loss",
    "bias_span",
    "bias_span_first",
    "bias_span_first_early",
    "load_churn",
    "load_churn_early",
    "load_churn_warmup",
    "load_churn_first",
    "load_churn_first_warmup",
    "router_row_dispersion_early",
    "router_logit_norm_first",
    "update_load_correlation",
]
BLOCKS = {
    "Gate 1 (cosine, 1B tokens)": ("runs/rq1", ("adamw", "muon")),
    "stress grid at 1x (constant, 0.5B tokens)": (
        "runs/rq2",
        ("adamw", "muon", "muon_except_router"),
    ),
}


def main() -> None:
    for title, (block, arms) in BLOCKS.items():
        rows = []
        for arm in arms:
            for seed in range(3):
                ends = endpoints(Path(block, f"mechanism-{arm}-s{seed}"))
                rows.append({"arm": arm, "seed": seed, **{r: ends.get(r) for r in READINGS}})
        table = pd.DataFrame(rows).set_index(["arm", "seed"]).astype(float)
        print(f"## {title}\n")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(table.round(4).T.to_string(), "\n")


if __name__ == "__main__":
    main()
