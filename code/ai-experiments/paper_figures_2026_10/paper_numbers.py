"""Every number of the arXiv paper's tables and text that comes from a block's endpoints, recomputed at full precision.

Reads ``runs/<block>/analysis/endpoints.csv`` of each block from the public Hub dataset ``marcoshernanz/tfg-runs`` (cached in ``outputs/hub/``), and prints the mean and the range over the seeds of every arm, and every paired difference the paper quotes with its range over the seeds and its signs. The paper rounds these once, so no number is rounded twice. Run from ``code/`` with ``.venv/bin/python ai-experiments/paper_figures_2026_10/paper_numbers.py > ai-experiments/paper_figures_2026_10/outputs/numbers.txt``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "outputs" / "hub"
HUB = "https://huggingface.co/datasets/marcoshernanz/tfg-runs/resolve/main"
BLOCKS = ("rq1", "rq3", "balancer", "balance_loss", "confirmation")
METRICS = (
    "val_loss",
    "bias_span_first",
    "bias_span",
    "router_input_similarity_first",
    "mean_score_span_first",
    "max_load_ratio_first_early",
    "router_update_common_share_first",
    "load_churn_warmup",
    "load_churn",
    "load_churn_early",
    "update_load_correlation",
    "update_load_correlation_spectral",
    "specialization",
    "usage_perplexity_final",
    "usage_perplexity_first_final",
)


def endpoints(block: str) -> pd.DataFrame:
    path = CACHE / block / "endpoints.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{HUB}/{block}/analysis/endpoints.csv", path)
    return pd.read_csv(path)


def arms(block: str, table: pd.DataFrame) -> None:
    print(f"== {block}: arm means [lowest, highest seed]")
    for arm, runs in table.groupby("arm", sort=False):
        seeds = sorted(runs["seed"].tolist())
        cells = [f"{name} {runs[name].mean():.5f} [{runs[name].min():.5f}, {runs[name].max():.5f}]" for name in METRICS if name in runs and runs[name].notna().all()]
        print(f"  {arm} seeds {seeds}")
        for cell in cells:
            print(f"      {cell}")


def pair(table: pd.DataFrame, mine: str, yours: str, names: tuple[str, ...] = METRICS) -> None:
    ours = table[table["arm"] == mine].set_index("seed")
    theirs = table[table["arm"] == yours].set_index("seed")
    seeds = ours.index.intersection(theirs.index)
    print(f"  {mine} - {yours}, seeds {list(seeds)}")
    for name in names:
        if name not in table or ours.loc[seeds, name].isna().any():
            continue
        diffs = (ours.loc[seeds, name] - theirs.loc[seeds, name]).to_numpy(dtype=float)
        signs = "".join("+" if d > 0 else "-" if d < 0 else "0" for d in diffs)
        print(f"      {name:34s} {diffs.mean():+.5f} [{diffs.min():+.5f}, {diffs.max():+.5f}] {signs}")


def main() -> None:
    tables = {block: endpoints(block) for block in BLOCKS}
    for block, table in tables.items():
        arms(block, table)
    print("== Paired differences")
    pair(tables["rq1"], "muon", "adamw")
    rq3 = tables["rq3"]
    for mine, yours in (
        ("muon", "adamw"),
        ("muon_except_router", "adamw"),
        ("muon_router", "adamw"),
        ("muon_except_router", "muon"),
        ("muon-muonlr0.5x", "adamw"),
        ("muon-muonlr0.5x", "muon"),
        ("muon_experts", "adamw"),
        ("muon_experts", "muon"),
        ("muon_attention_dense", "adamw"),
        ("muon_attention_dense", "muon"),
        ("muon_attention", "adamw"),
        ("muon_dense", "adamw"),
        ("muon-unbalanced", "adamw-unbalanced"),
    ):
        pair(rq3, mine, yours)
    balancer = tables["balancer"]
    for mine, yours in (("muon", "adamw"), ("muon-balancer4x", "adamw-balancer4x"), ("muon-balancer4x", "muon"), ("adamw-balancer4x", "adamw")):
        pair(balancer, mine, yours)
    # The balance-loss block holds only the runs with the loss; the runs without it are those of RQ3, on the same seeds.
    both = pd.concat([tables["balance_loss"], rq3[rq3["arm"].isin(["adamw", "muon", "muon_except_router"])]], ignore_index=True).drop_duplicates(subset=["arm", "seed"])
    for mine, yours in (
        ("muon-balanceloss0.0001", "adamw-balanceloss0.0001"),
        ("muon_except_router-balanceloss0.0001", "adamw-balanceloss0.0001"),
        ("muon-balanceloss0.0001", "muon"),
        ("muon_except_router-balanceloss0.0001", "muon_except_router"),
        ("adamw-balanceloss0.0001", "adamw"),
    ):
        pair(both, mine, yours)
    confirmation = tables["confirmation"]
    for mine, yours in (("muon", "adamw"), ("muon_except_router", "adamw"), ("muon_except_router", "muon")):
        pair(confirmation, mine, yours)


if __name__ == "__main__":
    main()
