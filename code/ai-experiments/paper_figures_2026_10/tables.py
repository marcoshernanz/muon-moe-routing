"""The rows of the arXiv paper's tables (paper/main.tex), as LaTeX, from the endpoints of each block at full precision.

Reads the endpoints that ``paper_numbers.py`` caches in ``outputs/hub/`` (downloading them from the public Hub dataset ``marcoshernanz/tfg-runs`` if they are missing) and prints every row of the RQ1, RQ3, rise and router tables, each value rounded once. The first-layer span and early overload of RQ1 are not in the RQ1 endpoints; their per-seed values are read from the routing logs by ``tfg.analysis.endpoints`` (the span as ``../thesis_figures_2026_09/extract.py`` reads it, which prints it to four decimals in ``../thesis_figures_2026_09/outputs/numbers.txt``), kept here to six decimals. Run from ``code/`` with ``.venv/bin/python ai-experiments/paper_figures_2026_10/tables.py > ai-experiments/paper_figures_2026_10/outputs/tables.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_numbers import endpoints

# The span of the first MoE layer in RQ1 at seeds 0, 1 and 2, from the routing logs to six decimals, so that each printed value is rounded once.
RQ1_FIRST = {"adamw": np.array([0.072034, 0.105218, 0.075048]), "muon": np.array([0.222948, 0.190429, 0.167731])}
# The early overload of the first MoE layer in RQ1 (max_load_ratio_first_early, steps 501 to 1,262) at seeds 0, 1 and 2, from the routing logs to six decimals.
RQ1_FIRST_EARLY_OVERLOAD = {"adamw": np.array([1.129725, 1.121041, 1.143881]), "muon": np.array([4.525909, 5.698127, 2.611529])}


def values(table: pd.DataFrame, arm: str, name: str) -> pd.Series:
    return table[table["arm"] == arm].set_index("seed")[name].sort_index()


def differences(table: pd.DataFrame, mine: str, yours: str, name: str) -> np.ndarray:
    ours, theirs = values(table, mine, name), values(table, yours, name)
    seeds = ours.index.intersection(theirs.index)
    return (ours.loc[seeds] - theirs.loc[seeds]).to_numpy(dtype=float)


def signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def cell(diffs: np.ndarray, digits: int) -> str:
    return f"${signed(diffs.mean(), digits)}$ \\seedrange{{{signed(diffs.min(), digits)}}}{{{signed(diffs.max(), digits)}}}"


def rq1() -> None:
    table = endpoints("rq1")
    rows = (
        ("Validation loss", "val_loss", 3),
        ("Span", "bias_span", 3),
        ("Span, first MoE layer", None, 3),
        ("Update--load correlation", "update_load_correlation", 3),
        ("Update--load correlation, spectral norm", "update_load_correlation_spectral", 3),
        ("Load churn", "load_churn", 3),
        ("Load churn, early", "load_churn_early", 3),
        ("Warmup churn", "load_churn_warmup", 3),
        ("Specialization", "specialization", 3),
    )
    print("% RQ1: metric & AdamW & Muon & difference [range]")
    for label, name, digits in rows:
        if name is None:
            adamw, muon = RQ1_FIRST["adamw"], RQ1_FIRST["muon"]
        else:
            adamw, muon = values(table, "adamw", name).to_numpy(), values(table, "muon", name).to_numpy()
        print(f"{label} & {adamw.mean():.{digits}f} & {muon.mean():.{digits}f} & {cell(muon - adamw, digits)} \\\\")


def rq3() -> None:
    table = endpoints("rq3")
    arms = (
        ("AdamW", "adamw"),
        ("Router on Muon", "muon_router"),
        ("Muon, router on AdamW", "muon_except_router"),
        ("Muon", "muon"),
        ("Muon, half rate", "muon-muonlr0.5x"),
        ("Experts on Muon", "muon_experts"),
        ("Attention on Muon", "muon_attention"),
        ("Dense FFN on Muon", "muon_dense"),
        ("Attention, dense FFN on Muon", "muon_attention_dense"),
    )
    columns = (
        ("val_loss", 3),
        ("bias_span_first", 3),
        ("router_input_similarity_first", 3),
        ("mean_score_span_first", 3),
        ("max_load_ratio_first_early", 2),
        ("router_update_common_share_first", 3),
        ("bias_span", 3),
        ("load_churn_warmup", 3),
    )
    print("% RQ3: arm & loss & span & cosine & offset & early overload & common share & span all & warmup churn")
    for label, arm in arms:
        cells = [f"{values(table, arm, name).mean():.{digits}f}" for name, digits in columns]
        print(f"{label} & " + " & ".join(cells) + " \\\\")


def rise() -> None:
    ablation = endpoints("rq3")
    balancer = endpoints("balancer")
    loss = endpoints("balance_loss")  # its analysis holds the runs of RQ3 without the loss too, on the same seeds
    large = endpoints("confirmation")
    rows = (
        ("RQ3", "Muon", ablation, "muon", "adamw"),
        ("RQ3", "Muon, router on AdamW", ablation, "muon_except_router", "adamw"),
        ("RQ3", "Muon, half rate", ablation, "muon-muonlr0.5x", "adamw"),
        ("Bias speed", "Muon, usual bias", balancer, "muon", "adamw"),
        ("Bias speed", "Muon, fast bias", balancer, "muon-balancer4x", "adamw-balancer4x"),
        ("Balance loss", "Muon", loss, "muon-balanceloss0.0001", "adamw-balanceloss0.0001"),
        ("Balance loss", "Muon, router on AdamW", loss, "muon_except_router-balanceloss0.0001", "adamw-balanceloss0.0001"),
        ("Large model", "Muon", large, "muon", "adamw"),
        ("Large model", "Muon, router on AdamW", large, "muon_except_router", "adamw"),
    )
    columns = (("bias_span_first", 3), ("max_load_ratio_first_early", 2), ("router_input_similarity_first", 3), ("val_loss", 3))
    print("% Rise: experiment & arm & span & early overload & cosine & loss, each arm minus AdamW with the same balancer")
    first = RQ1_FIRST["muon"] - RQ1_FIRST["adamw"]
    overload = RQ1_FIRST_EARLY_OVERLOAD["muon"] - RQ1_FIRST_EARLY_OVERLOAD["adamw"]
    main_comparison = endpoints("rq1")
    print(f"RQ1 & Muon & {cell(first, 3)} & {cell(overload, 2)} & -- & {cell(differences(main_comparison, 'muon', 'adamw', 'val_loss'), 3)} \\\\")
    for experiment, label, table, mine, yours in rows:
        cells = [cell(differences(table, mine, yours, name), digits) for name, digits in columns]
        print(f"{experiment} & {label} & " + " & ".join(cells) + " \\\\")


def router() -> None:
    ablation = endpoints("rq3")
    loss = endpoints("balance_loss")  # its analysis holds the runs of RQ3 without the loss too, on the same seeds
    large = endpoints("confirmation")
    rows = (
        ("Small model", ablation, "muon_except_router", "muon"),
        ("Small model, balance loss", loss, "muon_except_router-balanceloss0.0001", "muon-balanceloss0.0001"),
        ("Large model", large, "muon_except_router", "muon"),
    )
    columns = (
        ("load_churn_warmup", 3),
        ("max_load_ratio_first_early", 2),
        ("router_update_common_share_first", 3),
        ("bias_span_first", 3),
        ("val_loss", 3),
    )
    print("% Router: setting & warmup churn & early overload & common share & span & loss, router on AdamW minus Muon")
    for label, table, mine, yours in rows:
        cells = [cell(differences(table, mine, yours, name), digits) for name, digits in columns]
        print(f"{label} & " + " & ".join(cells) + " \\\\")
    print("% Signs per seed of the same differences")
    for label, table, mine, yours in rows:
        signs = ["".join("+" if d > 0 else "-" for d in differences(table, mine, yours, name)) for name, _ in columns]
        print(f"% {label}: " + " ".join(signs))


def main() -> None:
    rq1()
    rq3()
    rise()
    router()


if __name__ == "__main__":
    main()
