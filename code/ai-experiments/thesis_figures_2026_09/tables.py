"""The values of the Results tables of the memoria, read from each block's endpoints.csv.

Prints, per block, the seed mean (and range) of every reading a table uses, and the paired differences the text quotes, with tfg.claims.compare's interval and signs. Run from code/: .venv/bin/python ai-experiments/thesis_figures_2026_09/tables.py > ai-experiments/thesis_figures_2026_09/outputs/tables.txt
"""

import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from tfg.claims import compare  # noqa: E402

RUNS = CODE / "runs"

READINGS = [
    "val_loss",
    "bias_span_first",
    "router_input_similarity_first",
    "mean_score_span_first",
    "max_load_ratio_first_early",
    "bias_span",
    "load_churn_warmup",
    "router_update_common_share_first",
    "specialization",
    "update_load_correlation",
    "usage_perplexity_final",
    "usage_perplexity_first_final",
    "collapsed_experts_final",
    "raw_max_load_ratio_first",
]


def table(block):
    frame = pd.read_csv(RUNS / block / "analysis" / "endpoints.csv")
    return frame


def arms(frame, names):
    for name in names:
        rows = frame[frame.arm == name].sort_values("seed")
        if rows.empty:
            print(f"  {name}: missing")
            continue
        parts = []
        for reading in READINGS:
            if reading not in rows or rows[reading].isna().all():
                continue
            values = rows[reading].to_numpy()
            parts.append(f"{reading}={values.mean():.4f} [{values.min():.4f}, {values.max():.4f}]")
        print(f"  {name} seeds {list(rows.seed)}")
        for part in parts:
            print(f"      {part}")


def differences(frame, pairs, readings):
    for mine, yours in pairs:
        result = compare(frame, mine, yours, names=readings)
        for _, row in result.iterrows():
            print(f"  {mine} - {yours} {row['endpoint']}: {row['difference']:+.4f} ({row['low']:+.4f} to {row['high']:+.4f}) {row.get('signs', '')}")


def main():
    rq3 = table("rq3")
    print("== RQ3 arms")
    arms(rq3, ["adamw", "muon_router", "muon", "muon_except_router", "muon_experts", "muon_attention", "muon_dense", "muon_attention_dense", "muon-muonlr0.5x", "adamw-unbalanced", "muon-unbalanced"])
    print("== RQ3 differences")
    differences(rq3, [("muon-unbalanced", "adamw-unbalanced")], ["usage_perplexity_final", "usage_perplexity_first_final", "collapsed_experts_final", "val_loss"])
    differences(rq3, [("muon_except_router", "muon")], ["val_loss", "load_churn_warmup", "max_load_ratio_first_early", "router_update_common_share_first", "bias_span_first"])
    differences(rq3, [("muon_router", "adamw"), ("muon_attention", "adamw"), ("muon_dense", "adamw"), ("muon_experts", "adamw"), ("muon_attention_dense", "adamw")], ["bias_span_first", "router_input_similarity_first", "bias_span"])
    differences(rq3, [("muon-muonlr0.5x", "muon"), ("muon-muonlr0.5x", "adamw")], ["val_loss", "bias_span", "bias_span_first", "max_load_ratio_first_early"])
    differences(rq3, [("muon", "adamw")], ["specialization", "val_loss", "bias_span"])

    print("== Balancer arms")
    balancer = table("balancer")
    arms(balancer, sorted(balancer.arm.unique()))
    differences(balancer, [("adamw-balancer4x", "adamw"), ("muon-balancer4x", "muon")], ["val_loss", "router_input_similarity_first", "bias_span_first"])

    print("== Balance-loss arms")
    loss = table("balance_loss")
    arms(loss, sorted(loss.arm.unique()))

    print("== Large model arms")
    large = table("confirmation")
    arms(large, ["adamw", "muon", "muon_except_router"])
    differences(large, [("muon_except_router", "muon")], ["val_loss", "max_load_ratio_first_early", "router_update_common_share_first"])
    differences(large, [("muon", "adamw")], ["specialization", "update_load_correlation", "val_loss"])


if __name__ == "__main__":
    main()
