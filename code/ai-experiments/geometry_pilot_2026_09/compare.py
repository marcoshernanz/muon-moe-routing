"""The pilot's two readings, from the runs brought home to ``runs/pilot`` (laptop, from ~/code): first, whether the two reruns of the first chain's tuning runs, (a) at 2e-3 and (b) at 4e-3, match them bit for bit on every number both logged, which says the new readings leave training alone on the card; then, per run, the endpoints of the analysis the second registration reads: every reading integrated after the warmup over all the mixtures and in the first alone (``_first``), over the warmup and over the early window, beside the first chain's Muon run at half its rate, which is Muon at AdamW's rate. The endpoints are read twice: over each whole run, and over steps up to 3,000, where (d) was stopped, so that every arm is read on the same window (the last checkpoint's readings are the same in both). Writes outputs/bit-for-bit.txt, outputs/pilot.csv and outputs/pilot-3000.csv, and prints them."""

import json
import sys
from pathlib import Path

import pandas as pd
from tfg.analysis import endpoints

PILOT = Path("runs/pilot")
OUT = Path("ai-experiments/geometry_pilot_2026_09/outputs")
RERUNS = {  # the pilot's rerun and the first chain's run it repeats
    "mechanism-adamw-s0": Path("runs/tuning/adamw/lr0.002"),
    "mechanism-muon-s0": Path("runs/tuning/muon/lr0.004"),
}
COMMON = 3000  # the window every arm has, (d) included
CONTEXT = {"tuning-muon-lr0.002": Path("runs/tuning/muon/lr0.002")}  # Muon at half its rate
READINGS = [
    "val_loss",
    "bias_span",
    "bias_span_first",
    "bias_span_first_early",
    "bias_span_first_final",
    "router_input_similarity",
    "router_input_similarity_first",
    "mean_score_span_first",
    "raw_max_load_ratio_first",
    "max_load_ratio_first",
    "load_churn",
    "load_churn_warmup",
    "load_churn_early",
    "load_churn_first",
    "load_churn_first_warmup",
    "router_update_common_share",
    "router_update_common_share_warmup",
    "router_update_common_share_first",
    "router_row_dispersion_early",
    "router_logit_norm_first",
    "update_load_correlation",
]


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def first_difference(new: Path, old: Path) -> str:
    """Every number both runs logged, step by step: the losses, the norms and validation losses, and every routing reading the first chain had; the first step where they differ, or that they never do."""
    report = []
    for file in ("metrics.jsonl", "routing.jsonl"):
        a, b = lines(new / file), lines(old / file)
        steps = min(len(a), len(b))
        for x, y in zip(a[:steps], b[:steps]):
            if file == "routing.jsonl":
                x = {**x, "mixtures": [{k: m[k] for k in y["mixtures"][0]} for m in x["mixtures"]]}
            x = {k: v for k, v in x.items() if k in y and k != "tokens_per_s"}
            y = {k: v for k, v in y.items() if k != "tokens_per_s"}
            if x != y:
                keys = [k for k in y if x.get(k) != y[k]]
                report.append(f"{file}: first difference at step {y['step']} in {keys}")
                break
        else:
            report.append(
                f"{file}: {steps} lines, the same bit for bit (lengths {len(a)} and {len(b)})"
            )
    return "\n".join(report)


def main() -> None:
    runs = {d.name: d for d in sorted(PILOT.iterdir()) if (d / "metrics.jsonl").exists()}
    report = [
        f"{name} against {RERUNS[name]}:\n{first_difference(runs[name], RERUNS[name])}"
        for name in RERUNS
        if name in runs
    ]
    (OUT / "bit-for-bit.txt").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print("\n\n".join(report))
    for until, file in ((None, "pilot.csv"), (COMMON, f"pilot-{COMMON}.csv")):
        rows = {}
        for name, run_dir in {**runs, **CONTEXT}.items():
            ends = endpoints(run_dir, until=until)
            rows[name] = {r: ends.get(r) for r in READINGS}
        table = pd.DataFrame(rows).T
        table.to_csv(OUT / file)
        print(f"\n{file}")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(table.astype(float).round(4).T.to_string())


if __name__ == "__main__":
    sys.exit(main())
