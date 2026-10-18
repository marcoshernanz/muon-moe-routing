"""The data behind the figures and tables of the memoria's Results chapter, read from the runs in code/runs.

Writes one CSV per figure to thesis/tex/figures/data/, which the pgfplots figures under thesis/tex/figures/ read, and prints every number the chapter's tables and text quote to outputs/numbers.txt. The endpoints and the paired comparisons are those of tfg.analysis and tfg.claims (the analysis tables each block wrote under runs/<block>/analysis/), and the curves are read from the routing logs with tfg.analysis's readers. Run from code/ with ``.venv/bin/python ai-experiments/thesis_figures_2026_09/extract.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))

from tfg import analysis, claims  # noqa: E402

RUNS = CODE / "runs"
DATA = CODE.parent / "thesis" / "tex" / "figures" / "data"
OUT = HERE / "outputs"
WARMUP = 500
LINES: list[str] = []


def say(text: str = "") -> None:
    LINES.append(text)
    print(text)


def table(block: str, matched: bool = False) -> pd.DataFrame:
    name = "endpoints_matched.csv" if matched else "endpoints.csv"
    return pd.read_csv(RUNS / block / "analysis" / name)


def pair(frame: pd.DataFrame, mine: str, yours: str, names: list[str]) -> pd.DataFrame:
    """``mine`` minus ``yours`` on ``names``, paired by seed, with tfg.claims' interval and signs."""
    return claims.compare(frame, mine, yours, names=names)


def show(label: str, frame: pd.DataFrame) -> None:
    for row in frame.to_dict("records"):
        say(
            f"{label:34s} {row['endpoint']:36s} {row['difference']:+.4f} "
            f"({row['low']:+.4f} to {row['high']:+.4f}) effect {row['effect']:.1f} {row['signs']}"
        )


def per_seed(frame: pd.DataFrame, arm: str, name: str) -> str:
    runs = frame[frame["arm"] == arm].sort_values("seed")
    return ", ".join(f"{v:.4f}" for v in runs[name])


def binned(series: pd.Series, width: int) -> pd.Series:
    """Means over consecutive bins of ``width`` steps, indexed by the bin's centre."""
    bins = ((series.index - 1) // width) * width + width / 2
    return series.groupby(bins).mean()


def curves(runs: list[Path], name: str, *, first: bool, width: int, until: int | None = None) -> pd.DataFrame:
    """One reading against the step over ``runs`` (seeds of one arm): mean, lowest and highest seed, per bin of ``width`` steps."""
    series = []
    for run in runs:
        readings = analysis.routing_frame(run)
        frame = analysis.first_mixture(readings) if first else analysis.per_step(readings, pd.DataFrame())
        values = frame[name]
        if until is not None:
            values = values[values.index <= until]
        series.append(binned(values, width) if width > 10 else values)
    stacked = pd.concat(series, axis=1)
    return pd.DataFrame({"mean": stacked.mean(axis=1), "low": stacked.min(axis=1), "high": stacked.max(axis=1)})


def rest_of_mixtures(run: Path, name: str) -> pd.Series:
    readings = analysis.routing_frame(run)
    return readings[readings["mixture"] > 0].drop(columns="mixture").groupby("step")[name].mean()


def per_mixture(run: Path, name: str) -> pd.Series:
    """The reading integrated after the warmup, per mixture."""
    readings = analysis.routing_frame(run)
    after = readings[readings["step"] > WARMUP]
    return after.groupby("mixture")[name].mean()


def rq1() -> None:
    say("== RQ1: (a) against (b), seeds 0 to 2, 1B tokens")
    tokens = table("rq1")
    names = [
        "val_loss",
        "bias_span",
        "update_load_correlation",
        "update_load_correlation_spectral",
        "load_churn",
        "load_churn_early",
        "load_churn_warmup",
        "specialization",
        "usage_perplexity",
        "max_load_ratio",
    ]
    names = [n for n in names if n in tokens.columns]
    for n in names:
        say(f"  {n:34s} AdamW {per_seed(tokens, 'adamw', n)} | Muon {per_seed(tokens, 'muon', n)}")
    show("muon - adamw, equal tokens", pair(tokens, "muon", "adamw", names))

    # The span per mixture, and the first against the other six.
    rows = []
    for arm in ("adamw", "muon"):
        for seed in (0, 1, 2):
            run = RUNS / "rq1" / f"mechanism-{arm}-s{seed}"
            spans = per_mixture(run, "bias_span")
            for mixture, value in spans.items():
                rows.append({"arm": arm, "seed": seed, "layer": mixture + 1, "span": value})
            say(f"  per layer {arm} s{seed}: " + ", ".join(f"{v:.3f}" for v in spans))
            say(f"    first {spans.iloc[0]:.4f}, other six {spans.iloc[1:].mean():.4f}")
    layers = pd.DataFrame(rows)
    for arm in ("adamw", "muon"):
        wide = layers[layers["arm"] == arm].pivot(index="layer", columns="seed", values="span")
        out = pd.DataFrame({"layer": wide.index, "mean": wide.mean(axis=1), "low": wide.min(axis=1), "high": wide.max(axis=1)})
        out.to_csv(DATA / f"rq1-layers-{arm}.csv", index=False, float_format="%.5f")
    firsts = layers[layers["layer"] == 1].pivot(index="seed", columns="arm", values="span")
    rest = layers[layers["layer"] > 1].groupby(["seed", "arm"])["span"].mean().unstack()
    say(f"  first layer, muon - adamw per seed: {', '.join(f'{d:+.4f}' for d in firsts['muon'] - firsts['adamw'])}")
    say(f"  other six, muon - adamw per seed: {', '.join(f'{d:+.4f}' for d in rest['muon'] - rest['adamw'])}")

    # The first layer's span against the step, with the other six beside it.
    for arm in ("adamw", "muon"):
        runs = [RUNS / "rq1" / f"mechanism-{arm}-s{seed}" for seed in (0, 1, 2)]
        first = curves(runs, "bias_span", first=True, width=100)
        rest_frames = [binned(rest_of_mixtures(run, "bias_span"), 100) for run in runs]
        rest_stack = pd.concat(rest_frames, axis=1)
        out = pd.DataFrame(
            {
                "step": first.index,
                "first": first["mean"].to_numpy(),
                "first_low": first["low"].to_numpy(),
                "first_high": first["high"].to_numpy(),
                "rest": rest_stack.mean(axis=1).to_numpy(),
                "rest_low": rest_stack.min(axis=1).to_numpy(),
                "rest_high": rest_stack.max(axis=1).to_numpy(),
            }
        )
        out.to_csv(DATA / f"rq1-span-{arm}.csv", index=False, float_format="%.5f")
        early = [analysis.first_mixture(analysis.routing_frame(run))["bias_span"] for run in runs]
        for low, high in ((500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 15259)):
            values = [e[(e.index > low) & (e.index <= high)].mean() for e in early]
            say(f"  first layer {arm} steps {low + 1}-{high}: mean {np.mean(values):.3f}")

    # The annex: the hidden-state cosine within a domain, per mixture, at the last checkpoint.
    annex = pd.read_csv(RUNS / "rq1" / "analysis" / "annex.csv")
    say(f"  annex columns: {', '.join(annex.columns[:30])}")


def tuning() -> None:
    say("== Tuning: 0.5B tokens, seed 0")
    swept = analysis.sweeps(RUNS / "tuning")
    rows = []
    for row in swept.to_dict("records"):
        run = RUNS / "tuning" / row["optimizer"] / f"lr{row['rate']:g}"
        if not run.exists():
            run = next((RUNS / "tuning" / row["optimizer"]).glob(f"lr{row['rate']}*"))
        first = per_mixture(run, "bias_span").iloc[0]
        rows.append({"optimizer": row["optimizer"], "rate": row["rate"], "loss": row["val_loss"], "span": row["bias_span"], "first": first})
        say(f"  {row['optimizer']:9s} {row['rate']:.5f} loss {row['val_loss']:.4f} span {row['bias_span']:.4f} first {first:.4f}")
    frame = pd.DataFrame(rows)
    for optimizer, part in frame.groupby("optimizer"):
        part.to_csv(DATA / f"tuning-{optimizer}.csv", index=False, float_format="%.5f")


def stress() -> None:
    say("== RQ2: the stress runs")
    frame = pd.read_csv(RUNS / "rq2" / "stress" / "stress.csv")
    keep = ["family", "arm", "lr_multiplier", "balancer_multiplier", "seed", "loss_capped", "load_churn_warmup", "router_logit_max_peak", "residual_max_peak", "spikes"]
    frame = frame[keep]
    say(frame.groupby(["family", "lr_multiplier", "balancer_multiplier"])[["loss_capped", "load_churn_warmup", "router_logit_max_peak", "residual_max_peak", "spikes"]].agg(["mean", "min", "max"]).to_string())
    usual = frame[frame["balancer_multiplier"] == 1.0]
    for family, part in usual.groupby("family"):
        part.sort_values(["lr_multiplier", "seed"]).to_csv(DATA / f"stress-{family}.csv", index=False, float_format="%.5f")
        means = part.groupby("lr_multiplier")[["loss_capped", "load_churn_warmup", "router_logit_max_peak", "residual_max_peak"]].mean().reset_index()
        means.to_csv(DATA / f"stress-{family}-mean.csv", index=False, float_format="%.5f")
    fast = frame[frame["balancer_multiplier"] != 1.0]
    fast.to_csv(DATA / "stress-fast.csv", index=False, float_format="%.5f")
    say(pd.read_csv(RUNS / "rq2" / "stress" / "sensitivity.csv").to_string())
    events = pd.read_csv(RUNS / "rq2" / "stress" / "events.csv")
    say(f"  episodes: {len(events)}")
    say(events.to_string())


RQ3_ARMS = {
    "adamw": "(a)",
    "muon": "(b)",
    "muon_except_router": "(c)",
    "muon_experts": "(d)",
    "muon_attention": "(e)",
    "muon_router": "(g)",
    "muon_dense": "(h)",
    "muon_attention_dense": "(i)",
    "muon-muonlr0.5x": "(b) at half rate",
    "adamw-unbalanced": "(a) no bias",
    "muon-unbalanced": "(b) no bias",
}


def rq3() -> None:
    say("== RQ3: seeds 3 to 5, 1B tokens")
    tokens = table("rq3")
    names = [
        "val_loss",
        "bias_span_first",
        "router_input_similarity_first",
        "mean_score_span_first",
        "bias_span",
        "load_churn_warmup",
        "max_load_ratio_first_early",
        "router_update_common_share_first",
        "update_load_correlation",
        "specialization",
        "usage_perplexity_final",
        "usage_perplexity_first_final",
        "bias_span_first_early",
        "bias_span_first_final",
    ]
    names = [n for n in names if n in tokens.columns]
    summary = tokens.groupby("arm")[names].agg(["mean", "min", "max"])
    for arm, label in RQ3_ARMS.items():
        if arm not in summary.index:
            say(f"  missing arm {arm}")
            continue
        row = summary.loc[arm]
        say(f"  {label:18s} " + " | ".join(f"{n} {row[(n, 'mean')]:.4f} [{row[(n, 'min')]:.4f}, {row[(n, 'max')]:.4f}]" for n in names))
    main = ["bias_span_first", "router_input_similarity_first", "bias_span", "val_loss", "max_load_ratio_first_early", "router_update_common_share_first", "load_churn_warmup", "mean_score_span_first"]
    for mine, yours in [
        ("muon", "adamw"),
        ("muon_except_router", "adamw"),
        ("muon", "muon_router"),
        ("muon_except_router", "muon_router"),
        ("muon_router", "adamw"),
        ("muon_except_router", "muon"),
        ("muon_experts", "adamw"),
        ("muon_attention_dense", "adamw"),
        ("muon_attention", "adamw"),
        ("muon_dense", "adamw"),
        ("muon-muonlr0.5x", "adamw"),
        ("muon-muonlr0.5x", "muon"),
        ("muon", "muon_experts"),
        ("muon_except_router", "muon_experts"),
    ]:
        show(f"{mine} - {yours}, tokens", pair(tokens, mine, yours, main))
    show("muon-unbalanced - adamw-unbalanced", pair(tokens, "muon-unbalanced", "adamw-unbalanced", ["usage_perplexity_final", "usage_perplexity_first_final", "val_loss", "collapsed_experts_final"]))
    show("muon - adamw, specialization", pair(tokens, "muon", "adamw", ["specialization", "update_load_correlation"]))

    # The geometry: one point per balanced run.
    points = tokens[tokens["balanced"]].copy()
    points["label"] = points["arm"].map(RQ3_ARMS)
    high = points["router_input_similarity_first"] > 0.3
    points.assign(group=np.where(high, "high", "low"))[
        ["arm", "label", "seed", "bias_span_first", "router_input_similarity_first", "mean_score_span_first"]
    ].to_csv(DATA / "rq3-geometry.csv", index=False, float_format="%.5f")
    for group, part in points.groupby(high):
        part[["arm", "seed", "bias_span_first", "router_input_similarity_first", "mean_score_span_first"]].to_csv(
            DATA / f"rq3-geometry-{'high' if group else 'low'}.csv", index=False, float_format="%.5f"
        )
    fit = np.polyfit(np.log(points["mean_score_span_first"]), np.log(points["bias_span_first"]), 1)
    predicted = np.polyval(fit, np.log(points["mean_score_span_first"]))
    residual = np.log(points["bias_span_first"]) - predicted
    r2 = 1 - residual.var() / np.log(points["bias_span_first"]).var()
    say(f"  log span on log offset over {len(points)} balanced runs: slope {fit[0]:.3f}, R2 {r2:.3f}")
    fit_c = np.polyfit(np.log(points["router_input_similarity_first"]), np.log(points["bias_span_first"]), 1)
    residual_c = np.log(points["bias_span_first"]) - np.polyval(fit_c, np.log(points["router_input_similarity_first"]))
    say(f"  log span on log cosine: R2 {1 - residual_c.var() / np.log(points['bias_span_first']).var():.3f}")
    pd.DataFrame({"x": [0.03, 0.9]}).assign(y=lambda f: np.exp(np.polyval(fit, np.log(f["x"])))).to_csv(
        DATA / "rq3-geometry-fit.csv", index=False, float_format="%.5f"
    )

    # The same readings one arm per row, in the order of the figure of RQ3, by seed and as the mean of the seeds: the router and the backbone crossed, Muon at half rate, then one part of the backbone on Muon.
    order = ["adamw", "muon_router", "muon_except_router", "muon", "muon-muonlr0.5x", "muon_experts", "muon_attention", "muon_dense", "muon_attention_dense"]
    readings = ["bias_span_first", "router_input_similarity_first", "mean_score_span_first"]
    rows = points.assign(row=points["arm"].map({arm: i for i, arm in enumerate(order)}), high=high.astype(int))
    rows[["arm", "row", "high", "seed", *readings]].sort_values(["row", "seed"]).to_csv(DATA / "rq3-arms.csv", index=False, float_format="%.5f")
    rows.groupby(["row", "arm", "high"])[readings].mean().reset_index().to_csv(DATA / "rq3-arms-mean.csv", index=False, float_format="%.5f")

    # The first layer against the step, for (a), (b) and (c), to step 4,000.
    for arm in ("adamw", "muon", "muon_except_router"):
        runs = [RUNS / "rq3" / f"mechanism-{arm}-s{seed}" for seed in (3, 4, 5)]
        columns = {}
        for name in ("router_input_similarity", "mean_score_span", "max_load_ratio", "bias_span"):
            frame = curves(runs, name, first=True, width=30, until=4000)
            columns["step"] = frame.index.to_numpy()
            columns[name] = frame["mean"].to_numpy()
            columns[f"{name}_low"] = frame["low"].to_numpy()
            columns[f"{name}_high"] = frame["high"].to_numpy()
        pd.DataFrame(columns).to_csv(DATA / f"rq3-early-{arm}.csv", index=False, float_format="%.5f")
        frames = [analysis.first_mixture(analysis.routing_frame(run)) for run in runs]
        for name in ("mean_score_span", "bias_span", "router_input_similarity", "max_load_ratio"):
            peaks = [(int(f[name].idxmax()), float(f[name].max())) for f in frames]
            say(f"  {arm} first-layer {name} peak (step, value): {peaks}")
        for low, high in ((300, 500), (500, 1262), (1262, 3000), (3000, 15259)):
            values = [f[(f.index > low) & (f.index <= high)]["max_load_ratio"].mean() for f in frames]
            say(f"  {arm} busiest expert steps {low + 1}-{high}: " + ", ".join(f"{v:.2f}" for v in values))


def balancer() -> None:
    say("== Bias speed: seeds 6 to 8")
    tokens = table("balancer")
    names = ["bias_span_first", "max_load_ratio_first_early", "router_input_similarity_first", "bias_span", "val_loss", "update_load_correlation", "load_churn"]
    for arm in sorted(tokens["arm"].unique()):
        say(f"  {arm:22s} " + " | ".join(f"{n} {per_seed(tokens, arm, n)}" for n in names))
    for mine, yours in [("muon", "adamw"), ("muon-balancer4x", "adamw-balancer4x"), ("muon-balancer4x", "muon"), ("adamw-balancer4x", "adamw")]:
        show(f"{mine} - {yours}, tokens", pair(tokens, mine, yours, names))


def balance_loss() -> None:
    say("== Balance loss: seeds 3 to 5, paired with RQ3")
    tokens = table("balance_loss")
    names = ["bias_span_first", "router_input_similarity_first", "mean_score_span_first", "max_load_ratio_first_early", "raw_max_load_ratio_first", "router_update_common_share_first", "val_loss", "bias_span", "specialization"]
    names = [n for n in names if n in tokens.columns]
    for arm in sorted(tokens["arm"].unique()):
        say(f"  {arm:34s} " + " | ".join(f"{n} {per_seed(tokens, arm, n)}" for n in names))
    for mine, yours in [
        ("muon-balanceloss0.0001", "adamw-balanceloss0.0001"),
        ("muon_except_router-balanceloss0.0001", "adamw-balanceloss0.0001"),
        ("muon-balanceloss0.0001", "muon"),
        ("muon_except_router-balanceloss0.0001", "muon_except_router"),
        ("adamw-balanceloss0.0001", "adamw"),
    ]:
        show(f"{mine} - {yours}, tokens", pair(tokens, mine, yours, names))


def large() -> None:
    say("== The large model: 3B tokens, seeds 0 to 2")
    tokens = table("confirmation")
    names = ["val_loss", "bias_span_first", "router_input_similarity_first", "bias_span", "load_churn_warmup", "max_load_ratio_first_early", "router_update_common_share_first", "update_load_correlation", "specialization", "spikes"]
    names = [n for n in names if n in tokens.columns]
    for arm in sorted(tokens["arm"].unique()):
        say(f"  {arm:22s} " + " | ".join(f"{n} {per_seed(tokens, arm, n)}" for n in names))
    for mine, yours in [("muon", "adamw"), ("muon_except_router", "adamw"), ("muon_except_router", "muon")]:
        show(f"{mine} - {yours}, tokens", pair(tokens, mine, yours, names))
    recheck = RUNS / "confirmation" / "recheck"
    if recheck.exists():
        for run in sorted(recheck.glob("*")):
            if run.is_dir() and analysis.has_records(run):
                metrics = analysis.metrics_frame(run)
                say(f"  recheck {run.name}: final validation loss {metrics['val_loss'].dropna().iloc[-1]:.4f}")


def contrasts() -> None:
    say("== The contrasts tfg.claims decided, both views")
    for block in ("rq1", "rq3", "balancer", "balance_loss", "confirmation"):
        frame = pd.read_csv(RUNS / block / "analysis" / "contrasts.csv")
        for row in frame.to_dict("records"):
            say(
                f"  {block:13s} {row['claim']:14s} {row['endpoint']:34s} {row['lower']:38s} < {row['higher']:38s} {row['view']:6s} "
                f"{row['difference']:+.4f} ({row['low']:+.4f} to {row['high']:+.4f}) {row['signs']} {row['verdict']}"
            )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for step in (rq1, tuning, stress, rq3, balancer, balance_loss, large, contrasts):
        step()
        say()
    (OUT / "numbers.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
