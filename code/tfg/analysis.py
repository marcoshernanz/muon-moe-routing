"""Endpoints, summaries, comparisons, and plots from a directory of runs.

A run is a directory with ``settings.json``, ``metrics.jsonl``, and ``routing.jsonl``. Every scalar routing reading becomes a time-integrated endpoint: its mean over the mixtures at each logged step, then over the logged steps after the warmup, and again over the early window, the first 5 percent of the run's steps after the warmup, and over the warmup window, the warmup steps themselves; each is also read in the first mixture alone (``_first``, block 1, the one the dense block 0 feeds), where the first chain found Muon's larger span; the parameter-group readings, one per model rather than per mixture, are integrated the same way, and so is the count of collapsed experts, read off the logged loads. The final bias span, usage perplexity (also in the first mixture), collapse count, and validation loss come along, and so does the specialization panel's divergence, averaged over the mixtures, when the run has been evaluated on the panel (``tfg.panel``), and the annex readings (``tfg.annex``) the same way, which also make the appendix's table. From the losses and pre-clip gradient norms of every step, which the log lines carry, come the loss spikes under the plan's definition and under Qwen's raw one, counted after the warmup, where the rate is set (the descent of the first steps is no spike), and, per optimizer, the post-warmup median norm (what the stress protocol's clip is set from), its 99.9th percentile, and the fraction of steps that were clipped. Runs are grouped by arm (the partition plus the balancer, router-rate, rate-multiplier, and balance-loss settings that depart from the default) and compared in pairs of arms pairing seeds, with a bootstrap interval and the per-seed signs of the difference, which is what the gates decide on, and the effect size, the difference over the pooled seed spread. Every comparison is made twice: at equal tokens, and at matched loss, both runs of a pair cut at the step where each first reaches the worse of their two final validation losses at that seed. The plan's frozen predictions and the rule that decides them are in ``tfg.claims``: every pair of arms a claim sets against each other is a contrast, and a claim passes when every one of its contrasts does. The figures are the plan's: every curve with a line per arm and the band its seeds span, the same per mixture as the per-block view, the early window of the router-position arms, the C6 scatters of each expert's update norm against its share of the batch, the exploratory grid with Kosson et al.'s equilibrium angle as the reference on the angular steps, the strips of one point per seed for every confirmatory endpoint, the first mixture's router-input similarity against its span, one point per run, the primary endpoint in its two views, the specialization figure from the panel files (the domain-conditional usage of one run per arm as a heat map, and the divergence per arm), given the other tier's runs, the confirmation figure with the confirmatory endpoints at both tiers, and, given the tuning block, the frontier: for each optimizer swept (AdamW, Muon, and the momentum control), the final validation loss of every rate against the integrated bias span, over every mixture and in the first alone, which shows what a span is worth in loss at each rate. The tables are the run inventory, the endpoints, the per-arm summary, the comparisons with the reference arm, the contrasts with the claims they decide, and the sweeps behind the frontier. Every figure is written as a PNG to look at and as a PDF for the thesis.
"""

import json
import math
from operator import itemgetter
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, to_hex
from scipy import stats

from tfg import checkpoint
from tfg.claims import (
    CLAIMS,
    CONFIRMATORY,
    HEADLINE,
    PRIMARY,
    VIEWS,
    claims,
    compare,
    contrasts,
    pair_key,
)
from tfg.optim import BETAS, GROUPS, MATRICES, WEIGHT_DECAY
from tfg.panel import DOMAINS
from tfg.telemetry import GAINS
from tfg.tier import TIERS
from tfg.train import BALANCE_LOSS, MUON_LR_MULTIPLIER, arm

matplotlib.use("Agg")  # draw to files only, so the analysis runs on a headless machine

# The columns that are not endpoints.
SETTINGS = [
    "run",
    "tier",
    "arm",
    "partition",
    "balanced",
    "router_lr_multiplier",
    "lr_multiplier",
    "muon_lr_multiplier",
    "balancer_multiplier",
    "adamw_beta2",
    "balance_loss",
    "seed",
    "train_windows",
    "steps",
    "tokens",
    "status",
]
INVENTORY = ["run", "tier", "arm", "seed", "steps", "tokens", "val_loss", "status"]  # the table
EARLY = 0.05  # of the run's steps, after the warmup: the window of the `_early` endpoints
FIRST = "_first"  # the suffix of an endpoint read in the first mixture alone
NORM_QUANTILE = 0.999  # of an optimizer's pre-clip gradient norm over every step, as in Qwen
SPIKE_WINDOW = 201  # steps of the centered rolling median a step's loss is compared with
SPIKE_FLOOR = 0.1  # the excess over that median Qwen counts as a spike; the primary floor
SPIKE_SIGMAS = 5  # the primary definition also asks this many robust standard deviations
COLLAPSE_SHARE = 0.1  # of its fair share of the batch's slots: an expert with less is collapsed
PANEL = "specialization.json"  # written by tfg.panel; against the mixture rather than the step
ANNEX = "annex.json"  # written by tfg.annex; against the mixture too
BY_MIXTURE = (PANEL, ANNEX)  # the files read against the mixture rather than the step
HEATMAP_TOP = 2.0  # of the uniform share, at least: the top of the usage heat map's color scale
HEATMAP_COLORS = "coolwarm"  # centered on the uniform share
CURVES = {
    "metrics.jsonl": ["loss", "val_loss", "grad_norm", "lr_factor"],
    "routing.jsonl": [
        "bias_span",
        "usage_perplexity",
        "load_churn",
        "router_entropy",
        "router_input_similarity",
        "mean_score_span",
        "raw_max_load_ratio",
        "update_load_correlation",
        "router_row_dispersion",
        "router_update_common_share",
        "router_row_similarity",
        "residual_max",
        "attention_logit_max",
        "expert_kurtosis_max",
        "collapsed_experts",
        "angular_step_router",
        "angular_step_experts",
    ],
    PANEL: ["specialization", "specialization_nonspace"],
    ANNEX: [
        "saturation",
        "coactivation_max",
        "coactivation_mean",
        "routing_purity",
        "hidden_cosine_between",
        "hidden_cosine_within",
        "routing_cosine_between",
        "expert_similarity_mean",
        "gradient_alignment_mean",
    ],
}
REFERENCE_ARM = (
    "adamw"  # the all-AdamW arm: what the comparisons and the reference line are against
)
FRONTIER = ("adamw", "muon", "momentum")  # the sweeps the frontier draws: (a), (b), (f)
FRONTIER_SPANS = ["bias_span", PRIMARY]  # its two panels: every mixture, and the first alone
FRONTIER_COLUMNS = [  # of the table behind it
    "optimizer",
    "rate",
    "run",
    "status",
    "steps",
    "val_loss",
    *FRONTIER_SPANS,
    "bias_span_final",
]
FORMATS = ("png", "pdf")  # of every figure: one to look at, one for the thesis
STRIPS = CONFIRMATORY  # the endpoints drawn per arm with a point per seed
GEOMETRY = ("router_input_similarity_first", PRIMARY)  # the first mixture's input against its span
EARLY_ARMS = ("adamw", "muon", "muon_except_router", "muon_router")  # (a), (b), (c), (g)
EARLY_CURVES = ("load_churn", "router_entropy")  # the early window's readings
UNBALANCED_CURVE = "usage_perplexity"  # drawn once more over the no-balancer arms alone (C3)
SCATTERED = {
    "expert_update_norm": "",
    "expert_update_spectral_norm": "_spectral",
}  # C6: the file suffix
SCATTER_POINTS = 3  # steps of the C6 scatters: the first, the middle, and the last after the warmup
GRID = [  # the exploratory grid, one small panel each
    "margin_mean",
    "margin_low",
    "router_logit_norm",
    "raw_usage_perplexity",
    "router_svd_entropy",
    "router_row_similarity",
    "expert_kurtosis_max",
    "collapsed_experts",
    "router_logit_max",
    "expert_output_max",
    "expert_update_dispersion",
    *(f"weight_norm_{group}" for group in (*GROUPS, GAINS)),
    *(f"angular_step_{group}" for group in GROUPS),
]
GRID_COLUMNS = 4
MARKERS = ("o", "^")  # one per tier in the confirmation figure
TIER_SHIFTS = (-0.15, 0.15)  # of the tiers' points about the arm's position
BAND_ALPHA = 0.2  # of the band an arm's seeds span


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """The records of a JSON-lines file, without the last line when a crash cut it short."""
    return [json.loads(line) for line in checkpoint.records(path.read_text(encoding="utf-8"))]


def metrics_frame(run_dir: Path) -> pd.DataFrame:
    """One row per logged step, the validation loss empty at the steps without an evaluation."""
    frame = pd.DataFrame(read_jsonl(run_dir / "metrics.jsonl"))
    return frame if "val_loss" in frame.columns else frame.assign(val_loss=np.nan)  # none yet


def routing_frame(run_dir: Path) -> pd.DataFrame:
    """One row per logged step and mixture, with the scalar readings only, plus the number of collapsed experts: those with under ``COLLAPSE_SHARE`` of their fair share of the batch."""
    rows: list[dict[str, object]] = []
    for line in read_jsonl(run_dir / "routing.jsonl"):
        mixtures = line["mixtures"]
        assert isinstance(mixtures, list)
        for index, reading in enumerate(mixtures):
            scalars = {key: value for key, value in reading.items() if not isinstance(value, list)}
            load = np.asarray(reading["load"], dtype=float)  # [E]
            scalars["collapsed_experts"] = int((load < COLLAPSE_SHARE / len(load)).sum())
            rows.append({"step": line["step"], "mixture": index, **scalars})
    return pd.DataFrame(rows)


def group_frame(run_dir: Path) -> pd.DataFrame:
    """One row per logged step, indexed by step, with the parameter-group readings named as ``weight_norm_experts``; no columns for a run logged before they existed."""
    rows: dict[int, dict[str, object]] = {}
    for line in read_jsonl(run_dir / "routing.jsonl"):
        step, groups = line["step"], line.get("groups", {})
        assert isinstance(step, int) and isinstance(groups, dict)
        rows[step] = {
            f"{name}_{group}": value
            for group, reading in groups.items()
            for name, value in reading.items()
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def per_step(readings: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """One row per logged step, indexed by step: every scalar routing reading averaged over the mixtures, and the parameter-group readings; no rows for a run that ended before its first reading."""
    if readings.empty:
        return pd.DataFrame(index=pd.Index([], name="step"))
    return readings.drop(columns="mixture").groupby("step").mean().join(groups)


def first_mixture(readings: pd.DataFrame) -> pd.DataFrame:
    """The first mixture's routing readings, one row per logged step, indexed by step; no rows for a run that ended before its first reading."""
    if readings.empty:
        return pd.DataFrame(index=pd.Index([], name="step"))
    return readings[readings["mixture"] == 0].drop(columns="mixture").set_index("step")


def step_frame(run_dir: Path) -> pd.DataFrame:
    """``per_step`` of the run's files."""
    return per_step(routing_frame(run_dir), group_frame(run_dir))


def block_curves(readings: pd.DataFrame, name: str) -> pd.DataFrame:
    """One mixture reading against the step from the run's routing frame, a column per mixture in block order, the steps where no mixture has it dropped; empty for a reading the mixtures do not carry, like a parameter group's."""
    if name not in readings.columns:
        return pd.DataFrame(index=pd.Index([], name="step"))
    return readings.pivot(index="step", columns="mixture", values=name).dropna(how="all")


def window(frame: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    """The readings logged at the steps in ``(low, high]``; when none was, the one logged nearest to that window, so that a run too short for the plan's windows still has every endpoint."""
    steps = frame.index.to_numpy()  # [S]
    inside = (steps > low) & (steps <= high)  # [S]
    if inside.any() or frame.empty:
        return frame[inside]
    distance = np.where(steps <= low, low - steps, steps - high)  # [S]
    return frame.iloc[[int(distance.argmin())]]


def mixture_frame(run_dir: Path, file: str) -> pd.DataFrame:
    """One row per mixture of the panel's evaluation or of the annex readings, with the scalar readings the file carries; its lists and tables are left out."""
    mixtures = json.loads((run_dir / file).read_text(encoding="utf-8"))["mixtures"]
    rows = [{k: v for k, v in m.items() if not isinstance(v, (list, dict))} for m in mixtures]
    return pd.DataFrame(rows).astype(float)


def steps_of(windows: pd.Series) -> pd.Series:
    """One value per step from the windows the log lines carry, indexed by step; ``windows`` is indexed by the log step, where its window ends."""
    values = [np.asarray(window, dtype=float) for window in windows]  # [w] each
    steps = [np.arange(end - len(w) + 1, end + 1) for end, w in zip(windows.index, values)]
    return pd.Series(np.concatenate(values), index=np.concatenate(steps))


def spikes(losses: pd.Series) -> pd.DataFrame:
    """Per step, the loss, its excess over the centered rolling median of ``SPIKE_WINDOW`` steps, and whether it is a spike: under the primary definition, an excess above both ``SPIKE_FLOOR`` and ``SPIKE_SIGMAS`` robust standard deviations of the excess in the window (the median absolute deviation scaled by 1.4826, the rule of the K2-V2 report, arXiv 2512.06201); under the raw one, above ``SPIKE_FLOOR`` alone."""
    losses = losses.where(np.isfinite(losses))  # a diverged run ends on a loss that is no number
    excess = losses - losses.rolling(SPIKE_WINDOW, center=True, min_periods=1).median()
    deviation = excess.rolling(SPIKE_WINDOW, center=True, min_periods=1).apply(
        stats.median_abs_deviation, raw=True, kwargs={"scale": "normal", "nan_policy": "omit"}
    )  # without the policy one such loss would blank the deviation of the hundred steps before it
    return pd.DataFrame(
        {
            "loss": losses,
            "excess": excess,
            "spike": excess > np.maximum(SPIKE_FLOOR, SPIKE_SIGMAS * deviation),
            "spike_raw": excess > SPIKE_FLOOR,
        }
    )


def spike_table(run_dir: Path, *, until: int | None = None) -> pd.DataFrame:
    """``spikes`` over the run's training losses up to the logged step ``until``, for the steps after the warmup: the rolling median sees the warmup steps, the count leaves them out."""
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    losses = steps_of(metrics_frame(run_dir).set_index("step")["losses"])  # of every step [S]
    if until is not None:
        losses = losses[losses.index <= until]
    table = spikes(losses)
    return table[table.index > settings["warmup_steps"]]


def curve(run_dir: Path, file: str, name: str) -> pd.Series:
    """One quantity against the step, mixtures averaged, missing values dropped; the panel's against the mixture."""
    if file in BY_MIXTURE:
        readings = mixture_frame(run_dir, file)
        return readings[name] if name in readings.columns else pd.Series(dtype=float)
    if file == "metrics.jsonl":
        return metrics_frame(run_dir).groupby("step")[name].mean().dropna()
    frame = step_frame(run_dir)
    return frame[name].dropna() if name in frame.columns else pd.Series(dtype=float)


def status_of(run_dir: Path) -> str:
    """Where a run stands: ``diverged`` when its last record carries the trainer's mark, a validation loss that is not a number, or when it reached its last step with a validation loss no better than its first training loss, which is how the sensitivity counts it too; ``finished`` when its log reaches the last step; else ``unfinished``."""
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    records = read_jsonl(run_dir / "metrics.jsonl")
    if not records:
        return "unfinished"
    last, first = records[-1], records[0]["losses"]
    val_loss = last.get("val_loss", 0.0)
    assert isinstance(val_loss, float) and isinstance(first, list)
    finished = last["step"] == settings["steps"]
    if math.isnan(val_loss) or (finished and val_loss >= first[0]):
        return "diverged"
    return "finished" if finished else "unfinished"


def endpoints(run_dir: Path, *, until: int | None = None) -> dict[str, object]:
    """The settings that identify a run and every endpoint computed from its files, up to the logged step ``until`` (by default the whole run)."""
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    records = read_jsonl(run_dir / "metrics.jsonl")
    metrics = pd.DataFrame(records)
    metrics["evaluated"] = [("val_loss" in record) for record in records]  # NaN when diverged
    routing = routing_frame(run_dir)
    steps = per_step(routing, group_frame(run_dir))
    if until is not None:
        metrics = metrics[metrics["step"] <= until]
        routing = routing[routing["step"] <= until]
        steps = steps[steps.index <= until]
    first = first_mixture(routing)
    warmup = settings["warmup_steps"]
    bounds = {
        "": (warmup, np.inf),  # integrated: every logged step after the warmup
        "_early": (warmup, warmup + EARLY * settings["steps"]),
        "_warmup": (0, warmup),
    }
    integrated: dict[str, object] = {
        f"{name}{scope}{suffix}": float(value)
        for scope, frame in (("", steps), (FIRST, first))
        for suffix, (low, high) in bounds.items()
        for name, value in window(frame, low, high).mean().items()
    }
    for file in BY_MIXTURE:  # the final checkpoint's, whatever the cut
        if (run_dir / file).exists():
            readings = mixture_frame(run_dir, file)
            names = [name for name in CURVES[file] if name in readings.columns]
            means = readings[names].mean()
            integrated |= {str(name): float(value) for name, value in means.items()}
    by_step = metrics.set_index("step")
    losses = steps_of(by_step["losses"])  # of every step [S]
    excess = spike_table(run_dir, until=until)
    clips = {"adamw": settings["adamw_clip"], "muon": settings["muon_clip"]}
    clips["momentum"] = clips["muon"]
    for kind in by_step["grad_norms"].iloc[0]:  # the optimizers of the partition
        norms = steps_of(by_step["grad_norms"].map(itemgetter(kind)))  # of every step [S]
        integrated[f"grad_norm_median_{kind}"] = float(
            norms[norms.index > settings["warmup_steps"]].median()
        )
        integrated[f"grad_norm_p999_{kind}"] = float(norms.quantile(NORM_QUANTILE))
        integrated[f"clip_fraction_{kind}"] = float((norms > clips[kind]).mean())
    evaluated = metrics[metrics["evaluated"]]  # no row, and no such column, before the first
    val_loss = float(evaluated["val_loss"].iloc[-1]) if len(evaluated) else math.nan
    last = int(metrics["step"].iloc[-1])
    final = steps.iloc[-1] if len(steps) else pd.Series(dtype=float)  # the last reading's
    final_first = first.iloc[-1] if len(first) else pd.Series(dtype=float)
    per_10k = 10_000 / len(excess) if len(excess) else math.nan  # of the steps after the warmup
    shape = TIERS[settings["tier"]]  # its batch, unless the run was given another
    return {
        "run": run_dir.name,
        "tier": settings["tier"],
        "arm": arm_of(settings),
        "partition": settings["partition"],
        "balanced": settings["balanced"],
        "router_lr_multiplier": settings["router_lr_multiplier"],
        "lr_multiplier": settings["lr_multiplier"],
        "muon_lr_multiplier": settings.get("muon_lr_multiplier", MUON_LR_MULTIPLIER),
        "balancer_multiplier": settings["balancer_multiplier"],
        "adamw_beta2": settings.get("adamw_beta2", BETAS[1]),
        "balance_loss": settings.get("balance_loss", BALANCE_LOSS),
        "seed": settings["seed"],
        "train_windows": settings.get("train_windows"),  # what the data order depends on
        "steps": last,
        "tokens": last * settings.get("batch_size", shape.batch_size) * shape.context_len,
        "status": status_of(run_dir),  # of the whole run, whatever the cut
        "val_loss": val_loss,
        "loss_initial": float(losses.iloc[0]),
        "bias_span_final": float(final.get("bias_span", math.nan)),
        "bias_span_max": float(routing["bias_span"].max()) if len(routing) else math.nan,
        "usage_perplexity_final": float(final.get("usage_perplexity", math.nan)),
        f"bias_span{FIRST}_final": float(final_first.get("bias_span", math.nan)),
        f"usage_perplexity{FIRST}_final": float(final_first.get("usage_perplexity", math.nan)),
        "collapsed_experts_final": float(final.get("collapsed_experts", math.nan)),
        "spikes": int(excess["spike"].sum()),
        "spikes_raw": int(excess["spike_raw"].sum()),
        "spikes_per_10k": float(excess["spike"].sum() * per_10k),
        "spikes_raw_per_10k": float(excess["spike_raw"].sum() * per_10k),
        **integrated,
    }


def arm_of(settings: dict[str, object]) -> str:
    partition, balanced = settings["partition"], settings["balanced"]
    router_multiplier, multiplier, balancer_multiplier = (
        settings["router_lr_multiplier"],
        settings["lr_multiplier"],
        settings["balancer_multiplier"],
    )
    assert isinstance(partition, str) and isinstance(balanced, bool)
    assert isinstance(router_multiplier, (int, float)) and isinstance(multiplier, (int, float))
    assert isinstance(balancer_multiplier, (int, float))
    beta2 = settings.get("adamw_beta2", BETAS[1])  # absent from runs logged before the override
    muon_multiplier = settings.get("muon_lr_multiplier", MUON_LR_MULTIPLIER)  # the same
    balance_loss = settings.get("balance_loss", BALANCE_LOSS)  # the same
    assert isinstance(beta2, (int, float)) and isinstance(muon_multiplier, (int, float))
    assert isinstance(balance_loss, (int, float))
    return arm(
        partition,
        balanced=balanced,
        router_lr_multiplier=router_multiplier,
        lr_multiplier=multiplier,
        balancer_multiplier=balancer_multiplier,
        adamw_beta2=beta2,
        muon_lr_multiplier=muon_multiplier,
        balance_loss=balance_loss,
    )


def has_records(run_dir: Path) -> bool:
    """Whether ``run_dir`` is a run with a record to read: one that has only just started has none."""
    metrics = run_dir / "metrics.jsonl"
    return (run_dir / "routing.jsonl").exists() and metrics.exists() and bool(read_jsonl(metrics))


def run_dirs(runs_dir: Path) -> list[Path]:
    """Every run directory under ``runs_dir`` that has logged a record, in name order."""
    dirs = sorted(path for path in runs_dir.glob("*") if has_records(path))
    if not dirs:
        raise FileNotFoundError(f"no runs under {runs_dir}")
    return dirs


def collect(runs_dir: Path) -> pd.DataFrame:
    """One row of endpoints per run; refused when the runs did not all train on the same number of windows, because then a seed did not give them the same data order and they do not pair."""
    table = pd.DataFrame([endpoints(run_dir) for run_dir in run_dirs(runs_dir)])
    windows = table["train_windows"].dropna().unique()
    if len(windows) > 1:
        raise ValueError(f"the runs under {runs_dir} saw different training sets: {windows}")
    return table


def sweeps(tuning_dir: Path) -> pd.DataFrame:
    """The table behind the frontier: one row per run of the tuning block's sweeps of the optimizers in ``FRONTIER``, each under its optimizer's directory as ``lr<rate>``, the way ``tfg.sweep`` names them, with the optimizer swept, its rate, the run's status, its final validation loss, and its integrated and final bias span; a run that has logged nothing yet is left out."""
    rows: list[dict[str, object]] = []
    for optimizer in FRONTIER:
        runs = [d for d in (tuning_dir / optimizer).glob("lr*") if has_records(d)]
        for run_dir in sorted(runs, key=lambda d: float(d.name.removeprefix("lr"))):
            rate = float(run_dir.name.removeprefix("lr"))
            rows.append({"optimizer": optimizer, "rate": rate, **endpoints(run_dir)})
    if not rows:
        raise FileNotFoundError(f"no sweep runs under {tuning_dir}")
    return pd.DataFrame(rows)[FRONTIER_COLUMNS]


def step_reaching(run_dir: Path, loss: float) -> int:
    """The first logged step whose validation loss is at or below ``loss``, else the last step."""
    metrics = metrics_frame(run_dir)
    reached = metrics.loc[metrics["val_loss"] <= loss, "step"]
    return int(reached.iloc[0]) if not reached.empty else int(metrics["step"].iloc[-1])


Memo = dict[tuple[str, int], dict[Any, Any]]  # the endpoints of a run cut at a step


def matched(
    table: pd.DataFrame, runs_dir: Path, pair: tuple[str, str], *, memo: Memo | None = None
) -> pd.DataFrame:
    """The endpoints of a pair of arms again at matched loss: at every seed both have, each of the two runs cut at the first logged step whose validation loss is at or below the worse of their two final validation losses, recorded as ``matched_step``. The worse run of the two is cut where it first got that low, which is its last step unless its validation loss rose again afterwards, and both keep their last step when either has no final loss. ``memo`` keeps the cuts already computed, since an arm is cut at the same step in many pairs."""
    memo = {} if memo is None else memo
    both = table[table["arm"].isin(pair)]
    rows: list[dict[Any, Any]] = []
    for _, runs in both.groupby("seed"):
        if len(runs) != 2:  # a seed only one of the two has
            continue
        target = float(runs["val_loss"].max(skipna=False))  # the worse final loss, NaN if either is
        for whole in runs.to_dict("records"):
            run = str(whole["run"])
            until = step_reaching(runs_dir / run, target)
            if until == whole["steps"]:
                memo.setdefault((run, until), whole)  # not cut: the row the table has
            elif (run, until) not in memo:
                memo[run, until] = endpoints(runs_dir / run, until=until)
            rows.append({**memo[run, until], "matched_step": until})
    return pd.DataFrame(rows)


def finished_pairs(table: pd.DataFrame, pair: tuple[str, str]) -> pd.DataFrame:
    """The runs of the two arms at the seeds where both finished, which is what a contrast is decided on: a run still under way is at another token count, and one that diverged has no endpoints to compare."""
    done = table[(table["status"] == "finished") & table["arm"].isin(pair)]
    return done[done.groupby("seed")["arm"].transform("nunique") == len(set(pair))]


def endpoint_names(table: pd.DataFrame) -> list[str]:
    """The columns of a table of runs that are endpoints."""
    return [name for name in table.columns if name not in SETTINGS]


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean, spread, and count of every endpoint per arm."""
    return table.groupby("arm")[endpoint_names(table)].agg(["mean", "std", "count"])


def save_figure(out_dir: Path, name: str) -> Path:
    """Write the current figure as ``name.png`` and ``name.pdf`` under ``out_dir`` and close it; returns the PNG."""
    for extension in FORMATS:
        plt.savefig(out_dir / f"{name}.{extension}", bbox_inches="tight")
    plt.close()
    return out_dir / f"{name}.{FORMATS[0]}"


def legend(axis: Axes) -> None:
    """A legend on ``axis`` when something on it is labeled."""
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize="small")


def plot_arms(
    axis: Axes,
    curves: dict[Path, pd.Series],
    *,
    arms: dict[Path, str],
    palette: dict[str, str],
    marker: str | None = None,
) -> None:
    """A line per arm on ``axis``: the mean over its runs at each step, with the band between the lowest and the highest run; an arm with one run is that run's curve."""
    by_arm: dict[str, list[pd.Series]] = {}
    for run_dir, series in curves.items():
        if not series.empty:
            by_arm.setdefault(arms[run_dir], []).append(series)
    for name, group in by_arm.items():
        frame = pd.concat(group, axis=1)  # [S, runs], NaN where a run lacks the step
        label = f"{name} ({len(group)} seeds)" if len(group) > 1 else name
        axis.plot(
            frame.index,
            frame.mean(axis=1).to_numpy(),
            color=palette[name],
            label=label,
            marker=marker,
        )
        if len(group) > 1:
            axis.fill_between(
                frame.index,
                frame.min(axis=1).to_numpy(),
                frame.max(axis=1).to_numpy(),
                color=palette[name],
                alpha=BAND_ALPHA,
                linewidth=0,
            )


def series_of(frames: dict[Path, pd.DataFrame], name: str) -> dict[Path, pd.Series]:
    """One reading against the step per run from the per-step frames, missing values dropped; empty for a run logged without the reading."""
    return {
        run_dir: frame[name].dropna() if name in frame.columns else pd.Series(dtype=float)
        for run_dir, frame in frames.items()
    }


def plot_blocks(
    readings: dict[Path, pd.DataFrame],
    name: str,
    *,
    arms: dict[Path, str],
    palette: dict[str, str],
    out_dir: Path,
) -> Path | None:
    """The per-block view of one mixture reading: small multiples, one panel per mixture in block order, a line per arm; returns the file, or nothing when no run carries the reading per mixture."""
    curves = {run_dir: block_curves(frame, name) for run_dir, frame in readings.items()}
    count = max(len(frame.columns) for frame in curves.values())
    if count == 0:
        return None
    _, axes = plt.subplots(
        count, 1, figsize=(7, 2 * count), sharex=True, squeeze=False, layout="constrained"
    )
    for mixture, axis in enumerate(axes[:, 0]):
        per_run = {d: f[mixture] for d, f in curves.items() if mixture in f.columns}
        plot_arms(axis, per_run, arms=arms, palette=palette)
        axis.set_ylabel(f"{name}\nmixture {mixture}")
    axes[-1, 0].set_xlabel("step")
    legend(axes[0, 0])
    return save_figure(out_dir, f"{name}_blocks")


def strip(
    axis: Axes,
    table: pd.DataFrame,
    name: str,
    *,
    palette: dict[str, str],
) -> bool:
    """One endpoint per arm on ``axis``: a point per seed and the mean as a bar, the arms in name order; False when no run has it."""
    values = table.dropna(subset=[name]) if name in table.columns else table.iloc[:0]
    if values.empty:
        return False
    names = sorted(set(values["arm"]))
    for position, arm_name in enumerate(names):
        seeds = values.loc[values["arm"] == arm_name, name].to_numpy(dtype=float)  # [seeds]
        axis.scatter([position] * len(seeds), seeds, color=palette[arm_name], zorder=2)
        axis.hlines(seeds.mean(), position - 0.25, position + 0.25, color=palette[arm_name])
    axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
    axis.set_ylabel(name)
    return True


def plot_strips(table: pd.DataFrame, *, palette: dict[str, str], out_dir: Path) -> list[Path]:
    """One figure per endpoint in ``STRIPS`` the runs have: the primary endpoint and every confirmatory one, per arm with a point per seed."""
    files: list[Path] = []
    for name in STRIPS:
        _, axis = plt.subplots(figsize=(7, 4))
        if strip(axis, table, name, palette=palette):
            files.append(save_figure(out_dir, f"{name}_seeds"))
        else:
            plt.close()
    return files


def plot_views(tables: dict[str, pd.DataFrame], *, palette: dict[str, str], out_dir: Path) -> Path:
    """The primary endpoint in its two views side by side: at equal tokens and at matched loss."""
    _, axes = plt.subplots(
        1,
        len(tables),
        figsize=(4.5 * len(tables), 4),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for axis, (matching, table) in zip(axes[0], tables.items()):
        strip(axis, table, PRIMARY, palette=palette)
        axis.set_title(VIEWS[matching])
    return save_figure(out_dir, "views")


def plot_specialization(
    dirs: list[Path], *, arms: dict[Path, str], palette: dict[str, str], out_dir: Path
) -> list[Path]:
    """The specialization figure, from the runs evaluated on the panel: the domain-conditional usage of the last mixture of one run per arm, its lowest seed's, as a heat map of the share of each domain's tokens every expert takes relative to the uniform share; and the divergence per arm, a point per seed with the mean as a bar. Nothing when no run has been evaluated."""
    evaluated = [run_dir for run_dir in dirs if (run_dir / PANEL).exists()]
    if not evaluated:
        return []
    first: dict[str, Path] = {}  # the first run of each arm in name order: its lowest seed
    for run_dir in evaluated:
        first.setdefault(arms[run_dir], run_dir)
    results = {n: json.loads((d / PANEL).read_text(encoding="utf-8")) for n, d in first.items()}
    ratios = {
        name: np.array([r["mixtures"][-1]["share"][domain] for domain in DOMAINS]) / r["uniform"]
        for name, r in results.items()
    }  # [G, E] each
    top = max(HEATMAP_TOP, *(float(ratio.max()) for ratio in ratios.values()))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=top)
    _, axes = plt.subplots(
        len(ratios), 1, figsize=(7, 1 + 1.5 * len(ratios)), sharex=True, squeeze=False
    )
    for axis, (name, ratio) in zip(axes[:, 0], ratios.items()):
        axis.imshow(ratio, aspect="auto", cmap=HEATMAP_COLORS, norm=norm)
        axis.set_yticks(range(len(DOMAINS)), DOMAINS)
        axis.set_title(f"{first[name].name}: last mixture, step {results[name]['step']}", size=9)
    axes[-1, 0].set_xlabel("expert")
    plt.colorbar(
        ScalarMappable(norm=norm, cmap=HEATMAP_COLORS),
        ax=list(axes[:, 0]),
        label="share of the domain's tokens, over uniform",
    )
    files = [save_figure(out_dir, "specialization_heatmap")]
    values = pd.DataFrame(
        {
            "arm": [arms[run_dir] for run_dir in evaluated],
            "partition": [arms[run_dir] for run_dir in evaluated],
            "specialization": [
                float(curve(run_dir, PANEL, "specialization").mean()) for run_dir in evaluated
            ],
        }
    )
    _, axis = plt.subplots(figsize=(7, 4))
    strip(axis, values, "specialization", palette=palette)
    axis.set_ylabel("specialization (nats), mixtures averaged")
    files.append(save_figure(out_dir, "specialization_seeds"))
    return files


def plot_early(
    frames: dict[Path, pd.DataFrame],
    *,
    ends: dict[Path, int],
    arms: dict[Path, str],
    palette: dict[str, str],
    out_dir: Path,
) -> Path | None:
    """The early window: churn and router entropy through the warmup and the first ``EARLY`` of the run after it (``ends``), the router-position arms side by side; nothing when no run is one of them."""
    chosen = {run_dir: frame for run_dir, frame in frames.items() if arms[run_dir] in EARLY_ARMS}
    if not chosen:
        return None
    _, axes = plt.subplots(
        1,
        len(EARLY_CURVES),
        figsize=(4.5 * len(EARLY_CURVES), 4),
        squeeze=False,
        layout="constrained",
    )
    for axis, name in zip(axes[0], EARLY_CURVES):
        curves = {d: s[s.index <= ends[d]] for d, s in series_of(chosen, name).items()}
        plot_arms(axis, curves, arms=arms, palette=palette, marker=".")
        axis.set_xlabel("step")
        axis.set_ylabel(name)
    legend(axes[0, 0])
    return save_figure(out_dir, "early_window")


def plot_scatter(
    dirs: list[Path],
    *,
    warmups: dict[Path, int],
    arms: dict[Path, str],
    palette: dict[str, str],
    out_dir: Path,
) -> list[Path]:
    """The C6 panel: each expert's update norm against the share of the batch it took, the Frobenius norm the plan reads and the spectral one for the appendix, at ``SCATTER_POINTS`` logged steps after the warmup (the first, the middle, the last; a run too short for that gives its last), one run per arm, its lowest seed with a reading, every mixture's experts pooled: a row per arm, a column per step. Nothing when no run logged the norms per expert."""
    picked: dict[str, list[dict[str, object]]] = {}
    for run_dir in dirs:
        lines = [] if arms[run_dir] in picked else read_jsonl(run_dir / "routing.jsonl")
        if not lines:  # a later seed of an arm already picked, or a run without a reading
            continue
        after = [line for line in lines if int(str(line["step"])) > warmups[run_dir]] or lines[-1:]
        positions = np.linspace(0, len(after) - 1, SCATTER_POINTS).round().astype(int)  # [P]
        picked[arms[run_dir]] = [after[position] for position in positions]
    if not picked:
        return []
    files: list[Path] = []
    for reading, suffix in SCATTERED.items():
        drawn = False
        _, axes = plt.subplots(
            len(picked),
            SCATTER_POINTS,
            figsize=(3 * SCATTER_POINTS, 2.5 * len(picked)),
            squeeze=False,
            layout="constrained",
        )
        for row, (name, lines) in zip(axes, picked.items()):
            for axis, line in zip(row, lines):
                mixtures = line["mixtures"]
                assert isinstance(mixtures, list)
                for mixture in mixtures:
                    norms = mixture.get(reading)
                    if isinstance(norms, list):
                        axis.scatter(mixture["load"], norms, s=8, color=palette[name], alpha=0.6)
                        drawn = True
                axis.set_title(f"{name}: step {line['step']}", size=8)
            row[0].set_ylabel(reading, size=8)
        for axis in axes[-1]:
            axis.set_xlabel("share of the batch's slots")
        if drawn:
            files.append(save_figure(out_dir, f"update_load_scatter{suffix}"))
        else:
            plt.close()
    return files


def equilibrium(run_dir: Path) -> pd.Series:
    """Kosson et al.'s equilibrium angular step of AdamW with decoupled decay at each logged step of the run, from its rate there: the reference line of the angular-step panels."""
    metrics = metrics_frame(run_dir).set_index("step")
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    rate = settings["adamw_lr"] * metrics["lr_factor"]  # [S]
    return np.sqrt(2 * rate * WEIGHT_DECAY * (1 - BETAS[0]) / (1 + BETAS[0]))


def plot_grid(
    frames: dict[Path, pd.DataFrame],
    *,
    arms: dict[Path, str],
    palette: dict[str, str],
    out_dir: Path,
) -> Path | None:
    """The exploratory grid: one small panel per reading of ``GRID`` the runs have, a line per arm with its seed band; on the angular-step panels of the decayed groups, the equilibrium angle of the all-AdamW arm's first run as a dashed reference. Nothing when the runs have none of the readings."""
    present = [n for n in GRID if any(n in f.columns for f in frames.values())]
    if not present:
        return None
    reference = next((d for d, name in arms.items() if name == REFERENCE_ARM), None)
    rows = -(-len(present) // GRID_COLUMNS)  # ceiling
    figure, axes = plt.subplots(
        rows,
        GRID_COLUMNS,
        figsize=(3.5 * GRID_COLUMNS, 2.5 * rows),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for axis, name in zip(axes.flat, present):
        plot_arms(axis, series_of(frames, name), arms=arms, palette=palette)
        axis.set_ylabel(name, size=8)
        if reference and name in [f"angular_step_{group}" for group in MATRICES]:
            line = equilibrium(reference)
            axis.plot(
                line.index, line.to_numpy(), "k--", linewidth=0.8, label="equilibrium (adamw)"
            )
    for axis in list(axes.flat)[len(present) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("step")
    handles = {}  # one entry per label, from whichever panel drew it
    for axis in axes.flat:
        handles |= dict(zip(*reversed(axis.get_legend_handles_labels())))
    figure.legend(handles.values(), handles.keys(), loc="outside lower center", ncols=4)
    return save_figure(out_dir, "exploratory")


def plot_tiers(tables: dict[str, pd.DataFrame], *, palette: dict[str, str], out_dir: Path) -> Path:
    """The confirmation figure: the confirmatory endpoints the runs have, one panel each, per arm with a point per seed, a marker per table of runs, labeled by its key and its tier."""
    names = [n for n in CONFIRMATORY if any(n in t.columns for t in tables.values())]
    arms = sorted(set().union(*(set(t["arm"]) for t in tables.values())))
    rows = -(-len(names) // GRID_COLUMNS)  # ceiling
    figure, axes = plt.subplots(
        rows,
        GRID_COLUMNS,
        figsize=(3.5 * GRID_COLUMNS, 3 * rows),
        squeeze=False,
        layout="constrained",
    )
    for axis, name in zip(axes.flat, names):
        for (key, table), marker, shift in zip(tables.items(), MARKERS, TIER_SHIFTS):
            if name not in table.columns:
                continue
            values = table.dropna(subset=[name])
            label = f"{key} ({table['tier'].iloc[0]})"
            for position, arm_name in enumerate(arms):
                seeds = values.loc[values["arm"] == arm_name, name].to_numpy(dtype=float)
                axis.scatter(
                    [position + shift] * len(seeds),
                    seeds,
                    marker=marker,
                    color=palette[arm_name],
                    label=label if position == 0 else None,
                )
        axis.set_xticks(range(len(arms)), arms, rotation=30, ha="right", size=7)
        axis.set_ylabel(name, size=8)
    for axis in list(axes.flat)[len(names) :]:
        axis.set_visible(False)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    return save_figure(out_dir, "tiers")


def plot_frontier(swept: pd.DataFrame, *, palette: dict[str, str], out_dir: Path) -> Path:
    """The tuning block's frontier: per optimizer swept, the final validation loss of its finished runs against their integrated bias span, over every mixture and in the first alone, one panel each, joined in rate order with each point labeled by its rate and the rate that won filled in; a run that diverged has no loss and is left out."""
    _, axes = plt.subplots(
        1, len(FRONTIER_SPANS), figsize=(6 * len(FRONTIER_SPANS), 4), sharey=True, squeeze=False
    )
    for axis, span in zip(axes[0], FRONTIER_SPANS):
        for optimizer in [name for name in FRONTIER if name in set(swept["optimizer"])]:
            done = swept[(swept["optimizer"] == optimizer) & (swept["status"] == "finished")]
            color = palette[optimizer]
            axis.plot(
                done[span],
                done["val_loss"],
                marker="o",
                fillstyle="none",
                color=color,
                label=optimizer,
            )
            for _, run in done.iterrows():
                axis.annotate(
                    f"{run['rate']:g}",
                    (run[span], run["val_loss"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    size=7,
                    color=color,
                )
            if len(done):
                winner = done.loc[done["val_loss"].idxmin()]
                axis.scatter([winner[span]], [winner["val_loss"]], color=color, zorder=3)
        axis.set_xlabel(f"{span}, integrated")
    axes[0, 0].set_ylabel("final val_loss")
    legend(axes[0, 0])
    return save_figure(out_dir, "frontier")


def plot_geometry(table: pd.DataFrame, *, palette: dict[str, str], out_dir: Path) -> Path | None:
    """The first mixture's geometry against its span: per run, the integrated similarity of the router's inputs across tokens against the integrated bias span, a color per arm; nothing when no run logged the similarity."""
    x, y = GEOMETRY
    values = table.dropna(subset=[x, y]) if x in table.columns else table.iloc[:0]
    if values.empty:
        return None
    _, axis = plt.subplots(figsize=(6, 4.5))
    for arm_name, runs in values.groupby("arm"):
        axis.scatter(runs[x], runs[y], color=palette[str(arm_name)], label=arm_name)
    axis.set_xlabel(f"{x}, integrated")
    axis.set_ylabel(f"{y}, integrated")
    legend(axis)
    return save_figure(out_dir, "geometry")


def palette_of(names: list[str]) -> dict[str, str]:
    """A color per name: matplotlib's ten, or its twenty once there are more names than the ten, which would repeat (the RQ3 block has eleven arms)."""
    colors = matplotlib.colormaps["tab10" if len(names) <= 10 else "tab20"]
    return {name: to_hex(colors(index % colors.N)) for index, name in enumerate(names)}


def plot(
    runs_dir: Path, out_dir: Path, tables: dict[str, pd.DataFrame] | None = None
) -> list[Path]:
    """Every figure of the plan the runs support, from the runs and their endpoint ``tables`` (by default the equal-tokens one, collected here): one per curve with a line per arm and its seed band, the no-balancer arms' usage perplexity on its own, one more per mixture reading with a panel per mixture, the early window, the C6 scatters, the exploratory grid, the strips per seed, the geometry figure, the primary endpoint's two views when the matched-loss table is given, and the specialization figure; returns the files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = run_dirs(runs_dir)
    tables = tables or {"tokens": collect(runs_dir)}
    settings = {d: json.loads((d / "settings.json").read_text(encoding="utf-8")) for d in dirs}
    arms = {run_dir: arm_of(s) for run_dir, s in settings.items()}
    palette = palette_of(sorted(set(arms.values())))
    readings = {run_dir: routing_frame(run_dir) for run_dir in dirs}  # parsed once per run
    frames = {d: per_step(r, group_frame(d)) for d, r in readings.items()}
    files: list[Path] = []
    for file, names in CURVES.items():
        present = [run_dir for run_dir in dirs if (run_dir / file).exists()]
        for name in names if present else []:
            curves = (
                series_of(frames, name)
                if file == "routing.jsonl"
                else {run_dir: curve(run_dir, file, name) for run_dir in present}
            )
            if all(series.empty for series in curves.values()):  # logged before the reading
                continue
            _, axis = plt.subplots(figsize=(7, 4))
            plot_arms(axis, curves, arms=arms, palette=palette)
            axis.set_xlabel("mixture" if file in BY_MIXTURE else "step")
            axis.set_ylabel(name)
            legend(axis)
            files.append(save_figure(out_dir, name))
            if file == "routing.jsonl":
                blocks = plot_blocks(readings, name, arms=arms, palette=palette, out_dir=out_dir)
                files.extend([blocks] if blocks else [])
    unbalanced = {
        d: s for d, s in series_of(frames, UNBALANCED_CURVE).items() if not settings[d]["balanced"]
    }
    if unbalanced:
        _, axis = plt.subplots(figsize=(7, 4))
        plot_arms(axis, unbalanced, arms=arms, palette=palette)
        axis.set_xlabel("step")
        axis.set_ylabel(f"{UNBALANCED_CURVE}, no-balancer arms")
        legend(axis)
        files.append(save_figure(out_dir, f"{UNBALANCED_CURVE}_unbalanced"))
    ends = {d: int(s["warmup_steps"] + EARLY * s["steps"]) for d, s in settings.items()}
    warmups = {d: int(s["warmup_steps"]) for d, s in settings.items()}
    early = plot_early(frames, ends=ends, arms=arms, palette=palette, out_dir=out_dir)
    files.extend([early] if early else [])
    files.extend(plot_scatter(dirs, warmups=warmups, arms=arms, palette=palette, out_dir=out_dir))
    grid = plot_grid(frames, arms=arms, palette=palette, out_dir=out_dir)
    files.extend([grid] if grid else [])
    files.extend(plot_strips(tables["tokens"], palette=palette, out_dir=out_dir))
    geometry = plot_geometry(tables["tokens"], palette=palette, out_dir=out_dir)
    files.extend([geometry] if geometry else [])
    if "loss" in tables:
        files.append(plot_views(tables, palette=palette, out_dir=out_dir))
    files.extend(plot_specialization(dirs, arms=arms, palette=palette, out_dir=out_dir))
    return files


def views_table(
    table: pd.DataFrame, cuts: dict[tuple[str, str], pd.DataFrame], reference: str
) -> pd.DataFrame:
    """The matched-loss table the views figure draws: every arm's runs as cut in its pair with the reference arm, and the reference arm's as cut in its pair with the other arm of the ``HEADLINE`` pair, or whole without one."""
    other = next(name for name in HEADLINE if name != reference)
    rows = [cut[cut["arm"] != reference] for pair, cut in cuts.items() if reference in pair]
    headline = cuts.get(pair_key(other, reference), table)
    return pd.concat([headline[headline["arm"] == reference], *rows], ignore_index=True)


def report(
    runs_dir: Path,
    *,
    reference: str = REFERENCE_ARM,
    out_dir: Path | None = None,
    other_tier: Path | None = None,
    tuning_dir: Path | None = None,
) -> pd.DataFrame:
    """Write the endpoints at equal tokens and, pair of arms by pair, at matched loss, the run inventory, the per-arm summary, the annex readings' summary on their own when any run has them, every comparison with the reference arm under both matchings, over the seeds where both runs finished, the contrasts of the plan's claims with the claims' verdicts, and the plots under ``out_dir`` (by default ``runs_dir/analysis``), with the confirmation figure when the runs of the ``other_tier`` are given and the frontier with its table when the tuning block's ``tuning_dir`` is; print the claims and return the equal-tokens table."""
    out_dir = out_dir or runs_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = collect(runs_dir)
    table.to_csv(out_dir / "endpoints.csv", index=False)
    table[INVENTORY].to_csv(out_dir / "inventory.csv", index=False)
    summary = summarize(table)
    summary.to_csv(out_dir / "summary.csv")
    annex = [name for name in CURVES[ANNEX] if name in table.columns]
    if annex:  # the appendix's table: the annex readings per arm, for the arms that have them
        summarize(table.dropna(subset=annex, how="all"))[annex].to_csv(out_dir / "annex.csv")
    tables = {"tokens": table}
    arms = set(table["arm"])
    others = sorted(arms - {reference}) if reference in arms else []
    pairs = {pair_key(name, reference) for name in others}
    at = set(table["tier"])
    pairs |= {pair_key(*p) for c in CLAIMS if "loss" in c.views for p in c.pairs(arms, at)}
    memo: Memo = {}
    cuts = {pair: matched(table, runs_dir, pair, memo=memo) for pair in sorted(pairs)}
    cuts = {pair: cut for pair, cut in cuts.items() if not cut.empty}  # the pairs that share a seed
    if cuts:
        pd.concat(cuts, names=["first", "second", None]).reset_index(["first", "second"]).to_csv(
            out_dir / "endpoints_matched.csv", index=False
        )
    ended = {name: finished_pairs(table, (name, reference)) for name in others}
    compared = [name for name in others if pair_key(name, reference) in cuts and len(ended[name])]
    if compared:
        views = {"tokens": {name: ended[name] for name in compared}}
        views["loss"] = {
            name: finished_pairs(cuts[pair_key(name, reference)], (name, reference))
            for name in compared
        }
        comparisons = pd.concat(
            {
                matching: pd.concat(
                    [compare(t, n, reference, names=endpoint_names(t)) for n, t in by_arm.items()]
                )
                for matching, by_arm in views.items()
            },
            names=["matching", None],
        ).reset_index(level="matching")
        comparisons.to_csv(out_dir / "comparisons.csv", index=False)
        tables["loss"] = views_table(table, cuts, reference)
    decided = contrasts(table, cuts)
    if len(decided):
        decided.to_csv(out_dir / "contrasts.csv", index=False)
        claims(decided).to_csv(out_dir / "claims.csv", index=False)
        print(claims(decided).to_string(index=False))
    plot(runs_dir, out_dir, tables)
    if other_tier is not None:
        other = collect(other_tier)
        tiers = {runs_dir.name: table, other_tier.name: other}
        arms = sorted(set(table["arm"]) | set(other["arm"]))
        palette = palette_of(arms)
        plot_tiers(tiers, palette=palette, out_dir=out_dir)
    if tuning_dir is not None:
        swept = sweeps(tuning_dir)
        swept.to_csv(out_dir / "frontier.csv", index=False)
        names = sorted(set(table["arm"]) | set(swept["optimizer"]))
        palette = palette_of(names)
        plot_frontier(swept, palette=palette, out_dir=out_dir)
    print(summary.T.to_string())  # endpoints down, arms across: readable in a terminal
    return table


if __name__ == "__main__":
    tyro.cli(report)
