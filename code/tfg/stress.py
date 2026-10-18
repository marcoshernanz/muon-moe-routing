"""The stress protocol's analysis: rate sensitivity, spikes, and activation growth per family.

Under RQ2 every partition is trained at multiples of its tuned rates (``lr_multiplier`` in ``tfg.train``, on a constant schedule after the warmup), several seeds per multiple. A family is the arm at multiplier one, the unstressed run: the partition with the balancer and router-rate settings it was run with; the control arm whose balancer rate scales with the multiplier (``balancer_multiplier``) is a family of its own. This module reads those runs' endpoints (``tfg.analysis``) and writes, per family against the multiplier, the curves the plan asks for: the final loss, capped at the initial one so a diverged run counts as its start, whose mean height above the family's best multiplier, over the plan's multipliers, is Wortsman's rate sensitivity; the spikes per ten thousand steps under both definitions; the growth of the activations, the largest residual, attention logit, router logit, and expert output logged in the run; and the load churn in the warmup window, early, and over the run, which is where the router's optimizer shows first. Every primary spike also goes to an event bank: spikes on consecutive steps make one episode, listed with its peak, and the routing readings logged within ``TRACE_STEPS`` of the peak come along with their offset from it, so the routing signature around the events can be read off directly; the onsets table says, per episode and reading, how many steps before or after the peak the reading first left its baseline, which is the order of events the plan asks about, and the same around a placebo step ``PLACEBO`` before each peak, where there is no event, which tells a reading that drifts from one that answers the event.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from tfg.analysis import (
    SETTINGS,
    SPIKE_SIGMAS,
    collect,
    metrics_frame,
    routing_frame,
    save_figure,
    spike_table,
)
from tfg.program import MULTIPLIERS
from tfg.train import RUNS_DIR, arm

matplotlib.use("Agg")  # draw to files only, so the analysis runs on a headless machine

TRACE_STEPS = 200  # the readings logged within this many steps of a peak make its trace
LARGE_EXCESS = 0.5  # the plan lists every episode whose peak exceeds the median by this much
ACTIVATIONS = [  # the growth curves
    "residual_max",
    "attention_logit_max",
    "router_logit_max",
    "expert_output_max",
]
CAPPED = "loss_capped"  # the final validation loss, at most the initial training loss
FIGURES = {
    "sensitivity": [CAPPED],
    "spikes": ["spikes_per_10k", "spikes_raw_per_10k"],
    "activations": [f"{name}_peak" for name in ACTIVATIONS],
    "churn": ["load_churn_warmup", "load_churn_early", "load_churn"],
}
KEYS = ["run", "family", "lr_multiplier", "seed"]  # what identifies a run in the event bank
TRACED = [
    "loss",
    "grad_norm",
    "bias_span",
    "load_churn",
    "router_entropy",
    "residual_max",
    "expert_output_max",
    "expert_kurtosis_max",
]
TRACE_ALPHA = 0.5  # of each episode's line in the traces figure
BASELINE = (-TRACE_STEPS, -TRACE_STEPS // 2)  # the offsets from a peak a reading's baseline spans
PLACEBO = 2 * TRACE_STEPS  # this many steps before a peak the onsets are read again, with no event


def families(frame: pd.DataFrame) -> list[str]:
    """The family of every run: its arm at multiplier one."""
    return [
        arm(
            str(partition),
            balanced=bool(balanced),
            router_lr_multiplier=float(router_multiplier),
            balancer_multiplier=float(balancer_multiplier),
            adamw_beta2=float(beta2),
        )
        for partition, balanced, router_multiplier, balancer_multiplier, beta2 in zip(
            frame["partition"],
            frame["balanced"],
            frame["router_lr_multiplier"],
            frame["balancer_multiplier"],
            frame["adamw_beta2"],
        )
    ]


def peaks(run_dir: Path) -> dict[str, float]:
    """The largest value of each activation reading logged in the run, over every mixture."""
    readings = routing_frame(run_dir)  # empty for a run that ended before its first reading
    return {
        f"{name}_peak": float(readings[name].max()) if name in readings.columns else np.nan
        for name in ACTIVATIONS
    }


def table(runs_dir: Path) -> pd.DataFrame:
    """The endpoints of every run (``tfg.analysis``) with its family first, the activation peaks, and the final validation loss capped at the initial training loss, which a run still under way does not have yet."""
    frame = collect(runs_dir)
    frame.insert(0, "family", families(frame))
    capped = frame["val_loss"].where(
        frame["val_loss"] < frame["loss_initial"], frame["loss_initial"]
    )
    frame[CAPPED] = capped.where(frame["status"] != "unfinished")  # a diverged run: its start
    peaked = pd.DataFrame([peaks(runs_dir / str(run)) for run in frame["run"]])
    return pd.concat([frame, peaked], axis=1)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean, spread, and count over seeds of every endpoint per family and multiplier."""
    names = [name for name in frame.columns if name not in SETTINGS and name != "family"]
    return frame.groupby(["family", "lr_multiplier"])[names].agg(["mean", "std", "count"])


def sensitivity(frame: pd.DataFrame) -> pd.Series:
    """Wortsman's rate sensitivity per family: the mean over the multipliers of how far the capped loss, averaged over seeds, sits above the best multiplier's. Over the plan's grid and nothing else, so that a family that went one multiplier further stays comparable with one that did not; a family without an ended run at every multiplier of the grid has none, which leaves out the control arm's, run at a single multiplier, and every family of a block that runs part of the grid."""
    losses = frame.groupby(["family", "lr_multiplier"])[CAPPED].mean().unstack("lr_multiplier")
    losses = losses.reindex(columns=list(MULTIPLIERS))  # [family, multiplier]
    above = losses.sub(losses.min(axis=1), axis=0)
    return above.mean(axis=1, skipna=False).dropna().rename("sensitivity")


def episodes(run_dir: Path) -> pd.DataFrame:
    """The run's primary spikes as episodes, consecutive spike steps merged: the first and last step, the peak step with its loss and excess, and whether the peak is a large one."""
    per_step = spike_table(run_dir)
    spiking = per_step[per_step["spike"]]
    steps = spiking.index.to_series()
    episode = (np.diff(spiking.index, prepend=-2) > 1).cumsum()  # a gap starts a new one
    frame = pd.DataFrame(
        {
            "start": steps.groupby(episode).min(),
            "end": steps.groupby(episode).max(),
            "step": spiking.groupby(episode)["excess"].idxmax(),
        }
    ).reset_index(drop=True)
    frame["loss"] = per_step.loc[frame["step"], "loss"].to_numpy()
    frame["excess"] = per_step.loc[frame["step"], "excess"].to_numpy()
    frame["large"] = frame["excess"] > LARGE_EXCESS
    return frame


def placebos(found: pd.DataFrame, *, warmup: int) -> pd.Series:
    """The step ``PLACEBO`` before each episode's peak, kept where an episode could have been, after the warmup and a trace's length into the run, and where the trace around it holds no episode of the run: the onsets read there are those of the readings that rise with no event at all, by drifting."""
    steps = found["step"] - PLACEBO
    clear = [
        step > max(warmup, TRACE_STEPS)
        and not (
            (found["start"] <= step + TRACE_STEPS) & (found["end"] >= step - TRACE_STEPS)
        ).any()
        for step in steps
    ]
    return steps[np.array(clear, dtype=bool)]


def traces(run_dir: Path, steps: pd.Series) -> pd.DataFrame:
    """The routing readings of every mixture, with the step's loss and gradient norm, logged within ``TRACE_STEPS`` of each of the ``steps``: one row per peak, logged step, and mixture, with the logged step's offset from the peak."""
    routing = routing_frame(run_dir)
    if routing.empty:  # a run that diverged before its first reading
        return pd.DataFrame(columns=["peak", "step", "offset"])
    metrics = metrics_frame(run_dir).set_index("step")[["loss", "grad_norm"]]
    readings = routing.set_index("step").join(metrics)
    windows = {int(peak): readings.loc[peak - TRACE_STEPS : peak + TRACE_STEPS] for peak in steps}
    if not windows:
        return pd.DataFrame(columns=["peak", "step", "offset", *readings.columns])
    frame = pd.concat(windows, names=["peak", "step"]).reset_index()
    frame.insert(2, "offset", frame["step"] - frame["peak"])
    return frame


def events(runs_dir: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The event bank of the runs in ``frame``: every episode, the traces around their peaks, and the traces around their placebo steps, each row prefixed with what identifies its run."""
    banks: list[pd.DataFrame] = []
    aligned: list[pd.DataFrame] = []
    calm: list[pd.DataFrame] = []
    for values in zip(*(frame[key] for key in KEYS)):
        identity = dict(zip(KEYS, values))
        run_dir = runs_dir / str(identity["run"])
        found = episodes(run_dir)
        banks.append(found.assign(**identity)[[*KEYS, *found.columns]])
        settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
        calm_steps = placebos(found, warmup=settings["warmup_steps"])
        for steps, windows in ((found["step"], aligned), (calm_steps, calm)):
            window = traces(run_dir, steps)
            windows.append(window.assign(**identity)[[*KEYS, *window.columns]])
    return (
        pd.concat(banks, ignore_index=True),
        pd.concat(aligned, ignore_index=True),
        pd.concat(calm, ignore_index=True),
    )


def onsets(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per episode of the event bank's traces and reading of ``TRACED``, the offset from the peak at which the reading, mixtures averaged, first rises above its baseline and is still above it at the next logged step: its median over the ``BASELINE`` offsets plus ``SPIKE_SIGMAS`` robust standard deviations of it there. Two readings in a row because the baseline is ten readings or so, whose deviation one noisy reading in thirty would clear by chance. Empty where it never does, where the run has no baseline that far before the peak, or where the baseline does not vary, which leaves no scale to measure a rise against. A reading whose onset comes first moved first, unless it has onsets around the placebo steps as well: then it rises with no event, and its onsets say nothing of one."""
    names = [name for name in TRACED if name in aligned.columns]
    rows: list[dict[str, object]] = []
    for values, episode in aligned.groupby([*KEYS, "peak"]) if len(aligned) else []:
        trace = episode.groupby("offset")[names].mean()  # the mixtures averaged
        before = trace[(trace.index >= BASELINE[0]) & (trace.index <= BASELINE[1])]
        after = trace[trace.index > BASELINE[1]]
        deviation = (before - before.median()).abs().median() * 1.4826  # robust, per reading
        deviation = deviation.where(deviation > 0)  # a flat baseline gives no scale: never risen
        risen = after.gt(before.median() + SPIKE_SIGMAS * deviation)  # all False with no baseline
        risen &= risen.shift(-1, fill_value=False)  # and still risen at the next logged step
        identity: dict[str, object] = dict(zip([*KEYS, "peak"], values))
        for name in names:
            onset = risen.index[risen[name]].min() if risen[name].any() else None
            rows.append(identity | {"reading": name, "onset": onset})
    return pd.DataFrame(rows, columns=[*KEYS, "peak", "reading", "onset"])


def plot(summary: pd.DataFrame, sensitivities: pd.Series, out_dir: Path) -> list[Path]:
    """One figure per curve family with a panel per endpoint against the multiplier and a line per family, seeds averaged with their spread; returns the files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for name, endpoints in FIGURES.items():
        present = [e for e in endpoints if e in summary.columns.get_level_values(0)]
        if not present:
            continue
        _, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4), squeeze=False)
        for axis, endpoint in zip(axes[0], present):
            for family in summary.index.get_level_values("family").unique():
                rows = summary.xs(family, level="family")  # against the multiplier
                label = str(family)
                if name == "sensitivity" and family in sensitivities.index:
                    label += f" (sensitivity {sensitivities[family]:.3f})"
                axis.errorbar(
                    rows.index,
                    rows[(endpoint, "mean")],
                    yerr=rows[(endpoint, "std")].fillna(0.0),
                    marker="o",
                    capsize=3,
                    label=label,
                )
            axis.set_xscale("log", base=2)
            axis.set_xlabel("rate multiplier")
            axis.set_ylabel(endpoint)
            axis.legend(fontsize="small")
        files.append(save_figure(out_dir, name))
    return files


def plot_traces(aligned: pd.DataFrame, out_dir: Path) -> Path | None:
    """The event bank's traces aligned on the peak: one panel per reading of ``TRACED`` the bank has, a line per episode (mixtures averaged) against the offset from its peak, colored by family; nothing when there was no episode."""
    names = [name for name in TRACED if name in aligned.columns]
    if aligned.empty or not names:
        return None
    _, axes = plt.subplots(1, len(names), figsize=(3.5 * len(names), 3.5), squeeze=False)
    colors = {f: f"C{i}" for i, f in enumerate(sorted(set(aligned["family"].astype(str))))}
    for axis, name in zip(axes[0], names):
        seen: set[str] = set()
        for (family, _run, _peak), rows in aligned.groupby(["family", "run", "peak"]):
            trace = rows.groupby("offset")[name].mean()  # the mixtures averaged
            label = str(family) if family not in seen else None
            seen.add(str(family))
            axis.plot(
                trace.index,
                trace.to_numpy(),
                color=colors[str(family)],
                alpha=TRACE_ALPHA,
                label=label,
            )
        axis.axvline(0, color="black", linewidth=0.5)
        axis.set_xlabel("steps from the peak")
        axis.set_ylabel(name)
    axes[0, 0].legend(fontsize="x-small")
    return save_figure(out_dir, "traces")


def report(runs_dir: Path = RUNS_DIR, *, out_dir: Path | None = None) -> pd.DataFrame:
    """Write the stress table (one row per run), its summary per family and multiplier, the sensitivity per family, the curves, the event bank (``events.csv``, ``traces.csv``, and ``onsets.csv``), and the traces figure under ``out_dir``, by default ``runs_dir/stress``; print and return the table."""
    out_dir = out_dir or runs_dir / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = table(runs_dir)
    frame.to_csv(out_dir / "stress.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out_dir / "summary.csv")
    sensitivities = sensitivity(frame)
    sensitivities.to_csv(out_dir / "sensitivity.csv")
    plot(summary, sensitivities, out_dir)
    bank, aligned, calm = events(runs_dir, frame)
    bank.to_csv(out_dir / "events.csv", index=False)
    aligned.to_csv(out_dir / "traces.csv", index=False)
    found = [onsets(aligned).assign(placebo=False), onsets(calm).assign(placebo=True)]
    pd.concat(found, ignore_index=True).to_csv(out_dir / "onsets.csv", index=False)
    plot_traces(aligned, out_dir)
    shown = [CAPPED, *FIGURES["spikes"], *FIGURES["activations"]]
    print(summary[[(name, "mean") for name in shown]].to_string())
    print(sensitivities.to_string())
    print(bank.to_string(index=False) if not bank.empty else "no spikes")
    return frame


def main(args: list[str] | None = None) -> pd.DataFrame:
    """The command line: ``python -m tfg.stress --runs-dir runs/rq2``."""
    return tyro.cli(report, args=args)


if __name__ == "__main__":
    main()
