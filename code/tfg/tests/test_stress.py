"""Tests of the stress protocol's analysis: families, sensitivity, episodes, traces, report."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from tfg.stress import (
    CAPPED,
    TRACE_STEPS,
    episodes,
    events,
    main,
    onsets,
    placebos,
    sensitivity,
    table,
    traces,
)
from tfg.tests.synthetic import write_run


def noisy(steps: int, bumps: dict[int, float]) -> list[float]:
    """A loss of 4.0 with a noise of ±0.05 in a period of four (two up, two down), plus the bumps at the given steps; bumping steps 1 or 2 modulo 4 keeps the median at 4.0."""
    losses = [4.0 + (0.05 if step % 4 in (1, 2) else -0.05) for step in range(1, steps + 1)]
    for step, bump in bumps.items():
        assert step % 4 in (1, 2)
        losses[step - 1] += bump
    return losses


def write_grid(runs_dir: Path) -> None:
    """Three families: AdamW at 1x, 2x, and 4x with two seeds, where the 4x runs spike and one of them diverges, Muon at 1x and 2x with one seed, and Muon's control arm at 2x, whose balancer runs twice as fast."""
    for seed in range(2):
        write_run(runs_dir, "adamw", seed, spans=[0.2] * 12, val_losses=[2.0 + 0.1 * seed])
        write_run(
            runs_dir,
            "adamw",
            seed,
            spans=[0.2] * 12,
            lr_multiplier=2.0,
            val_losses=[2.5 + 0.1 * seed],
        )
    write_run(
        runs_dir,
        "adamw",
        0,
        spans=[0.2] * 12,
        lr_multiplier=4.0,
        val_losses=[3.0],
        losses=noisy(120, {30: 0.2, 61: 1.0, 62: 0.8, 90: 0.4}),  # over the floor; both twice
    )
    write_run(
        runs_dir,
        "adamw",
        1,
        spans=[0.2] * 12,
        lr_multiplier=4.0,
        val_losses=[math.nan],  # diverged
        losses=noisy(120, {50: 2.0}),
    )
    write_run(runs_dir, "muon", 0, spans=[0.2] * 12, val_losses=[1.8])
    write_run(runs_dir, "muon", 0, spans=[0.2] * 12, lr_multiplier=2.0, val_losses=[1.9])
    write_run(
        runs_dir,
        "muon",
        0,
        spans=[0.2] * 12,
        lr_multiplier=2.0,
        balancer_multiplier=2.0,
        val_losses=[1.7],
    )


def test_the_table_names_the_family_and_caps_the_loss_of_a_diverged_run(tmp_path: Path) -> None:
    write_grid(tmp_path)
    frame = table(tmp_path).set_index("run")
    assert list(frame.columns[:2]) == ["family", "tier"]
    assert frame["family"].to_dict() == {
        "adamw-lr2x-s0": "adamw",
        "adamw-lr2x-s1": "adamw",
        "adamw-lr4x-s0": "adamw",
        "adamw-lr4x-s1": "adamw",
        "adamw-s0": "adamw",
        "adamw-s1": "adamw",
        "muon-lr2x-balancer2x-s0": "muon-balancer2x",  # the control arm: its own family
        "muon-lr2x-s0": "muon",
        "muon-s0": "muon",
    }
    assert frame.loc["adamw-lr4x-s0", CAPPED] == 3.0  # below its initial 4.05
    assert frame.loc["adamw-lr4x-s1", CAPPED] == 4.05  # diverged: its initial loss
    assert frame.loc["adamw-s0", CAPPED] == 2.0
    assert (
        frame.loc["adamw-lr4x-s0", "spikes"] == 3 and frame.loc["adamw-lr4x-s0", "spikes_raw"] == 4
    )
    assert frame.loc["adamw-s0", "spikes"] == 0 and frame.loc["adamw-s0", "spikes_raw"] == 0
    per_10k = 10_000 / 110  # of the 110 steps after the warmup
    assert frame.loc["adamw-lr4x-s0", "spikes_per_10k"] == pytest.approx(3 * per_10k)
    assert frame.loc["adamw-lr4x-s0", "spikes_raw_per_10k"] == pytest.approx(4 * per_10k)
    assert frame.loc["muon-s0", "residual_max_peak"] == pytest.approx(3.1)  # 1 + 1.1 + 1
    assert frame.loc["muon-s0", "expert_output_max_peak"] == pytest.approx(4.1)
    assert frame.loc["muon-s0", "attention_logit_max_peak"] == pytest.approx(6.1)


def test_the_sensitivity_is_the_mean_height_above_the_best_multiplier(tmp_path: Path) -> None:
    write_grid(tmp_path)
    assert sensitivity(table(tmp_path)).empty  # the plan's grid goes to 8x, and no family has
    for seed in range(2):
        write_run(tmp_path, "adamw", seed, spans=[0.2] * 12, lr_multiplier=8.0, val_losses=[3.9])
    values = sensitivity(table(tmp_path))
    # AdamW, seeds averaged: 2.05 at 1x, 2.55 at 2x, (3.0 + 4.05) / 2 at 4x, 3.9 at 8x; the best is 1x.
    assert values["adamw"] == pytest.approx((0.0 + 0.5 + 1.475 + 1.85) / 4)
    assert "muon" not in values and "muon-balancer2x" not in values  # short of the grid so far
    write_run(tmp_path, "muon", 0, spans=[0.2] * 12, lr_multiplier=4.0, val_losses=[2.0])
    write_run(tmp_path, "muon", 0, spans=[0.2] * 12, lr_multiplier=8.0, val_losses=[2.4])
    values = sensitivity(table(tmp_path))
    assert values["muon"] == pytest.approx((0.0 + 0.1 + 0.2 + 0.6) / 4)
    write_run(tmp_path, "muon", 0, spans=[0.2] * 12, lr_multiplier=16.0, val_losses=[3.9])
    further = sensitivity(table(tmp_path))  # a family that went past the plan's grid
    assert further["muon"] == values["muon"] and further["adamw"] == values["adamw"]


def test_the_sensitivity_leaves_out_what_has_not_covered_the_grid(tmp_path: Path) -> None:
    write_grid(tmp_path)
    for seed in range(2):
        write_run(tmp_path, "adamw", seed, spans=[0.2] * 12, lr_multiplier=8.0, val_losses=[3.9])
    write_run(tmp_path, "muon", 0, spans=[0.2] * 12, lr_multiplier=8.0, val_losses=[2.4])
    values = sensitivity(table(tmp_path))
    assert list(values.index) == ["adamw"]  # Muon has no 4x run, its control arm one multiplier
    part = table(tmp_path)
    assert sensitivity(part[part["lr_multiplier"].isin([2.0, 4.0])]).empty  # a block of 2x and 4x
    under_way = write_run(tmp_path, "muon", 0, spans=[0.2] * 6, lr_multiplier=4.0)
    settings = json.loads((under_way / "settings.json").read_text(encoding="utf-8"))
    (under_way / "settings.json").write_text(json.dumps(settings | {"steps": 120}))
    frame = table(tmp_path).set_index("run")
    assert frame.loc["muon-lr4x-s0", "status"] == "unfinished"
    assert frame[CAPPED].isna().to_dict()["muon-lr4x-s0"]  # its loss so far is no final loss
    assert list(sensitivity(frame.reset_index()).index) == ["adamw"]
    (under_way / "settings.json").write_text(json.dumps(settings))  # and now it has ended
    assert sensitivity(table(tmp_path))["muon"] == pytest.approx((0.0 + 0.1 + 0.2 + 0.6) / 4)


def test_episodes_merge_consecutive_spikes_and_keep_the_peak(tmp_path: Path) -> None:
    write_grid(tmp_path)
    found = episodes(tmp_path / "adamw-lr4x-s0")
    assert found[["start", "end", "step"]].to_numpy().tolist() == [[61, 62, 61], [90, 90, 90]]
    assert found["excess"].tolist() == pytest.approx([1.05, 0.45])
    assert found["loss"].tolist() == pytest.approx([5.05, 4.45])
    assert found["large"].tolist() == [True, False]
    assert episodes(tmp_path / "adamw-s0").empty
    assert list(episodes(tmp_path / "adamw-s0").columns) == [
        "start",
        "end",
        "step",
        "loss",
        "excess",
        "large",
    ]


def test_traces_align_the_readings_logged_around_a_peak(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2] * 50)  # 500 steps, logged every 10
    frame = traces(run_dir, pd.Series([100]))
    assert len(frame) == 30 * 2  # steps 10 to 300, two mixtures each
    assert list(frame.columns[:4]) == ["peak", "step", "offset", "mixture"]
    assert frame["offset"].min() == -90 and frame["offset"].max() == TRACE_STEPS
    assert (frame["offset"] == frame["step"] - 100).all() and (frame["peak"] == 100).all()
    assert {"bias_span", "residual_max", "loss", "grad_norm"} <= set(frame.columns)
    assert frame.loc[frame["step"] == 10, "residual_max"].tolist() == [1.0, 2.0]
    assert frame.loc[frame["step"] == 20, "loss"].tolist() == [4.0, 4.0]
    two = traces(run_dir, pd.Series([100, 400]))
    assert len(two) == 30 * 2 + 31 * 2  # steps 200 to 500 around the second
    assert two["peak"].unique().tolist() == [100, 400]
    empty = traces(run_dir, pd.Series([], dtype=int))
    assert empty.empty and list(empty.columns[:4]) == ["peak", "step", "offset", "mixture"]


def test_the_event_bank_prefixes_every_row_with_its_run(tmp_path: Path) -> None:
    write_grid(tmp_path)
    bank, aligned, calm = events(tmp_path, table(tmp_path))
    assert calm.empty  # the peaks are too early in these runs for a placebo step before them
    assert list(bank.columns[:4]) == ["run", "family", "lr_multiplier", "seed"]
    assert bank["run"].tolist() == ["adamw-lr4x-s0", "adamw-lr4x-s0", "adamw-lr4x-s1"]
    assert bank["step"].tolist() == [61, 90, 50] and bank["large"].tolist() == [True, False, True]
    assert (bank["lr_multiplier"] == 4.0).all()
    assert aligned["run"].value_counts().to_dict() == {"adamw-lr4x-s0": 48, "adamw-lr4x-s1": 24}
    assert aligned.loc[aligned["peak"] == 50, "seed"].unique().tolist() == [1]


def test_onsets_give_the_offset_where_a_reading_first_leaves_its_baseline() -> None:
    offsets = list(range(-200, 201, 10))
    churn = [0.1 + 0.001 * (i % 3) for i in range(len(offsets))]  # a baseline with some noise
    span = churn.copy()
    entropy = churn.copy()
    for index, offset in enumerate(offsets):
        churn[index] += 0.5 if offset >= -30 else 0.0  # the churn rises 30 steps before the peak
        span[index] += 0.5 if offset >= 20 else 0.0  # the span 20 steps after it
        span[index] += 0.5 if offset == -60 else 0.0  # after one reading alone that does not count
        entropy[index] = 3.0 + (1e-9 if offset >= -50 else 0.0)  # a flat baseline: nothing to clear
    rows = [
        {"run": "r", "family": "f", "lr_multiplier": 4.0, "seed": 0, "peak": 1000}
        | {"offset": offset, "mixture": mixture, "load_churn": c, "bias_span": b, "loss": 4.0}
        | {"router_entropy": e}
        for offset, c, b, e in zip(offsets, churn, span, entropy)
        for mixture in range(2)
    ]
    found = onsets(pd.DataFrame(rows)).set_index("reading")
    assert found.loc["load_churn", "onset"] == -30 and found.loc["bias_span", "onset"] == 20
    assert pd.isna(found.loc["loss", "onset"])  # flat: it never leaves
    assert pd.isna(found.loc["router_entropy", "onset"])  # nor does a wobble on a flat baseline
    assert found.loc["load_churn", "peak"] == 1000 and found.loc["load_churn", "run"] == "r"
    frame = pd.DataFrame(rows)
    late = frame[frame["offset"] > -100]  # a peak early in the run
    assert onsets(late)["onset"].isna().all()  # no baseline to leave
    assert onsets(pd.DataFrame(columns=["run", "peak", "offset"])).empty


def test_a_placebo_step_tells_a_reading_that_drifts_from_one_that_answers(tmp_path: Path) -> None:
    drift = [0.2 + 0.002 * t + 0.0005 * (t % 3) for t in range(100)]  # the span creeps up all along
    write_run(tmp_path, "adamw", 0, spans=drift, lr_multiplier=4.0, losses=noisy(1000, {801: 1.0}))
    bank, aligned, calm = events(tmp_path, table(tmp_path))
    assert bank["step"].tolist() == [801] and calm["peak"].unique().tolist() == [401]
    real, placebo = (onsets(frame).set_index("reading")["onset"] for frame in (aligned, calm))
    assert real["bias_span"] == placebo["bias_span"] == 39  # a rise with and without the event
    assert pd.isna(real["loss"]) and pd.isna(placebo["loss"])
    found = episodes(tmp_path / "adamw-lr4x-s0")
    assert placebos(found, warmup=10).tolist() == [401]
    assert placebos(found, warmup=500).empty  # no episode is counted inside the warmup either
    twice = pd.concat([found.assign(start=401, end=401, step=401), found], ignore_index=True)
    assert placebos(
        twice, warmup=10
    ).empty  # one too early in the run, the other with an episode in its trace


def write_early_divergence(runs_dir: Path) -> Path:
    """Muon at 4x, ended at step 2 on a loss that is no number, before any routing reading."""
    run_dir = write_run(runs_dir, "muon", 0, spans=[0.2] * 12, lr_multiplier=4.0)
    record: dict[str, object] = {
        "step": 2,
        "loss": math.nan,
        "lr_factor": 1.0,
        "grad_norm": math.nan,
    }
    record |= {"grad_norms": {"adamw": [0.5, math.nan]}, "losses": [4.0, math.nan]}
    (run_dir / "metrics.jsonl").write_text(json.dumps(record | {"val_loss": math.nan}) + "\n")
    (run_dir / "routing.jsonl").write_text("")
    return run_dir


def test_a_run_that_diverged_before_its_first_reading_has_no_traces(tmp_path: Path) -> None:
    run_dir = write_early_divergence(tmp_path)
    assert episodes(run_dir).empty and traces(run_dir, pd.Series([2])).empty
    frame = table(tmp_path)
    assert frame["status"].tolist() == ["diverged"] and frame[CAPPED].tolist() == [4.0]
    assert frame["spikes_per_10k"].isna().all()  # no step after the warmup: no rate, not a zero
    bank, aligned, calm = events(tmp_path, frame)
    assert bank.empty and aligned.empty and calm.empty and onsets(aligned).empty


def test_the_report_writes_the_tables_the_curves_and_the_bank(tmp_path: Path) -> None:
    write_grid(tmp_path)
    write_early_divergence(tmp_path)
    for name, seeds, loss in (("adamw", 2, 3.9), ("muon", 1, 2.4)):  # to the end of the plan's grid
        for seed in range(seeds):
            write_run(tmp_path, name, seed, spans=[0.2] * 12, lr_multiplier=8.0, val_losses=[loss])
    frame = main(["--runs-dir", str(tmp_path)])
    out_dir = tmp_path / "stress"
    figures = ("sensitivity", "spikes", "activations", "churn", "traces")
    assert {p.name for p in out_dir.iterdir()} == {
        "stress.csv",
        "summary.csv",
        "sensitivity.csv",
        "events.csv",
        "traces.csv",
        "onsets.csv",
        *(f"{name}.{extension}" for name in figures for extension in ("png", "pdf")),
    }
    assert len(frame) == 13 and len(pd.read_csv(out_dir / "stress.csv")) == 13
    summary = pd.read_csv(out_dir / "summary.csv", header=[0, 1], index_col=[0, 1])
    assert summary.index.tolist() == [
        ("adamw", 1.0),
        ("adamw", 2.0),
        ("adamw", 4.0),
        ("adamw", 8.0),
        ("muon", 1.0),
        ("muon", 2.0),
        ("muon", 4.0),
        ("muon", 8.0),
        ("muon-balancer2x", 2.0),
    ]
    assert summary.loc[("adamw", 4.0), (CAPPED, "mean")] == pytest.approx(3.525)
    assert summary.loc[("adamw", 4.0), ("spikes", "count")] == 2
    sensitivities = pd.read_csv(out_dir / "sensitivity.csv", index_col=0)["sensitivity"]
    assert sensitivities.index.tolist() == ["adamw", "muon"]  # the families with every multiplier
    assert sensitivities["muon"] == pytest.approx(
        (0.0 + 0.1 + 2.2 + 0.6) / 4
    )  # 4x: its initial 4.0
    assert len(pd.read_csv(out_dir / "events.csv")) == 3
    assert len(pd.read_csv(out_dir / "traces.csv")) == 72
    found = pd.read_csv(out_dir / "onsets.csv")
    readings = ["bias_span", "expert_output_max", "grad_norm", "load_churn", "loss", "residual_max"]
    assert sorted(set(found["reading"])) == readings  # the traced readings the runs logged
    assert len(found) == 3 * 6 and found["onset"].isna().all()  # peaks too early for a baseline
    assert not found["placebo"].any()  # and for a placebo step before them
