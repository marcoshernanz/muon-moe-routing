"""Tests of the run program on the synthetic stream at the laptop tier; a minute on a laptop."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from tfg import program
from tfg.analysis import endpoints
from tfg.program import (
    BALANCER_SEEDS,
    BETA2_CHECK,
    CLIP_FACTOR,
    HORIZON,
    RATES_FILE,
    RQ3,
    RQ3_SEEDS,
    SEEDS,
    Run,
    Settings,
    balance_loss_runs,
    balancer_runs,
    confirmation,
    evaluate,
    gate1,
    main,
    rq2,
    rq3,
    rq3_runs,
    steps_of,
    tuning,
)
from tfg.stress import table
from tfg.tests.synthetic import QUICK, write_panel, write_shards
from tfg.tier import TIERS

TOKENS = 5 * 8 * 64  # five laptop steps
RATES = {"adamw": 3e-3, "muon": 3e-3}
RATES_FLAGS = {
    "adamw_lr": RATES["adamw"],
    "muon_lr": RATES["muon"],
}  # the same, as given to a block


def settings(tmp_path: Path) -> Settings:
    return Settings(
        runs_dir=tmp_path / "runs", compiled=False, data_dir=write_shards(tmp_path), **QUICK
    )


def saved(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))


def clip(calibration: Path, kind: str) -> float:
    """The clip a family takes from its calibration run for one optimizer."""
    median = endpoints(calibration).get(f"grad_norm_median_{kind}")
    assert isinstance(median, float)
    return CLIP_FACTOR * median


def test_steps_follow_the_plan_token_counts() -> None:
    assert steps_of("mechanism", 1_000_000_000) == 15_259
    assert steps_of("mechanism", 500_000_000) == 7_629
    assert steps_of("confirmation", 3_000_000_000) == 45_776
    assert steps_of("laptop", TOKENS) == 5


def test_the_rq3_block_lists_thirty_three_runs_seed_by_seed_on_seeds_of_its_own() -> None:
    runs = rq3_runs(RQ3_SEEDS, {"adamw": 2e-3, "muon": 4e-3})
    assert len(runs) == 33 and len({run.name("mechanism") for run in runs}) == 33
    assert not set(RQ3_SEEDS) & set(SEEDS)  # no seed the first chain's runs had
    assert [run.seed for run in runs] == [seed for seed in RQ3_SEEDS for _ in range(11)]
    assert runs[:2] == [Run("adamw", 3), Run("muon", 3)]  # the headline pair first
    equal = [run for run in runs if run.muon_lr_multiplier != 1]
    assert [(run.partition, run.muon_lr_multiplier) for run in equal] == [("muon", 0.5)] * 3
    assert equal[0].name("mechanism") == "mechanism-muon-muonlr0.5x-s3"
    assert sum(not run.balanced for run in runs) == 6
    assert len(rq3_runs((3,), {"adamw": 2e-3, "muon": 2e-3})) == 10  # no arm at the rate it has


def test_the_balancer_block_lists_twelve_runs_seed_by_seed_on_seeds_of_its_own() -> None:
    runs = balancer_runs(BALANCER_SEEDS)
    names = [run.name("mechanism") for run in runs]
    assert not set(BALANCER_SEEDS) & (set(SEEDS) | set(RQ3_SEEDS))  # no seed an earlier run had
    assert names[:4] == [
        "mechanism-adamw-s6",
        "mechanism-muon-s6",
        "mechanism-adamw-balancer4x-s6",
        "mechanism-muon-balancer4x-s6",
    ]
    assert len(set(names)) == 12 and [run.seed for run in runs] == [6] * 4 + [7] * 4 + [8] * 4


def test_the_balance_loss_block_lists_rq3s_three_partitions_on_rq3s_seeds_with_the_loss() -> None:
    runs = balance_loss_runs(RQ3_SEEDS)
    names = [run.name("mechanism") for run in runs]
    assert names[:3] == [
        "mechanism-adamw-balanceloss0.0001-s3",
        "mechanism-muon-balanceloss0.0001-s3",
        "mechanism-muon_except_router-balanceloss0.0001-s3",
    ]
    assert len(set(names)) == 9 and [run.seed for run in runs] == [3] * 3 + [4] * 3 + [5] * 3
    # Each run is RQ3's run of its arm and seed but for the loss, which RQ3's runs do not have.
    references = {
        run.name("mechanism") for run in rq3_runs(RQ3_SEEDS, {"adamw": 2e-3, "muon": 4e-3})
    }
    for run in runs:
        assert replace(run, balance_loss=0.0).name("mechanism") in references


def test_tuning_sweeps_both_rates_and_the_control_then_checks_the_second_moment_decay(
    tmp_path: Path,
) -> None:
    shared = settings(tmp_path)
    rates = tuning(
        shared,
        tier="laptop",
        adamw_rates=(3e-3, 1e-5),
        muon_rates=(3e-3,),
        control_rates=(1e-3,),
        tokens=TOKENS,
    )
    block_dir = shared.runs_dir / "tuning"
    assert rates == {**RATES, "momentum": 1e-3}
    assert json.loads((block_dir / RATES_FILE).read_text(encoding="utf-8")) == rates
    assert (block_dir / "adamw" / "lr1e-05" / "checkpoint.pt").exists()
    assert (block_dir / "adamw" / "lr0.9" / "settings.json").exists()  # one point past the edge
    assert not (block_dir / "adamw" / "lr270").exists()  # and no further
    assert saved(block_dir / "muon" / "lr0.003")["adamw_lr"] == 3e-3  # held at its winner
    control = saved(block_dir / "momentum" / "lr0.001")  # swept for the frontier
    assert control["partition"] == "momentum" and control["muon_lr"] == 1e-3
    assert control["adamw_lr"] == 3e-3  # held at its winner too
    check = saved(block_dir / "laptop-adamw-beta20.98-s0")
    assert check["adamw_beta2"] == BETA2_CHECK and check["adamw_lr"] == 3e-3
    assert check["steps"] == 5 and check["schedule"] == "cosine"


def test_tuning_stops_when_a_tuned_rate_is_still_an_edge_after_one_more_point(
    tmp_path: Path,
) -> None:
    shared = settings(tmp_path)
    with pytest.raises(SystemExit, match="adamw's best rate, 0.001, is still an edge"):
        tuning(
            shared,
            tier="laptop",
            adamw_rates=(1e-5, 1e-4),  # the higher wins, and so does the point past it
            muon_rates=(3e-3,),
            control_rates=(1e-3,),
            tokens=TOKENS,
        )
    block_dir = shared.runs_dir / "tuning"
    assert (block_dir / "adamw" / "lr0.001" / "settings.json").exists()  # the one more point
    assert not (block_dir / RATES_FILE).exists()  # so no later block reads a rate
    assert not (block_dir / "muon").exists()  # and Muon's sweep does not hold AdamW at it


def test_gate_one_reads_the_tuned_rates_and_a_cut_block_resumes_without_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = settings(tmp_path)
    with pytest.raises(FileNotFoundError, match="rates.json"):
        gate1(shared, tier="laptop", seeds=(0,), tokens=TOKENS)
    tuned = {"adamw": 3e-3, "muon": 6e-3}
    (shared.runs_dir / "tuning").mkdir(parents=True)
    (shared.runs_dir / "tuning" / RATES_FILE).write_text(json.dumps(tuned), encoding="utf-8")
    first = gate1(shared, tier="laptop", seeds=(0,), tokens=TOKENS)
    assert first == [shared.runs_dir / "rq1" / f"laptop-{p}-s0" for p in ("adamw", "muon")]
    assert saved(first[1])["muon_lr"] == 6e-3

    trained: list[str] = []
    train = program.train

    def counting(**kwargs: Any) -> object:
        run_dir = kwargs["run_dir"]
        assert isinstance(run_dir, Path)
        if len(trained) == 3:  # the box goes away during the fourth run
            trained.append("cut")
            raise KeyboardInterrupt
        trained.append(run_dir.name)
        return train(**kwargs)

    monkeypatch.setattr(program, "train", counting)
    with pytest.raises(KeyboardInterrupt):
        rq3(shared, tier="laptop", seeds=(3,), tokens=TOKENS)
    assert trained == ["laptop-adamw-s3", "laptop-muon-s3", "laptop-muon_except_router-s3", "cut"]
    block_dir = shared.runs_dir / "rq3"
    written = {n: (block_dir / n / "metrics.jsonl").stat().st_mtime_ns for n in trained[:3]}
    monkeypatch.setattr(program, "train", train)
    dirs = rq3(shared, tier="laptop", seeds=(3,), tokens=TOKENS)  # relaunched
    assert all(
        (block_dir / n / "metrics.jsonl").stat().st_mtime_ns == t for n, t in written.items()
    )
    flagged = ("muon-muonlr0.5x", "adamw-unbalanced", "muon-unbalanced")
    assert [d.name for d in dirs] == [f"laptop-{arm}-s3" for arm in (*RQ3, *flagged)]
    equal = saved(block_dir / "laptop-muon-muonlr0.5x-s3")
    assert equal["muon_lr"] == 6e-3 and equal["muon_lr_multiplier"] == 0.5  # trained at 3e-3
    unbalanced = saved(block_dir / "laptop-muon-unbalanced-s3")
    assert not unbalanced["balanced"] and unbalanced["muon_lr_multiplier"] == 1
    assert all(saved(d)["steps"] == 5 and saved(d)["schedule"] == "cosine" for d in dirs)
    assert all((d / "checkpoint.pt").exists() and (d / "early.pt").exists() for d in dirs)


def test_the_stress_grid_calibrates_each_family_clips_on_its_own_run(tmp_path: Path) -> None:
    shared = settings(tmp_path)
    dirs = rq2(
        shared, tier="laptop", **RATES_FLAGS, seeds=(0,), multipliers=(1.0, 4.0), tokens=TOKENS
    )
    block_dir = shared.runs_dir / "rq2"
    assert [d.name for d in dirs] == [
        "laptop-adamw-s0",
        "laptop-adamw-lr4x-s0",
        "laptop-adamw-lr16x-s0",  # nothing diverged at the top multiplier: one more level
        "laptop-muon-s0",
        "laptop-muon-lr4x-s0",
        "laptop-muon-lr4x-balancer4x-s0",
        "laptop-muon-lr16x-s0",
        "laptop-muon_except_router-s0",
        "laptop-muon_except_router-lr4x-s0",
        "laptop-muon_except_router-lr16x-s0",
    ]
    for partition in ("adamw", "muon", "muon_except_router"):
        calibration = block_dir / "calibration" / f"laptop-{partition}-s0"
        base = saved(calibration)
        assert base["adamw_clip"] == 1.0 and base["muon_clip"] == 1.0
        assert base["schedule"] == "constant" and base["lr_multiplier"] == 1.0
        for run_dir in (d for d in dirs if d.name.startswith(f"laptop-{partition}-")):
            clipped = saved(run_dir)
            assert clipped["adamw_clip"] == clip(calibration, "adamw")
            expected = clip(calibration, "muon") if partition != "adamw" else 1.0
            assert clipped["muon_clip"] == expected and clipped["schedule"] == "constant"
    control = saved(block_dir / "laptop-muon-lr4x-balancer4x-s0")
    assert control["lr_multiplier"] == 4.0 and control["balancer_multiplier"] == 4.0
    frame = table(block_dir)  # the stress analysis sees the families, not the calibration runs
    assert sorted(set(frame["family"])) == [
        "adamw",
        "muon",
        "muon-balancer4x",
        "muon_except_router",
    ]
    assert len(frame) == 10


def test_the_stress_grid_stops_at_the_multiplier_a_family_diverged_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = settings(tmp_path)
    status_of = program.status_of

    def diverged_at_the_top(run_dir: Path) -> str:
        """As if the all-AdamW family's runs at 2x had diverged."""
        return "diverged" if run_dir.name == "laptop-adamw-lr2x-s0" else status_of(run_dir)

    monkeypatch.setattr(program, "status_of", diverged_at_the_top)
    dirs = rq2(
        shared, tier="laptop", **RATES_FLAGS, seeds=(0,), multipliers=(1.0, 2.0), tokens=TOKENS
    )
    extended = [d.name for d in dirs if "lr16x" in d.name]
    assert extended == ["laptop-muon-lr16x-s0", "laptop-muon_except_router-lr16x-s0"]
    again = rq2(
        shared,
        tier="laptop",
        **RATES_FLAGS,
        seeds=(0,),
        multipliers=(1.0, 2.0),
        extension=None,
        tokens=TOKENS,
    )
    assert not [d for d in again if "lr16x" in d.name]
    assert not [d for d in again if "balancer" in d.name]  # no 4x arm to read the control against
    assert len(again) == 6


def test_the_control_arm_has_no_say_in_whether_its_family_goes_further(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = settings(tmp_path)
    status_of = program.status_of

    def control_diverged(run_dir: Path) -> str:
        """As if the balancer-speed control had diverged, at the top multiplier as it is here."""
        return "diverged" if "balancer" in run_dir.name else status_of(run_dir)

    monkeypatch.setattr(program, "status_of", control_diverged)
    dirs = rq2(shared, tier="laptop", **RATES_FLAGS, seeds=(0,), multipliers=(4.0,), tokens=TOKENS)
    names = [d.name for d in dirs]
    assert "laptop-muon-lr4x-balancer4x-s0" in names and "laptop-muon-lr16x-s0" in names


def test_the_stress_grid_takes_no_clips_from_a_calibration_run_that_diverged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_of = program.status_of

    def diverged_in_calibration(run_dir: Path) -> str:
        return "diverged" if run_dir.parent.name == "calibration" else status_of(run_dir)

    monkeypatch.setattr(program, "status_of", diverged_in_calibration)
    with pytest.raises(RuntimeError, match="calibration run .*laptop-adamw-s0 diverged"):
        rq2(
            settings(tmp_path),
            tier="laptop",
            **RATES_FLAGS,
            seeds=(0,),
            multipliers=(1.0,),
            tokens=TOKENS,
        )
    assert not list((tmp_path / "runs" / "rq2").glob("laptop-*"))  # no family run was started


def test_the_confirmation_tier_rechecks_four_rates_then_runs_three_partitions_below_them(
    tmp_path: Path,
) -> None:
    shared = settings(tmp_path)
    dirs = confirmation(
        shared,
        tier="laptop",
        adamw_lr=6e-3,
        muon_lr=1.5e-3,
        seeds=(0,),
        recheck_tokens=TOKENS,
        tokens=2 * TOKENS,
    )
    block_dir = shared.runs_dir / "confirmation"
    grids = {
        "adamw": ["lr0.00075", "lr0.0015", "lr0.003", "lr0.006"],
        "muon": ["lr0.0001875", "lr0.000375", "lr0.00075", "lr0.0015"],
    }
    for optimizer, grid in grids.items():  # the four rates, and one more past a winning edge
        sweep_dir = block_dir / "recheck" / optimizer
        swept = json.loads((sweep_dir / "sweep.json").read_text(encoding="utf-8"))
        extension = [f"lr{swept['extension']:g}"] if swept["extension"] else []
        assert sorted(p.name for p in sweep_dir.iterdir() if p.is_dir()) == sorted(grid + extension)
        assert bool(extension) == (f"lr{swept['best']:g}" in (grid[0], grid[-1], *extension))
    rates = json.loads((block_dir / "recheck" / RATES_FILE).read_text(encoding="utf-8"))
    assert set(rates) == {"adamw", "muon"}
    assert [d.name for d in dirs] == [
        "laptop-adamw-s0",
        "laptop-muon-s0",
        "laptop-muon_except_router-s0",
    ]
    for run_dir in dirs:
        run = saved(run_dir)
        assert (
            run["adamw_lr"] == HORIZON * rates["adamw"]
        )  # the winners, for a run three times as long
        assert run["muon_lr"] == HORIZON * rates["muon"]
        assert run["steps"] == 10  # the confirmation runs are longer than the re-check


def test_evaluate_runs_the_panel_and_the_annex_once_over_finished_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = settings(tmp_path)
    panel_dir = write_panel(tmp_path)
    with pytest.raises(FileNotFoundError, match="no runs"):
        evaluate(shared, block="rq1", panel_dir=panel_dir)
    dirs = gate1(shared, tier="laptop", **RATES_FLAGS, seeds=(0,), tokens=TOKENS)
    with pytest.raises(FileNotFoundError, match="panel"):
        evaluate(shared, block="rq1", panel_dir=tmp_path / "nowhere")
    stale = saved(dirs[1]) | {"steps": 20}  # as if the second run had been cut short
    (dirs[1] / "settings.json").write_text(json.dumps(stale), encoding="utf-8")
    assert evaluate(shared, block="rq1", panel_dir=panel_dir) == dirs[:1]
    assert not (dirs[1] / "specialization.json").exists()
    files = [dirs[0] / "specialization.json", dirs[0] / "annex.json"]
    assert all(f.exists() for f in files)
    written = [f.stat().st_mtime_ns for f in files]
    (dirs[1] / "settings.json").write_text(json.dumps(saved(dirs[1]) | {"steps": 5}))
    assert evaluate(shared, block="rq1", panel_dir=panel_dir, batch_size=2) == dirs
    assert [f.stat().st_mtime_ns for f in files] == written  # evaluated once
    assert (
        json.loads((dirs[0] / "annex.json").read_text())["batch_size"] == TIERS["laptop"].batch_size
    )
    assert (
        json.loads((dirs[1] / "annex.json").read_text())["batch_size"] == 2
    )  # the flag reaches it
    assert capsys.readouterr().out.count("evaluated before") == 1


def test_evaluate_leaves_a_run_whose_checkpoint_is_behind_its_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = settings(tmp_path)
    panel_dir = write_panel(tmp_path)
    dirs = gate1(shared, tier="laptop", **RATES_FLAGS, seeds=(0,), tokens=TOKENS)
    # As a run killed while it wrote its last checkpoint: the log is whole, the model an earlier one.
    state = torch.load(dirs[1] / "checkpoint.pt", weights_only=True)
    torch.save(state | {"step": state["step"] - 1}, dirs[1] / "checkpoint.pt")
    assert evaluate(shared, block="rq1", panel_dir=panel_dir) == dirs[:1]
    assert not (dirs[1] / "specialization.json").exists()
    assert "checkpoint behind its log" in capsys.readouterr().out


def test_the_command_line_runs_a_block_with_the_shared_flags(tmp_path: Path) -> None:
    shared = settings(tmp_path)
    flags = [
        "--runs-dir", str(shared.runs_dir), "--data-dir", str(shared.data_dir), "--no-compiled",
        "--warmup-steps", "2", "--log-every", "5", "--eval-every", "10", "--eval-batches", "1",
        "--checkpoint-every", "10", "--tier", "laptop", "--tokens", str(TOKENS),
    ]  # fmt: skip
    dirs = main(["gate1", *flags, "--adamw-lr", "3e-3", "--muon-lr", "3e-3", "--seeds", "1"])
    assert isinstance(dirs, list) and [d.name for d in dirs] == [
        "laptop-adamw-s1",
        "laptop-muon-s1",
    ]
    assert saved(dirs[0])["adamw_lr"] == 3e-3 and saved(dirs[0])["seed"] == 1
    assert saved(dirs[0]) == saved(shared.runs_dir / "rq1" / "laptop-adamw-s1")
