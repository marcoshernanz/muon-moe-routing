"""Tests of the learning-rate sweep on the synthetic stream; seconds on a laptop."""

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from tfg.sweep import best, beyond, sweep
from tfg.tests.synthetic import QUICK, write_shards


def quick_sweep(
    tmp_path: Path, optimizer: str, rates: list[float], **overrides: Any
) -> dict[float, float]:
    settings: dict[str, Any] = {
        "tier": "laptop",
        "steps": 10,
        **QUICK,
        "sweep_dir": tmp_path / "sweep",
        "data_dir": write_shards(tmp_path),
        **overrides,
    }
    return sweep(optimizer=optimizer, rates=rates, **settings)


def test_the_best_rate_leaves_out_the_runs_that_diverged() -> None:
    assert best({1e-3: 3.0, 2e-3: 2.5, 4e-3: math.nan}) == 2e-3
    assert best({1e-3: math.inf, 2e-3: 2.5}) == 2e-3
    with pytest.raises(ValueError, match="every rate diverged"):
        best({1e-3: math.nan, 2e-3: math.nan})


def test_a_best_rate_at_an_edge_of_the_grid_asks_for_the_next_point() -> None:
    assert beyond({1e-3: 3.0, 2e-3: 2.5, 4e-3: 2.0}) == 8e-3  # the top edge: one step up
    assert beyond({2.5e-4: 2.0, 5e-4: 2.5, 1e-3: 3.0}) == 1.25e-4  # the bottom edge: one down
    assert beyond({4e-3: 2.0, 1e-3: 3.0, 2e-3: 2.5}) == 8e-3  # in whatever order they ran
    assert beyond({1e-3: 3.0, 2e-3: 2.0, 4e-3: 2.5}) is None  # inside the grid
    assert beyond({1e-3: 3.0, 2e-3: 2.0, 4e-3: math.nan}) is None  # the edge diverged
    assert beyond({1e-3: 3.0}) is None  # no step to take


def test_sweep_extends_the_grid_once_past_a_best_rate_at_its_edge(tmp_path: Path) -> None:
    losses = quick_sweep(tmp_path, "adamw", [3e-3, 1e-5])  # 3e-3 wins: 0.9 is one step past it
    assert list(losses) == [3e-3, 1e-5, 0.9] and best(losses) == 3e-3
    table = json.loads((tmp_path / "sweep" / "sweep.json").read_text(encoding="utf-8"))
    assert table["extension"] == 0.9 and table["edge"] is False and table["best"] == 3e-3
    settings = json.loads((tmp_path / "sweep" / "lr0.9" / "settings.json").read_text("utf-8"))
    assert settings["adamw_lr"] == 0.9
    assert quick_sweep(tmp_path, "adamw", [3e-3, 1e-5]) == losses  # relaunched: nothing new


def test_every_rate_of_a_sweep_starts_on_an_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emptied: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    quick_sweep(tmp_path, "adamw", [3e-3, 1e-5], extend=False)
    assert len(emptied) == 2  # once before each run, as a block's runs do
    quick_sweep(tmp_path, "adamw", [3e-3, 1e-5], extend=False)
    assert len(emptied) == 4  # a relaunch too, before it finds each run finished


def test_sweep_runs_every_rate_and_picks_the_best(tmp_path: Path) -> None:
    losses = quick_sweep(tmp_path, "adamw", [3e-3, 1e-5], extend=False)
    assert set(losses) == {3e-3, 1e-5}
    assert all(math.isfinite(loss) for loss in losses.values())
    assert losses[3e-3] < losses[1e-5]  # ten steps at 1e-5 barely move the loss

    table = json.loads((tmp_path / "sweep" / "sweep.json").read_text(encoding="utf-8"))
    assert table["best"] == 3e-3 and table["val_loss"] == {
        "0.003": losses[3e-3],
        "1e-05": losses[1e-5],
    }
    assert table["extension"] is None and table["edge"] is True  # at the edge, and left there
    assert (tmp_path / "sweep" / "lr0.003" / "checkpoint.pt").exists()
    assert (tmp_path / "sweep" / "lr1e-05" / "checkpoint.pt").exists()
    skipped = quick_sweep(tmp_path, "adamw", [3e-3, 1e-5], extend=False)
    assert skipped == losses  # finished runs are skipped


def test_a_muon_sweep_holds_the_adamw_rate(tmp_path: Path) -> None:
    quick_sweep(
        tmp_path,
        "muon",
        [2e-3],
        steps=2,
        adamw_lr=5e-4,
        schedule="constant",
        warmup_steps=1,
        log_every=2,
        eval_every=2,
        checkpoint_every=2,
    )
    settings = json.loads(
        (tmp_path / "sweep" / "lr0.002" / "settings.json").read_text(encoding="utf-8")
    )
    assert settings["partition"] == "muon" and settings["schedule"] == "constant"
    assert settings["muon_lr"] == 2e-3 and settings["adamw_lr"] == 5e-4


def test_the_control_is_swept_on_its_own_partition_at_muon_rate(tmp_path: Path) -> None:
    quick_sweep(
        tmp_path,
        "momentum",
        [5e-4],
        steps=2,
        adamw_lr=3e-3,
        schedule="constant",
        warmup_steps=1,
        log_every=2,
        eval_every=2,
        checkpoint_every=2,
    )
    settings = json.loads(
        (tmp_path / "sweep" / "lr0.0005" / "settings.json").read_text(encoding="utf-8")
    )
    assert settings["partition"] == "momentum"
    assert settings["muon_lr"] == 5e-4 and settings["adamw_lr"] == 3e-3


def test_sweep_refuses_unknown_optimizers_and_empty_grids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="adamw"):
        quick_sweep(tmp_path, "sgd", [1e-3])
    with pytest.raises(ValueError, match="no rates"):
        quick_sweep(tmp_path, "adamw", [])
