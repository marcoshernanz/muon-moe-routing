"""Tests of saving and restoring a run: bitwise continuation, replacement, and rewinding."""

import json
from pathlib import Path

import pytest
import torch
from torch.optim import Optimizer

from tfg import checkpoint
from tfg.data import VOCAB_SIZE
from tfg.model import LanguageModel
from tfg.optim import build
from tfg.tier import LAPTOP
from tfg.train import train_step


def fresh(partition: str, seed: int) -> tuple[LanguageModel, dict[str, Optimizer]]:
    torch.manual_seed(seed)
    model = LanguageModel(VOCAB_SIZE, LAPTOP)
    return model, build(model, partition, adamw_lr=1e-3, muon_lr=1e-3)


def batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(step)
    tokens = torch.randint(VOCAB_SIZE, (2, LAPTOP.context_len + 1), generator=generator)  # [B, T+1]
    return tokens[:, :-1], tokens[:, 1:]


def flat(model: LanguageModel) -> torch.Tensor:
    """Every saved weight and buffer of the model as one vector."""
    return torch.cat([tensor.flatten() for tensor in model.state_dict().values()])


@pytest.mark.parametrize("partition", ["muon", "momentum", "muon_router"])
def test_a_resumed_run_continues_bitwise(tmp_path: Path, partition: str) -> None:
    model, optimizers = fresh(partition, seed=0)
    for step in range(2):
        train_step(model, optimizers, batch(step), factor=1.0)
    path = checkpoint.save(tmp_path, model, optimizers, step=2)
    assert path == tmp_path / "checkpoint.pt"
    assert not path.with_suffix(".tmp").exists()
    train_step(model, optimizers, batch(2), factor=1.0)

    resumed, resumed_optimizers = fresh(partition, seed=1)  # other init, all overwritten
    assert checkpoint.load(tmp_path, resumed, resumed_optimizers) == 2
    train_step(resumed, resumed_optimizers, batch(2), factor=1.0)
    assert torch.equal(flat(resumed), flat(model))
    assert any(mixture.router_bias.abs().sum() > 0 for mixture in resumed.decoder.mixtures())
    assert list(resumed_optimizers) == list(optimizers)
    for kind, optimizer in optimizers.items():
        assert [g["peak_lr"] for g in optimizer.param_groups] == [
            g["peak_lr"] for g in resumed_optimizers[kind].param_groups
        ]


def test_load_without_a_checkpoint_starts_from_zero_and_changes_nothing(tmp_path: Path) -> None:
    model, optimizers = fresh("adamw", seed=0)
    before = flat(model)
    assert checkpoint.load(tmp_path, model, optimizers) == 0
    assert torch.equal(flat(model), before)


def test_save_replaces_the_previous_checkpoint(tmp_path: Path) -> None:
    model, optimizers = fresh("adamw", seed=0)
    checkpoint.save(tmp_path, model, optimizers, step=1)
    checkpoint.save(tmp_path, model, optimizers, step=2)
    assert [p.name for p in tmp_path.iterdir()] == ["checkpoint.pt"]
    assert checkpoint.load(tmp_path, model, optimizers) == 2


def test_save_model_keeps_the_model_alone_beside_the_checkpoint(tmp_path: Path) -> None:
    model, optimizers = fresh("adamw", seed=0)
    checkpoint.save(tmp_path, model, optimizers, step=2)
    path = checkpoint.save_model(tmp_path, model, step=1)
    assert path == tmp_path / "early.pt" and not path.with_suffix(".tmp").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.pt", "early.pt"]
    state = torch.load(path, weights_only=True)
    assert set(state) == {"step", "model"} and state["step"] == 1
    other, _ = fresh("adamw", seed=1)
    assert checkpoint.load_model(tmp_path, other, file=checkpoint.EARLY_FILE) == 1
    assert torch.equal(flat(other), flat(model))
    assert checkpoint.load_model(tmp_path, other) == 2  # the checkpoint by default


def test_load_rejects_another_partition(tmp_path: Path) -> None:
    model, optimizers = fresh("adamw", seed=0)
    checkpoint.save(tmp_path, model, optimizers, step=1)
    other, other_optimizers = fresh("muon", seed=0)
    with pytest.raises(ValueError, match=r"holds the optimizers \['adamw'\]"):
        checkpoint.load(tmp_path, other, other_optimizers)


def test_rewind_drops_the_records_past_the_step(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps({"step": step}) + "\n" for step in (10, 20, 30)))
    checkpoint.rewind(path, 20)
    assert path.read_text().splitlines() == ['{"step": 10}', '{"step": 20}']
    written = path.stat().st_mtime_ns
    checkpoint.rewind(path, 20)  # nothing past the step: the file is not rewritten
    assert path.stat().st_mtime_ns == written
    checkpoint.rewind(path, 0)
    assert path.read_text() == ""
    checkpoint.rewind(tmp_path / "missing.jsonl", 20)
    assert not (tmp_path / "missing.jsonl").exists()
    path.write_text('{"step": 10}\n{"step": 20}\n{"step": 3')  # a crash in the middle of a line
    checkpoint.rewind(path, 30)
    assert path.read_text() == '{"step": 10}\n{"step": 20}\n'
    path.write_text('{"step": 1\n{"step": 20}\n')  # a torn line anywhere else is no crash's doing
    with pytest.raises(json.JSONDecodeError):
        checkpoint.rewind(path, 30)


def test_last_step_reads_the_last_record_and_not_a_torn_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    assert checkpoint.last_step(path) is None
    path.write_text("")
    assert checkpoint.last_step(path) is None
    path.write_text('{"step": 10}\n{"step": 20}\n{"step": 3')
    assert checkpoint.last_step(path) == 20


def test_load_model_restores_the_weights_alone_and_returns_the_step(tmp_path: Path) -> None:
    model, optimizers = fresh("muon", 0)
    train_step(model, optimizers, batch(0), factor=1.0)
    checkpoint.save(tmp_path, model, optimizers, step=7)
    other, _ = fresh("adamw", 1)  # another partition and seed: no optimizers to match
    assert not torch.equal(flat(other), flat(model))
    assert checkpoint.load_model(tmp_path, other) == 7
    assert torch.equal(flat(other), flat(model))
    with pytest.raises(FileNotFoundError):
        checkpoint.load_model(tmp_path / "nowhere", other)


def test_the_saved_step_is_read_without_the_model(tmp_path: Path) -> None:
    model, optimizers = fresh("adamw", seed=0)
    assert checkpoint.saved_step(tmp_path) is None  # no checkpoint yet
    checkpoint.save(tmp_path, model, optimizers, step=7)
    checkpoint.save_model(tmp_path, model, step=1)  # the early save is not the checkpoint
    assert checkpoint.saved_step(tmp_path) == 7


def test_discard_empties_the_run_directory(tmp_path: Path) -> None:
    model, optimizers = fresh("adamw", seed=0)
    checkpoint.save(tmp_path, model, optimizers, step=2)
    checkpoint.save_model(tmp_path, model, step=1)
    for name in ("metrics.jsonl", "annex.json", "checkpoint.tmp"):
        (tmp_path / name).write_text('{"step": 2}\n', encoding="utf-8")
    checkpoint.discard(tmp_path)
    assert not list(tmp_path.iterdir())
    checkpoint.discard(tmp_path)  # nothing to remove: nothing happens
    assert checkpoint.load(tmp_path, model, optimizers) == 0
