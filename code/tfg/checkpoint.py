"""Save and restore a run: the model, every optimizer, and the step count, in one file per run.

A batch is a pure function of the seed and the step and nothing in a step draws random numbers, so weights, optimizer states, and the step are the whole state of a run. The file is PyTorch's own state-dict format, written to a temporary name and renamed over the old checkpoint, so a crash mid-write leaves the previous one intact. The metrics files can outrun the checkpoint by a crash in the other direction, so ``rewind`` drops their records past the restored step, along with a last line the crash cut short. A run also keeps the model alone from early in training, under its own name, for the annex reading that compares the routing at the end with the routing then.
"""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

FILE = "checkpoint.pt"
EARLY_FILE = "early.pt"  # the model alone, from early in the run, for the annex's router saturation


def write(state: dict[str, Any], path: Path) -> Path:
    """Write a state to ``path`` atomically: to a temporary name, then renamed over the old file."""
    partial = path.with_suffix(".tmp")
    torch.save(state, partial)
    partial.replace(path)
    return path


def save(run_dir: Path, model: nn.Module, optimizers: dict[str, Optimizer], *, step: int) -> Path:
    """Write the state after ``step`` steps, replacing the previous checkpoint atomically."""
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizers": {kind: optimizer.state_dict() for kind, optimizer in optimizers.items()},
    }
    return write(state, run_dir / FILE)


def save_model(run_dir: Path, model: nn.Module, *, step: int, file: str = EARLY_FILE) -> Path:
    """Write the model alone after ``step`` steps as ``file``: what an evaluation needs of an earlier point of the run, without the optimizer state that makes a checkpoint several times the size."""
    return write({"step": step, "model": model.state_dict()}, run_dir / file)


def read(run_dir: Path, model: nn.Module, file: str = FILE) -> dict[str, Any]:
    """A saved state of a run, loaded onto the model's device."""
    device = next(model.parameters()).device
    return torch.load(run_dir / file, map_location=device, weights_only=True)


def load_model(run_dir: Path, model: nn.Module, *, file: str = FILE) -> int:
    """Restore the model alone, as an evaluation needs, from the checkpoint or, as ``file``, from a model-only save; returns the step it was saved at."""
    state = read(run_dir, model, file)
    model.load_state_dict(state["model"])
    return int(state["step"])


def saved_step(run_dir: Path) -> int | None:
    """The step of the run's checkpoint, none without one. The file is memory-mapped, so reading the step does not read the weights."""
    path = run_dir / FILE
    if not path.exists():
        return None
    return int(torch.load(path, map_location="cpu", weights_only=True, mmap=True)["step"])


def load(run_dir: Path, model: nn.Module, optimizers: dict[str, Optimizer]) -> int:
    """Restore the state into the model and optimizers in place.

    Returns the step to continue from: zero when the run has no checkpoint yet. Mismatched architectures or partitions fail loudly in ``load_state_dict``.
    """
    path = run_dir / FILE
    if not path.exists():
        return 0
    state = read(run_dir, model)
    if set(state["optimizers"]) != set(optimizers):
        raise ValueError(
            f"{path} holds the optimizers {sorted(state['optimizers'])}, "
            f"the partition builds {sorted(optimizers)}"
        )
    model.load_state_dict(state["model"])
    for kind, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][kind])
    return int(state["step"])


def discard(run_dir: Path) -> None:
    """Remove every file of the run, for one that starts over. A run writes its first checkpoint only ``checkpoint_every`` steps in, and until then a resume would find the one of the run it replaced; and the replaced run's panel and annex would pass for the new run's, since ``tfg.program evaluate`` reads a run only when it lacks them."""
    for path in run_dir.iterdir():
        if path.is_file():
            path.unlink()


def records(text: str) -> list[str]:
    """The lines of a JSON-lines file that hold a record: a crash in the middle of a write leaves a torn last line, which holds none and is left out."""
    lines = text.splitlines()
    try:
        json.loads(lines[-1] if lines else "{}")
    except json.JSONDecodeError:
        return lines[:-1]
    return lines


def last_step(path: Path) -> int | None:
    """The step of a JSON-lines file's last record, none when it has no record."""
    lines = records(path.read_text(encoding="utf-8")) if path.exists() else []
    return int(json.loads(lines[-1])["step"]) if lines else None


def rewind(path: Path, step: int) -> None:
    """Drop the records of a JSON-lines file that were written after ``step``, and a torn last line."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    kept = "".join(f"{line}\n" for line in records(text) if json.loads(line)["step"] <= step)
    if kept != text:  # a file with nothing to drop is left untouched
        path.write_text(kept, encoding="utf-8")
