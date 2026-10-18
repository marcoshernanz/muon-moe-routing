"""Tune one optimizer's learning rate: one run per rate, then a table of final validation losses.

Each optimizer's rate is tuned before any comparison. AdamW is swept on the partition that trains everything with it. Muon is swept on the partition that trains every matrix with it, with the AdamW rate held at its tuned value, because the tables and norms stay on AdamW in every arm. The momentum control is swept the same way on its own partition, not to tune it, since no later block runs it, but for the frontier of the plan's section 7.2, which sets each optimizer's loss against its balancing effort over its whole sweep. Every rate is an ordinary run in its own directory under the sweep's, resumed if it was interrupted and skipped if it finished, so a preempted sweep continues with the same command, and the same command on several GPUs shares the rates out (``tfg.workers``), each worker waiting for the others' before it reads the table. When the best rate is an edge of the grid, the grid is extended by one point past that edge, once; a run that diverged has no validation loss and cannot win.
"""

import gc
import json
import math
from pathlib import Path

import torch
import tyro

from tfg import workers
from tfg.data import DATA_DIR
from tfg.train import (
    ADAMW_LR,
    CHECKPOINT_EVERY,
    EVAL_BATCHES,
    EVAL_EVERY,
    LOG_EVERY,
    MUON_LR,
    RUNS_DIR,
    SCHEDULE,
    SEED,
    WARMUP_STEPS,
    train,
)

# The partition each optimizer is swept on; the control takes Muon's rate.
PARTITION_OF = {"adamw": "adamw", "muon": "muon", "momentum": "momentum"}


def final_val_loss(run_dir: Path) -> float:
    """The validation loss of a finished run: its last step always evaluates, and a run that diverged ends on a NaN."""
    last = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    return float(json.loads(last)["val_loss"])


def best(losses: dict[float, float]) -> float:
    """The rate with the lowest final validation loss, the runs that diverged left out."""
    finite = {rate: loss for rate, loss in losses.items() if math.isfinite(loss)}
    if not finite:
        raise ValueError(f"every rate diverged: {sorted(losses)}")
    return min(finite, key=finite.__getitem__)


def beyond(losses: dict[float, float]) -> float | None:
    """The rate one grid step past the best one when the best one is an edge of the grid, else nothing."""
    rates = sorted(losses)
    if len(rates) < 2 or best(losses) not in (rates[0], rates[-1]):
        return None
    edge, inner = (rates[0], rates[1]) if best(losses) == rates[0] else (rates[-1], rates[-2])
    return float(f"{edge * edge / inner:g}")  # rounded as the run's directory is named


def sweep(
    *,
    tier: str,
    optimizer: str,
    rates: list[float],
    steps: int,
    adamw_lr: float = ADAMW_LR,
    batch_size: int | None = None,
    warmup_steps: int = WARMUP_STEPS,
    schedule: str = SCHEDULE,
    log_every: int = LOG_EVERY,
    eval_every: int = EVAL_EVERY,
    eval_batches: int = EVAL_BATCHES,
    checkpoint_every: int = CHECKPOINT_EVERY,
    seed: int = SEED,
    compiled: bool = False,
    recompute: int = 0,
    extend: bool = True,
    sweep_dir: Path | None = None,
    data_dir: Path = DATA_DIR,
) -> dict[float, float]:
    """Train one run per rate; return, print, and write to ``sweep.json`` the final validation loss of each. ``adamw_lr`` is the tuned AdamW rate a Muon or control sweep holds fixed. With ``extend``, a best rate at an edge of the grid adds the rate one grid step past it."""
    if optimizer not in PARTITION_OF:
        raise ValueError(f"optimizer {optimizer!r} is not one of {sorted(PARTITION_OF)}")
    if not rates:
        raise ValueError("no rates to sweep")
    sweep_dir = sweep_dir or RUNS_DIR / f"sweep-{tier}-{optimizer}-s{seed}"
    rate_of: dict[Path, float] = {}

    def run(run_dir: Path) -> None:
        rate = rate_of[run_dir]
        # Each rate starts on an empty cache, as a block's runs do (tfg.program): the last run's freed blocks would sit beside the new ones, and a worker's second AdamW run of the tuning on a 5090 hit the allocator's limit that way (2026-09-23).
        gc.collect()
        torch.cuda.empty_cache()
        adamw_rate, muon_rate = (rate, MUON_LR) if optimizer == "adamw" else (adamw_lr, rate)
        train(
            tier=tier,
            partition=PARTITION_OF[optimizer],
            steps=steps,
            batch_size=batch_size,
            adamw_lr=adamw_rate,
            muon_lr=muon_rate,
            warmup_steps=warmup_steps,
            schedule=schedule,
            log_every=log_every,
            eval_every=eval_every,
            eval_batches=eval_batches,
            checkpoint_every=checkpoint_every,
            seed=seed,
            compiled=compiled,
            recompute=recompute,
            run_dir=run_dir,
            data_dir=data_dir,
            resume=True,
        )

    def losses_of(grid: list[float]) -> dict[float, float]:
        """Train the grid's runs, sharing them with any other worker on the sweep, and wait for all of them."""
        dirs = {sweep_dir / f"lr{rate:g}": rate for rate in grid}
        rate_of.update(dirs)
        workers.each(list(dirs), run, wait=True)
        return {rate: final_val_loss(run_dir) for run_dir, rate in dirs.items()}

    losses = losses_of(rates)
    extension = beyond(losses) if extend else None
    if extension is not None:
        print(f"the best rate, {best(losses):g}, is an edge of the grid: adding {extension:g}")
        losses |= losses_of([extension])
    table = {
        "tier": tier,
        "optimizer": optimizer,
        "steps": steps,
        "val_loss": {f"{rate:g}": loss for rate, loss in losses.items()},
        "best": best(losses),
        "extension": extension,
        "edge": beyond(losses) is not None,  # still at an edge: the grid does not bracket it
    }
    (sweep_dir / "sweep.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    for rate, loss in losses.items():
        print(f"lr {rate:g}: val_loss {loss:.4f}" + ("  <- best" if rate == best(losses) else ""))
    if table["edge"]:
        print(f"the best rate, {best(losses):g}, is still an edge of the grid: widen it by hand")
    return losses


if __name__ == "__main__":
    tyro.cli(sweep)
