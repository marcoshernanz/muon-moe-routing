"""Synthetic inputs for the tests: a predictable token stream for the ones that train, with quick settings, and ready-made run directories for the ones that analyze."""

import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from tfg.data import VOCAB_SIZE
from tfg.panel import DOMAINS
from tfg.train import arm, train

STATES = 500  # every token fixes the next one, so the stream is learnable in a few dozen steps
WHITESPACE = [1, 2]  # the ids the synthetic panel's mask marks


class Intervals(TypedDict):
    warmup_steps: int
    log_every: int
    eval_every: int
    eval_batches: int
    checkpoint_every: int


QUICK = Intervals(warmup_steps=2, log_every=5, eval_every=10, eval_batches=1, checkpoint_every=10)


def train_briefly(run_dir: Path, data_dir: Path) -> Path:
    """Ten steps of the laptop tier at the tuned rates, under the quick intervals."""
    train(
        tier="laptop",
        steps=10,
        adamw_lr=3e-3,
        muon_lr=3e-3,
        **QUICK,
        run_dir=run_dir,
        data_dir=data_dir,
    )
    return run_dir


def write_shards(tmp_path: Path) -> Path:
    stream = (np.arange(60_000) * 37) % STATES + 1
    np.save(tmp_path / "train_00000.npy", stream[:50_000].astype(np.uint16))
    np.save(tmp_path / "validation_00000.npy", stream[50_000:].astype(np.uint16))
    return tmp_path


def write_panel(tmp_path: Path, tokens: int = 1024) -> Path:
    """A specialization panel of ``tokens`` per domain, each domain a different stretch of the stream; every third token of the code slice is a marked id, so that slice loses tokens when counted without whitespace."""
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    stream = (np.arange(len(DOMAINS) * tokens) * 37) % STATES + 1
    for index, domain in enumerate(DOMAINS):
        ids = stream[index * tokens : (index + 1) * tokens].astype(np.uint16)
        if domain == "code":
            ids[::3] = WHITESPACE[0]
        np.save(panel_dir / f"{domain}.npy", ids)
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    mask[WHITESPACE] = True
    np.save(panel_dir / "whitespace.npy", mask)
    return panel_dir


def write_run(
    runs_dir: Path,
    partition: str,
    seed: int,
    *,
    spans: list[float],
    balanced: bool = True,
    lr_multiplier: float = 1.0,
    balancer_multiplier: float = 1.0,
    val_losses: list[float] | None = None,
    losses: list[float] | None = None,
    loads: list[list[float]] | None = None,
    panel: list[float] | None = None,
    annex: list[float] | None = None,
    tier: str = "laptop",
) -> Path:
    """A run of ``len(spans)`` logged steps with two mixtures whose bias spans differ by 0.1, the validation losses of its last logged steps (by default ``2.0 + seed`` on the last, which from seed 2 on is no better than the default training loss, so the analysis counts that run as diverged unless ``val_losses`` says otherwise), and, when given, a panel evaluation with these divergences per mixture (half as much without whitespace) and a usage share per domain, and an annex file with these saturations per mixture (half as much routing purity). Every logged step carries the training loss of its ten steps (``losses``, one per step of the run, by default 4.0 throughout), AdamW's pre-clip norm on each of them (0.5, but the log step's own, which is the step number over 100) and the rate factor, 1 throughout. The warmup is ten steps and AdamW's rate 1e-3. The activation readings grow by 0.1 per logged step, the residual's by 1 per mixture; so does the router group's weight norm from 1, while the expert group's angular step grows by 0.01 from 0.01. Both mixtures share the load of ``loads`` at each logged step, by default half and half of two experts."""
    name = arm(
        partition,
        balanced=balanced,
        router_lr_multiplier=1.0,
        lr_multiplier=lr_multiplier,
        balancer_multiplier=balancer_multiplier,
    )
    run_dir = runs_dir / f"{name}-s{seed}"
    run_dir.mkdir(parents=True)
    settings = {
        "tier": tier,
        "partition": partition,
        "balanced": balanced,
        "router_lr_multiplier": 1.0,
        "lr_multiplier": lr_multiplier,
        "balancer_multiplier": balancer_multiplier,
        "adamw_clip": 1.0,
        "muon_clip": 1.0,
        "adamw_lr": 1e-3,
        "warmup_steps": 10,
        "seed": seed,
        "steps": 10 * len(spans),
    }
    (run_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    losses = losses or [4.0] * (10 * len(spans))
    assert len(losses) == 10 * len(spans)
    metrics: list[dict[str, object]] = [
        {
            "step": 10 * (t + 1),
            "loss": losses[10 * t + 9],
            "lr_factor": 1.0,
            "grad_norm": (t + 1) / 10,
            "grad_norms": {"adamw": [0.5] * 9 + [(t + 1) / 10]},
            "losses": losses[10 * t : 10 * (t + 1)],
        }
        for t in range(len(spans))
    ]
    val_losses = val_losses or [2.0 + seed]
    for entry, val_loss in zip(metrics[-len(val_losses) :], val_losses):
        entry["val_loss"] = val_loss
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    lines = [
        {
            "step": 10 * (t + 1),
            "mixtures": [
                {
                    "bias_span": span + 0.1 * mixture,
                    "load": loads[t] if loads else [0.5, 0.5],
                    "usage_perplexity": 2.0,
                    "load_churn": 0.0 if t == 0 else 0.2,
                    "update_load_correlation": None if t == 0 else 0.5,
                    "residual_max": 1.0 + 0.1 * t + mixture,
                    "router_logit_max": 2.0 + 0.1 * t,
                    "expert_output_max": 3.0 + 0.1 * t,
                    "attention_logit_max": 5.0 + 0.1 * t,
                }
                for mixture in range(2)
            ],
            "groups": {
                "router": {"weight_norm": 1.0 + 0.1 * t, "angular_step": 0.05},
                "experts": {"weight_norm": 2.0, "angular_step": 0.01 * (t + 1)},
            },
        }
        for t, span in enumerate(spans)
    ]
    (run_dir / "routing.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    if panel is not None:
        share = {domain: [0.5 + 0.1 * g, 0.5 - 0.1 * g] for g, domain in enumerate(DOMAINS)}
        mixtures = [
            {"share": share, "specialization": v, "specialization_nonspace": v / 2} for v in panel
        ]
        evaluation = {"step": 10 * len(spans), "uniform": 0.5, "mixtures": mixtures}
        (run_dir / "specialization.json").write_text(json.dumps(evaluation), encoding="utf-8")
    if annex is not None:
        readings = [
            {"saturation": v, "routing_purity": v / 2, "gradient_alignment": [v, None]}
            for v in annex
        ]
        evaluation = {"step": 10 * len(spans), "mixtures": readings}
        (run_dir / "annex.json").write_text(json.dumps(evaluation), encoding="utf-8")
    return run_dir
