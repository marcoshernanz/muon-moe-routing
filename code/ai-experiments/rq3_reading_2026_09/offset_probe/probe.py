"""One training run of the thesis model with every mixture's per-expert score offset read beside the thesis telemetry, and, optionally, the first mixture's router fed its inputs minus their batch mean.

The question is the step the second chain's logs leave open. Across its 27 balanced runs the first mixture's integrated bias span follows the spread of the router's mean score across experts (`mean_score_span`), not the router-input cosine; the plan's H6 names the route between the two: a direction every token's input shares adds a fixed amount to each expert's logit, and the bias cancels it. Since the router is linear, the batch-mean logit of expert e is exactly w_e . x_bar, the router row times the batch-mean input, so the offset factors into the length of the mean input (the shared direction's strength, which the cosine reads) and how unevenly the rows lean along it. The logs keep neither per expert. This run keeps, per logged step and mixture: every expert's bias, batch-mean logit and batch-mean score, the load it took and the load its scores alone would give, the mean input's length and the mean length of an input, every router row's norm and its component along the mean input, and the offset the uncentred mean input would give (the same as the mean logit unless the input is centred).

With ``--centre`` the first mixture's router scores each token's input minus the batch mean of the inputs, so every expert's mean logit is zero by construction and no shared direction reaches its scores; the experts, the shared expert and every other mixture see the inputs as before. That is an intervention on the model, used here as a test of the mechanism and not as a proposal: if the span Muon adds is the bias cancelling the offset the mean input puts in the scores, the centred run's first-mixture span should fall to AdamW's.

Everything else is the thesis code through imports of ``tfg`` (the model, the partitions, the data order, the schedule, clipping, the balancer, the telemetry, the trainer); nothing under ``tfg`` imports from here. Eager, as on the T4, where the experts run slice by slice.

At the mechanism tier (on a rented 5090) it trains the thesis's own run, compiled and with the first mixture block recomputed as the chains train, on the schedule of a 1B-token run (``--steps 15259``), and ``--stop-after 3000`` ends it after that step's validation loss, so its first 3,000 steps are the second chain's run of the same arm and seed with the readings added; ``--balancer-multiplier`` sets the bias's rate as ``tfg.train`` does.

Usage, from ``code/``: PYTHONPATH=. uv run python ai-experiments/rq3_reading_2026_09/offset_probe/probe.py OUT --partition muon --seed 0 [--muon-lr-multiplier 0.5] [--balancer-multiplier 4] [--centre] [--steps 3000] [--stop-after N] [--tier kaggle_t4] [--compiled --recompute 1]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tfg import train as trainer
from tfg.model import MixtureOfExperts
from tfg.telemetry import Telemetry, raw_fractions

ADAMW_LR, MUON_LR = 2e-3, 4e-3  # the mechanism tier's tuned rates (thesis/PLAN.md, section 6)
SAVE_EVERY = 50  # logged steps between saves of the readings, so a cut session keeps them

original_route = MixtureOfExperts.route
original_write = Telemetry.write


def route(self: MixtureOfExperts, tokens: torch.Tensor):  # [N, D]
    """The router on its inputs, centred when the mixture is marked; the mean input, uncentred, kept for the reading."""
    if self.training:
        self.probe_mean_input = tokens.detach().float().mean(0)  # [D]
        self.probe_mean_length = tokens.detach().float().norm(dim=-1).mean()  # []
    if getattr(self, "probe_centre", False):
        tokens = tokens - tokens.mean(0, keepdim=True)  # [N, D]
    return original_route(self, tokens)


class Readings:
    """The per-expert readings of every logged step, as arrays saved beside the run's logs."""

    def __init__(self, path: Path):
        self.path = path
        self.steps: list[int] = []
        self.rows: dict[str, list[np.ndarray]] = {}

    def add(self, step: int, values: dict[str, np.ndarray]) -> None:
        self.steps.append(step)
        for name, value in values.items():
            self.rows.setdefault(name, []).append(value)
        if len(self.steps) % SAVE_EVERY == 0:
            self.save()

    def save(self) -> None:
        arrays = {name: np.stack(rows).astype(np.float32) for name, rows in self.rows.items()}
        np.savez_compressed(self.path, steps=np.array(self.steps), **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--partition", default="muon")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--muon-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--centre", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tier", default="kaggle_t4")
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--balancer-multiplier", type=float, default=1.0)
    parser.add_argument("--stop-after", type=int, default=0, help="end the run after this step's log and validation loss; 0 runs every step")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="0 saves only at the end")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--recompute", type=int, default=0)
    args = parser.parse_args()

    name = args.partition
    if args.muon_lr_multiplier != 1:
        name += f"-muonlr{args.muon_lr_multiplier:g}x"
    if args.balancer_multiplier != 1:
        name += f"-balancer{args.balancer_multiplier:g}x"
    if args.centre:
        name += "-centred"
    run_dir = args.out / f"{args.tier}-{name}-s{args.seed}"
    if (run_dir / "probe.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    readings = Readings(run_dir / "offsets.npz")
    began = time.perf_counter()

    def finish() -> None:
        """Save the readings and the run's summary, and remove the saves, which are not needed."""
        readings.save()
        records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        last = next((r for r in reversed(records) if "val_loss" in r), None)
        (run_dir / "probe.json").write_text(
            json.dumps(
                {
                    "partition": args.partition,
                    "seed": args.seed,
                    "muon_lr_multiplier": args.muon_lr_multiplier,
                    "balancer_multiplier": args.balancer_multiplier,
                    "centre": args.centre,
                    "steps": args.steps,
                    "stop_after": args.stop_after,
                    "last_step": records[-1]["step"] if records else 0,
                    "tier": args.tier,
                    "compiled": args.compiled,
                    "recompute": args.recompute,
                    "val_loss": last["val_loss"] if last else None,
                    "val_step": last["step"] if last else None,
                    "seconds": time.perf_counter() - began,
                }
            ),
            encoding="utf-8",
        )
        for saved in run_dir.glob("*.pt"):
            saved.unlink()

    MixtureOfExperts.route = route

    def write(self: Telemetry, step: int, before) -> None:
        if args.stop_after and step > args.stop_after:  # the log step after the last one kept: its readings are not
            finish()
            raise SystemExit(0)
        original_write(self, step, before)
        values: dict[str, list[np.ndarray]] = {}
        for mixture, snap in zip(self.mixtures, before.mixtures):
            logits = mixture.router_logits  # of the step's forward, on the router's own inputs [N, E]
            mean_input = mixture.probe_mean_input  # uncentred [D]
            weight = snap.router.double()  # the rows the forward used [E, D]
            length = mean_input.double().norm()  # []
            lean = weight @ (mean_input.double() / length)  # each row's component along the mean input [E]
            per_mixture = {
                "bias": snap.bias,
                "mean_logit": logits.mean(0),
                "mean_score": torch.sigmoid(logits).mean(0),
                "load": mixture.expert_load.float() / mixture.expert_load.sum(),
                "raw_load": raw_fractions(logits, mixture.num_active),
                "row_norm": weight.norm(dim=1),
                "lean": lean,
                "offset": lean * length,  # w_e . x_bar: the mean logit an uncentred input gives
                "lengths": torch.stack([length, mixture.probe_mean_length.double()]),
            }
            for key, value in per_mixture.items():
                values.setdefault(key, []).append(value.detach().double().cpu().numpy())
        readings.add(step, {key: np.stack(value) for key, value in values.items()})  # [B, E] each

    Telemetry.write = write

    original_model = trainer.LanguageModel

    def model_with_centre(*a, **k):
        model = original_model(*a, **k)
        if args.centre:
            model.decoder.mixtures()[0].probe_centre = True
        return model

    trainer.LanguageModel = model_with_centre
    trainer.train(
        tier=args.tier,
        partition=args.partition,
        steps=args.steps,
        adamw_lr=ADAMW_LR,
        muon_lr=MUON_LR,
        muon_lr_multiplier=args.muon_lr_multiplier,
        balancer_multiplier=args.balancer_multiplier,
        seed=args.seed,
        eval_batches=args.eval_batches,
        checkpoint_every=args.checkpoint_every or args.steps // 10 * 10 or 10,
        compiled=args.compiled,
        recompute=args.recompute,
        run_dir=run_dir,
    )
    finish()

if __name__ == "__main__":
    main()
