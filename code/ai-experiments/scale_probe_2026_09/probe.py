"""One training run of the thesis model at the confirmation tier with readings of how saturated the first mixture's router is and how often the balancer's bias rather than the router picks a token's experts, beside the offset probe's per-expert readings, and, optionally, the first mixture's router fed its inputs minus their batch mean.

The question is the one the third chain's logs leave. At the confirmation tier under Muon the first mixture's router logits grow to a norm of 16 to 40 (3 to 5 under AdamW), the router-input cosine to 0.5 to 0.8 (under 0.11), the mean selection margin falls below the bias's step of 0.001, the load churns three to twenty times as much, the busiest expert stays at two to eight times its share for the whole run, and the router on AdamW (arm (c)) has its loss spikes with the span at its ceiling of 1 + 2u. The reading proposed: a direction every token's input shares pushes the logits far from zero, the sigmoid flattens the scores, and the bias, added after the sigmoid, then decides the selection, so each of its fixed steps moves many tokens at once. The offset probe (``rq3_reading_2026_09/offset_probe``) showed at the mechanism tier that centring the first mixture's router input removes the early overload and most of the extra span; this run asks the same at the confirmation tier, for the router on Muon and on AdamW.

Per logged step and mixture it keeps the offset probe's per-expert readings (bias, batch-mean logit and score, the load taken and the load the scores alone give, the mean input's length and an input's mean length, every router row's norm and its component along the mean input, the offset the uncentred mean input gives) and six numbers of the batch: the share of scores above 0.99 and below 0.01 (the sigmoid flat), the share of tokens whose selection margin is under two bias steps (one step of the balancer can swap their last chosen expert, since it moves two biases in opposite directions), the share of a token's chosen experts the router's scores alone would not have chosen (what the bias decides), and the mean and 10 percent quantile of the router's own margin, the k-th raw score minus the next, without the bias. "Saturated" here is the sigmoid's flat range, not OLMoE's router saturation of the annex.

With ``--centre`` the first mixture's router scores each token's input minus the batch mean of the inputs, as in the offset probe: an intervention used as a test of the mechanism, not a proposal. Everything else is the thesis code through imports of ``tfg`` (the model, the partitions, the data order, the schedule, clipping, the balancer, the telemetry, the trainer); nothing under ``tfg`` imports from here.

At the confirmation tier (on a rented 5090) it trains the third chain's run of its arm and seed, compiled and with the first nine mixture blocks recomputed, on the schedule of the 3B-token run (``--steps 45776``) at the chain's rates (``--adamw-lr 7e-4 --muon-lr 1.4e-3``), and ``--stop-after 3000`` ends it after that step's validation loss, so a plain run's first 3,000 steps are the chain's with the readings added.

Usage, from ``code/``: PYTHONPATH=. uv run python ai-experiments/scale_probe_2026_09/probe.py OUT --partition muon --seed 0 [--centre] [--balancer-multiplier 4] [--tier confirmation --steps 45776 --stop-after 3000 --adamw-lr 7e-4 --muon-lr 1.4e-3 --compiled --recompute 9]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tfg import train as trainer
from tfg.model import BIAS_UPDATE_RATE, MixtureOfExperts
from tfg.telemetry import MARGIN_QUANTILE, Telemetry, raw_fractions, selection_margins

SAVE_EVERY = 50  # logged steps between saves of the readings, so a cut session keeps them
FLAT = 0.01  # a score within this of 0 or 1 sits where the sigmoid is flat
SUMMARY = ("saturated_high", "saturated_low", "flippable", "bias_decided", "raw_margin_mean", "raw_margin_low")

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


def summary(logits: torch.Tensor, bias: torch.Tensor, num_active: int, step_size: float) -> torch.Tensor:  # [N, E], [E]
    """The six numbers of ``SUMMARY`` for one mixture's batch."""
    scores = torch.sigmoid(logits.float())  # [N, E]
    margins = selection_margins(logits.float(), bias.float(), num_active)  # [N]
    chosen = (scores + bias).topk(num_active, dim=-1).indices  # [N, K]
    raw_top = scores.topk(num_active + 1, dim=-1)  # [N, K+1]
    raw_chosen = raw_top.indices[:, :num_active]  # [N, K]
    kept = (chosen[:, :, None] == raw_chosen[:, None, :]).any(-1).float().mean(-1)  # [N]
    raw_margins = raw_top.values[:, num_active - 1] - raw_top.values[:, num_active]  # [N]
    return torch.stack(
        [
            (scores > 1 - FLAT).float().mean(),
            (scores < FLAT).float().mean(),
            (margins < 2 * step_size).float().mean(),
            1 - kept.mean(),
            raw_margins.mean(),
            raw_margins.quantile(MARGIN_QUANTILE),
        ]
    )  # [6]


class Readings:
    """The readings of every logged step, as arrays saved beside the run's logs."""

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
        np.savez_compressed(self.path, steps=np.array(self.steps), summary_names=np.array(SUMMARY), **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--partition", default="muon")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--centre", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tier", default="laptop")
    parser.add_argument("--adamw-lr", type=float, default=7e-4)
    parser.add_argument("--muon-lr", type=float, default=1.4e-3)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--balancer-multiplier", type=float, default=1.0)
    parser.add_argument("--stop-after", type=int, default=0, help="end the run after this step's log and validation loss; 0 runs every step")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="0 saves only at the end")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--recompute", type=int, default=0)
    args = parser.parse_args()

    name = args.partition
    if args.balancer_multiplier != 1:
        name += f"-balancer{args.balancer_multiplier:g}x"
    if args.centre:
        name += "-centred"
    run_dir = args.out / f"{args.tier}-{name}-s{args.seed}"
    if (run_dir / "probe.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    readings = Readings(run_dir / "readings.npz")
    step_size = BIAS_UPDATE_RATE * args.balancer_multiplier
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
                    "centre": args.centre,
                    "balancer_multiplier": args.balancer_multiplier,
                    "adamw_lr": args.adamw_lr,
                    "muon_lr": args.muon_lr,
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
                "summary": summary(logits, snap.bias.to(logits.device), mixture.num_active, step_size),
            }
            for key, value in per_mixture.items():
                values.setdefault(key, []).append(value.detach().double().cpu().numpy())
        readings.add(step, {key: np.stack(value) for key, value in values.items()})  # [B, E], [B, 2], [B, 6]

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
        adamw_lr=args.adamw_lr,
        muon_lr=args.muon_lr,
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
