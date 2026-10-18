"""The offset probe's runs (``probe.py``) on a Kaggle session of two T4s at the ``kaggle_t4`` tier, one worker per GPU, the jobs in order of importance so that a session cut short keeps the ones that matter: at seed 0 and then seed 1, all Muon, all AdamW, all Muon with the first mixture's router input centred, all Muon at half its rate, Muon on the experts alone, and all AdamW centred; at seed 2 the first three. Written by ``make_kernel.py``, which puts ``probe.py`` and the hashes of the thesis modules it uses in."""

import hashlib
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

HASHES = {'model': 'd4580dbb1eb0a04e42ea96d47d95044e4b7ded1700ad15e3e6cb91e70559c29c', 'optim': '369c3c41565af44273deef5b6a5b7d5fcc46b0589fbe92d3a1bb4ca238bac64f', 'telemetry': '96bc3fc5ae51894cfe4cb4db50f9a2ac1d698992d84c8163b0f4e3c72727ac53', 'tier': '2a53a64ed6c6c9a71bfa3f3018052d4bf13a8aa330a444f8fde2b6dfd2e421a1', 'train': '70177de0a7a72f7820ad2935ce3e215729e71c005346d33ea54b8ca3aa46903d', 'data': '706c25ed2b181669e3b3547135c89f05afc9c5e0fbf7cc161e99f7e26d741f91', 'checkpoint': '0bf07ed269f5249aaff77029b99a90ca8fc54a50f3c503a2f84edba94cd026d3'}
PROBE = r'''"""One training run of the thesis model with every mixture's per-expert score offset read beside the thesis telemetry, and, optionally, the first mixture's router fed its inputs minus their batch mean.

The question is the step the second chain's logs leave open. Across its 27 balanced runs the first mixture's integrated bias span follows the spread of the router's mean score across experts (`mean_score_span`), not the router-input cosine; the plan's H6 names the route between the two: a direction every token's input shares adds a fixed amount to each expert's logit, and the bias cancels it. Since the router is linear, the batch-mean logit of expert e is exactly w_e . x_bar, the router row times the batch-mean input, so the offset factors into the length of the mean input (the shared direction's strength, which the cosine reads) and how unevenly the rows lean along it. The logs keep neither per expert. This run keeps, per logged step and mixture: every expert's bias, batch-mean logit and batch-mean score, the load it took and the load its scores alone would give, the mean input's length and the mean length of an input, every router row's norm and its component along the mean input, and the offset the uncentred mean input would give (the same as the mean logit unless the input is centred).

With ``--centre`` the first mixture's router scores each token's input minus the batch mean of the inputs, so every expert's mean logit is zero by construction and no shared direction reaches its scores; the experts, the shared expert and every other mixture see the inputs as before. That is an intervention on the model, used here as a test of the mechanism and not as a proposal: if the span Muon adds is the bias cancelling the offset the mean input puts in the scores, the centred run's first-mixture span should fall to AdamW's.

Everything else is the thesis code through imports of ``tfg`` (the model, the partitions, the data order, the schedule, clipping, the balancer, the telemetry, the trainer); nothing under ``tfg`` imports from here. Eager, as on the T4, where the experts run slice by slice.

Usage, from ``code/``: PYTHONPATH=. uv run python ai-experiments/rq3_reading_2026_09/offset_probe/probe.py OUT --partition muon --seed 0 [--muon-lr-multiplier 0.5] [--centre] [--steps 3000] [--tier kaggle_t4]
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
    args = parser.parse_args()

    name = args.partition
    if args.muon_lr_multiplier != 1:
        name += f"-muonlr{args.muon_lr_multiplier:g}x"
    if args.centre:
        name += "-centred"
    run_dir = args.out / f"{args.tier}-{name}-s{args.seed}"
    if (run_dir / "probe.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    readings = Readings(run_dir / "offsets.npz")

    MixtureOfExperts.route = route

    def write(self: Telemetry, step: int, before) -> None:
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
    began = time.perf_counter()
    records = trainer.train(
        tier=args.tier,
        partition=args.partition,
        steps=args.steps,
        adamw_lr=ADAMW_LR,
        muon_lr=MUON_LR,
        muon_lr_multiplier=args.muon_lr_multiplier,
        seed=args.seed,
        eval_batches=args.eval_batches,
        checkpoint_every=args.steps,
        run_dir=run_dir,
    )
    readings.save()
    (run_dir / "probe.json").write_text(
        json.dumps(
            {
                "partition": args.partition,
                "seed": args.seed,
                "muon_lr_multiplier": args.muon_lr_multiplier,
                "centre": args.centre,
                "steps": args.steps,
                "tier": args.tier,
                "val_loss": next((r["val_loss"] for r in reversed(records) if "val_loss" in r), None),
                "seconds": time.perf_counter() - began,
            }
        ),
        encoding="utf-8",
    )
    for saved in run_dir.glob("*.pt"):  # the saves are not needed and would fill the session's output
        saved.unlink()


if __name__ == "__main__":
    main()
'''

CODE = Path("/tmp/code")  # outside /kaggle/working, so the venv is not saved as output
OUT = Path("/kaggle/working")
STEPS = 3000
DEADLINE = 9.5 * 3600  # seconds after which no new run starts, to stay inside the session
ARMS = [  # partition, Muon's rate multiplier, centred
    ("muon", 1.0, False),
    ("adamw", 1.0, False),
    ("muon", 1.0, True),
    ("muon", 0.5, False),
    ("muon_experts", 1.0, False),
    ("adamw", 1.0, True),
]
JOBS = [(*arm, seed) for seed in (0, 1) for arm in ARMS] + [(*arm, 2) for arm in ARMS[:3]]
start = time.perf_counter()


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=CODE if CODE.exists() else None, check=True)


sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv")
root = next(Path("/kaggle/input").rglob("pyproject.toml")).parent
shutil.copytree(root, CODE, dirs_exist_ok=True)
for name, expected in HASHES.items():
    found = hashlib.sha256((CODE / "tfg" / f"{name}.py").read_bytes()).hexdigest()
    print(f"tfg/{name}.py: {'the same as on the laptop' if found == expected else 'DIFFERENT from the laptop'}", flush=True)
(CODE / "probe.py").write_text(PROBE, encoding="utf-8")
sh("pip install -q uv && uv --version && uv sync --frozen")
sh('uv run python -c "from tfg.data import download; download(train_shards=3, validation_shards=1)" && ls data/fineweb-edu')

gpus = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True, check=True).stdout)
print(f"{gpus} GPUs, {len(JOBS)} runs", flush=True)
queue, lock = list(JOBS), threading.Lock()


def worker(gpu: int) -> None:
    while True:
        with lock:
            if not queue or time.perf_counter() - start > DEADLINE:
                return
            partition, multiplier, centre, seed = queue.pop(0)
        began = time.perf_counter()
        label = f"{partition}-{multiplier:g}-{'centred' if centre else 'plain'}-s{seed}"
        log = OUT / f"log-{label}.txt"
        flags = f"--partition {partition} --seed {seed} --muon-lr-multiplier {multiplier} --steps {STEPS}{' --centre' if centre else ''}"
        cmd = f"PYTHONPATH=. uv run python probe.py {OUT}/runs {flags} > {log} 2>&1"
        code = subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=CODE, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=False).returncode
        tail = log.read_text(encoding="utf-8").splitlines()[-1:] if log.exists() else []
        print(f"gpu {gpu}: {label}: exit {code} in {time.perf_counter() - began:.0f}s {str(tail)[:300]}", flush=True)


workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(max(gpus, 1))]
for thread in workers:
    thread.start()
for thread in workers:
    thread.join()
print(f"done in {time.perf_counter() - start:.0f}s; not run: {queue}", flush=True)
