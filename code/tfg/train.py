"""Train the language model and write one JSON line of metrics per log step.

The loop is the part every experiment shares: forward under bf16 autocast on CUDA with float32 master weights, cross-entropy, gradient clipping, the optimizer steps, the loss-free balancing step, and the learning-rate schedule: a linear warmup, then cosine decay or a constant rate. What differs between arms, which optimizer trains which parameter group, is the partition built in ``tfg.optim``. Each optimizer's parameters are clipped at their own threshold. A log line carries, for every step since the previous one, the training loss and each optimizer's pre-clip gradient norm, so the analysis counts loss spikes and takes norm quantiles over every step of a run. Log steps are also watched by ``tfg.telemetry``, which writes the routing readings of every mixture and the weight readings of every parameter group to a second file, and every ``checkpoint_every`` steps the run is saved by ``tfg.checkpoint`` so it can be resumed; one percent into the run the model alone is saved too, the early point the annex's router saturation compares the end with. With ``compiled`` the loss of a batch, the model and its cross-entropy, runs through ``torch.compile``, in training and in evaluation; the model object itself stays eager, so checkpoints and telemetry see it as usual. With ``recompute`` set to K the backward pass recomputes the first K mixture blocks instead of keeping their activations, for a tier that does not fit in memory otherwise. A run whose training loss or gradient norm stops being a number has diverged, because the update that follows leaves NaNs in the weights: it ends on that step with a last record whose validation loss is NaN, which is what marks it, and a relaunch leaves it as it is. Six more factors live here: ``balanced`` off leaves the balancer's biases at zero, ``router_lr_multiplier`` scales the routers' learning rate, ``lr_multiplier`` scales both tuned rates, which is the stress grid's factor, ``muon_lr_multiplier`` scales Muon's alone, which puts it at AdamW's rate in the RQ3 arm that asks whether Muon's larger rate is what raises the span, ``balancer_multiplier`` scales the balancer's update rate, which the stress grid's control arm ties to the rate multiplier, and ``balance_loss`` is the weight of DeepSeek-V3's sequence-wise balance loss beside the bias, added to the cross-entropy the gradient is taken of while the logs and the evaluation read the cross-entropy alone, zero but in the block that asks what the frontier recipes' small loss changes; ``adamw_beta2`` overrides AdamW's second-moment decay for the tuning block's check run. Runs are deterministic: the same seed gives the same numbers, and a resumed run matches an uninterrupted one bit for bit.

Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    V   vocabulary size
"""

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
import torch.nn.functional as F
import tyro
from torch import nn
from torch.optim import Optimizer

from tfg import checkpoint
from tfg.data import DATA_DIR, VOCAB_SIZE, Batches, load_batches
from tfg.model import BIAS_UPDATE_RATE, LanguageModel
from tfg.optim import BETAS, build, parameters
from tfg.telemetry import Telemetry
from tfg.tier import TIERS

# Run.
SEED = 0
STEPS = 1000
LOG_EVERY = 10
EVAL_EVERY = 250  # must be a multiple of LOG_EVERY
EVAL_BATCHES = 32  # a fixed prefix of the validation split, in file order
CHECKPOINT_EVERY = 1000  # must be a multiple of LOG_EVERY
EARLY_CHECKPOINT = 0.01  # of the run's steps: there the model alone is saved, for the annex
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# Optimization.
PARTITION = "adamw"
ADAMW_LR = 1e-3  # both peak rates are tuned per optimizer before any comparison
MUON_LR = 1e-3
ADAMW_BETA2 = BETAS[1]  # AdamW's second-moment decay; the tuning block's check run raises it
LR_MULTIPLIER = 1.0  # on both tuned rates; the stress grid's factor
MUON_LR_MULTIPLIER = 1.0  # on Muon's rate alone, after the one on both
WARMUP_STEPS = 500  # linear warmup, then the schedule; the same count in every block of the plan
SCHEDULES = ("cosine", "constant")  # cosine decay for the comparisons, constant for the stress runs
SCHEDULE = "cosine"
FLOOR = 0.1  # the cosine ends at this fraction of the peak rate on the last step
CLIP_NORM = 1.0  # each optimizer's gradient is clipped at its own threshold, this one by default
BALANCED = True  # off only on the arms that bridge to the no-balancer literature
ROUTER_LR_MULTIPLIER = 1.0  # the router's rate relative to its optimizer's; swept on all-AdamW
BALANCER_MULTIPLIER = 1.0  # on the balancer's update rate; the stress control arm scales it
BALANCE_LOSS = 0.0  # the sequence-wise balance loss's weight; off but in the balance-loss block
# Settings added after the first runs, which had them at these values.
ADDED = {"muon_lr_multiplier": MUON_LR_MULTIPLIER, "balance_loss": BALANCE_LOSS}

# Reproducibility. The CUDA kernels that reduce with atomics differ run to run by about 1e-4 in the loss; their deterministic alternatives cost nothing measurable on a T4. cuBLAS reads its workspace setting once, before its first call, so it is set here rather than in ``train``.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Memory. Segments the caching allocator can grow in place fragment less than fixed ones, which a 32 GB card needs at the mechanism tier; the allocator reads the setting once, at the first allocation on the GPU. It changes where tensors live, never what is computed.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Record(TypedDict):
    """One metrics line: the log step's readings and, for every step since the previous log step (this one last), the training loss and each optimizer's pre-clip gradient norm; in a run with the balance loss, also the log step's sequence-wise balance loss, summed over the mixtures and before its weight."""

    step: int
    loss: float
    lr_factor: float
    grad_norm: float  # over every parameter
    grad_norms: dict[str, list[float]]
    losses: list[float]
    tokens_per_s: float
    sequence_balance: NotRequired[float]
    val_loss: NotRequired[float]


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arm(
    partition: str,
    *,
    balanced: bool,
    router_lr_multiplier: float,
    lr_multiplier: float = 1.0,
    balancer_multiplier: float = 1.0,
    adamw_beta2: float = ADAMW_BETA2,
    muon_lr_multiplier: float = MUON_LR_MULTIPLIER,
    balance_loss: float = BALANCE_LOSS,
) -> str:
    """The name of an arm: the partition, plus what else departs from default."""
    name = partition
    if not balanced:
        name += "-unbalanced"
    if router_lr_multiplier != 1:
        name += f"-router{router_lr_multiplier:g}x"
    if lr_multiplier != 1:
        name += f"-lr{lr_multiplier:g}x"
    if muon_lr_multiplier != 1:
        name += f"-muonlr{muon_lr_multiplier:g}x"
    if balancer_multiplier != 1:
        name += f"-balancer{balancer_multiplier:g}x"
    if adamw_beta2 != ADAMW_BETA2:
        name += f"-beta2{adamw_beta2:g}"
    if balance_loss != BALANCE_LOSS:
        name += f"-balanceloss{balance_loss:g}"
    return name


def autocast(dev: torch.device) -> torch.autocast:
    """bf16 for the matrix multiplies on CUDA, with float32 master weights; a no-op elsewhere."""
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def lr_factor(step: int, *, steps: int, warmup_steps: int, schedule: str) -> float:
    """The fraction of the peak rate at ``step``: a linear warmup, then one under the constant schedule, or a cosine from one down to ``FLOOR`` on the last step under the cosine schedule.

    A pure function of the step, so a resumed run needs no scheduler state.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = (step + 1 - warmup_steps) / (steps - warmup_steps)  # (0, 1], one on the last step
    return FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # [B, T, V], [B, T]
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())  # []


Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # inputs, targets [B, T] each -> []


def loss_function(model: LanguageModel, *, compiled: bool = False) -> Loss:
    """The model's cross-entropy on a batch, as one function of the inputs and the targets. With ``compiled`` it runs through ``torch.compile``, which then fuses the loss with the output head in one graph instead of writing the logits, the largest tensor of a step, to memory in float32 to read them back (the mechanism tier's peak went from 44 to 32 GB on the H100). The model object itself stays eager, so checkpoints, telemetry, and anything else that calls it see it as usual."""

    def loss_of(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # [B, T] each
        return cross_entropy(model(inputs), targets)  # []

    return torch.compile(loss_of) if compiled else loss_of


def to_device(
    batch: tuple[torch.Tensor, torch.Tensor], dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:  # [B, T] each
    inputs, targets = batch
    return inputs.to(dev, non_blocking=True), targets.to(dev, non_blocking=True)


def train_step(
    model: LanguageModel,
    optimizers: dict[str, Optimizer],
    batch: tuple[torch.Tensor, torch.Tensor],  # [B, T] each
    *,
    factor: float,
    clip_norms: dict[str, float] | None = None,
    loss_of: Loss | None = None,
    balanced: bool = BALANCED,
    balancer_multiplier: float = BALANCER_MULTIPLIER,
    balance_loss: float = BALANCE_LOSS,
) -> tuple[float, dict[str, float]]:
    """One step of every optimizer on one batch; returns the loss and, per optimizer, the pre-clip norm of its parameters' gradient.

    Each optimizer's gradient is clipped at its threshold in ``clip_norms``, by default ``CLIP_NORM`` for every one. ``loss_of`` is the loss to use when it is not the model's cross-entropy run as it is: the compiled one. Without ``balanced`` the balancer's biases are left where they are, at zero; with it, they move at ``balancer_multiplier`` times the default rate. A ``balance_loss`` above zero adds the mixtures' sequence-wise balance loss at that weight to the cross-entropy before the backward pass, for a model built to keep it; the loss returned is the cross-entropy alone. The loss and the norms are read off the GPU once the step is queued, since reading a number waits for the GPU to reach it and the optimizers' kernels are queued faster while it is still busy.
    """
    loss_of = loss_of or loss_function(model)
    clip_norms = clip_norms or dict.fromkeys(optimizers, CLIP_NORM)
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * factor
    inputs, targets = batch
    with autocast(inputs.device):
        loss = loss_of(inputs, targets)  # []
    total = loss + balance_loss * model.sequence_balance() if balance_loss else loss  # []
    total.backward()
    norms = {
        kind: nn.utils.clip_grad_norm_(parameters(optimizer), clip_norms[kind])  # []
        for kind, optimizer in optimizers.items()
    }
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if balanced:
        model.update_biases(BIAS_UPDATE_RATE * balancer_multiplier)
    return loss.item(), {kind: norm.item() for kind, norm in norms.items()}


@torch.no_grad()
def evaluate(
    model: LanguageModel, batches: Batches, num_batches: int, *, loss_of: Loss | None = None
) -> float:
    """Mean loss over the first ``num_batches`` batches in file order; leaves the model training. ``loss_of`` is as in ``train_step``."""
    loss_of = loss_of or loss_function(model)
    dev = next(model.parameters()).device
    model.eval()
    total = 0.0
    for step in range(num_batches):
        inputs, targets = to_device(batches[step], dev)
        with autocast(dev):
            total += loss_of(inputs, targets).item()
    model.train()
    return total / num_batches


def check_schedule(
    schedule: str, *, steps: int, warmup_steps: int, log_every: int, every: tuple[int, int]
) -> None:
    """Refuse a schedule that is not one of ``SCHEDULES``, a cosine decay with no step left after the warmup, and evaluation and checkpoint intervals, ``every``, that are not multiples of ``log_every``."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule {schedule!r} is not one of {SCHEDULES}")
    if schedule == "cosine" and warmup_steps >= steps:
        raise ValueError(f"cosine decay needs warmup_steps {warmup_steps} below steps {steps}")
    if every[0] % log_every or every[1] % log_every:
        raise ValueError(
            f"eval_every {every[0]} and checkpoint_every {every[1]} "
            f"must be multiples of log_every {log_every}"
        )


def last_record(run_dir: Path) -> Record | None:
    """The last record of the run's metrics, none when it has no record yet."""
    path = run_dir / "metrics.jsonl"
    lines = checkpoint.records(path.read_text(encoding="utf-8")) if path.exists() else []
    return json.loads(lines[-1]) if lines else None


def diverged(run_dir: Path) -> bool:
    """Whether the run in ``run_dir`` ended on a loss or a gradient that was not a number: its last record says so with a NaN validation loss."""
    last = last_record(run_dir)
    return last is not None and math.isnan(last.get("val_loss", 0.0))


def complete(run_dir: Path) -> bool:
    """Whether the run's metrics reach the last step of the settings it was made with."""
    last, path = last_record(run_dir), run_dir / "settings.json"
    if last is None or not path.exists():
        return False
    return last["step"] == json.loads(path.read_text(encoding="utf-8"))["steps"]


def finished(run_dir: Path, *, steps: int) -> bool:
    """Whether a relaunch has nothing left to train: the run diverged, or it is logged to its last step, no more than those steps are asked for, and its checkpoint stands there too or is gone. A run killed while it wrote its last checkpoint has a whole log and a checkpoint behind it: it goes on from that checkpoint and trains its tail again, so that the model on disk is the one the log ends on. With the checkpoint gone there is nothing to go on from, and a relaunch would train the run again from nothing."""
    if diverged(run_dir):
        return True
    last, saved = last_record(run_dir), checkpoint.saved_step(run_dir)
    if last is None or not complete(run_dir):
        return False
    return saved is None or (steps <= last["step"] and saved == last["step"])


def fixed(settings: dict[str, object], *, extendable: bool) -> dict[str, object]:
    """The settings a relaunched run may not change: everything but the device and how often it checkpoints, which change nothing it computes, and, for a run that is ``extendable``, one under the constant schedule with a checkpoint to go on from, the total steps, because then the schedule does not depend on them and the run can be trained longer: it ends on the weights of an uninterrupted run of the new length, with the record of its old last step kept in its log and its early save where the old length put it. Cosine decay is shaped by the total steps, so they stay fixed."""
    free = {"device", "checkpoint_every"}
    if extendable and settings["schedule"] == "constant":
        free.add("steps")
    return {key: value for key, value in settings.items() if key not in free}


def record_settings(
    run_dir: Path, settings: dict[str, object], *, resume: bool, over: bool = False
) -> None:
    """Settle the run's directory and write its settings. On resume, and for a run that is ``over``, the settings are first checked against the recorded ones; a run that is over keeps the settings file it has. A run that resumes must have both logs at its checkpoint or past it: a record is written before the checkpoint of its step, so a log that stops short of the checkpoint has lost records, which a machine that went down with the log unwritten can do, and the run would go on with a hole in its log. It must also be asked for more steps than its checkpoint holds: cut to the step it would go on from, or to one before it, it would train nothing and end without the evaluation that closes a run. A run that starts over first loses every file of the run it replaces. A setting a run recorded before it existed, one of ``ADDED``, is read at the value every run had then."""
    path = run_dir / "settings.json"
    if resume or over:
        recorded: dict[str, object] = {**ADDED, **json.loads(path.read_text(encoding="utf-8"))}
        recorded = fixed(recorded, extendable=not over)
        wanted = fixed(settings, extendable=not over)
        if recorded != wanted:
            changed = sorted(
                key
                for key in recorded.keys() | wanted.keys()
                if recorded.get(key) != wanted.get(key)
            )
            raise ValueError(
                f"settings differ from the run checkpointed in {run_dir}: {', '.join(changed)}"
            )
    if over:
        return
    steps = settings["steps"]
    assert isinstance(steps, int)
    if not resume:
        checkpoint.discard(run_dir)
    else:
        saved = checkpoint.saved_step(run_dir)
        assert saved is not None  # a resume goes on from a checkpoint
        for name in ("metrics.jsonl", "routing.jsonl"):
            last = checkpoint.last_step(run_dir / name)
            if last is None or last < saved:
                raise ValueError(
                    f"{run_dir} is checkpointed at step {saved} and its {name} stops at {last}: "
                    "train it again without --resume"
                )
        if saved >= steps:
            raise ValueError(
                f"{run_dir} is checkpointed at step {saved}, not short of the {steps} asked for"
            )
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def restore(run_dir: Path, model: LanguageModel, optimizers: dict[str, Optimizer]) -> int:
    """The step to go on from: the checkpoint's, with the logs rewound to it, or zero for a run that starts over, whose directory ``record_settings`` emptied."""
    start = checkpoint.load(run_dir, model, optimizers)
    for name in ("metrics.jsonl", "routing.jsonl"):
        checkpoint.rewind(run_dir / name, start)
    return start


def train(
    *,
    tier: str,
    partition: str = PARTITION,
    steps: int = STEPS,
    batch_size: int | None = None,
    adamw_lr: float = ADAMW_LR,
    muon_lr: float = MUON_LR,
    adamw_beta2: float = ADAMW_BETA2,
    lr_multiplier: float = LR_MULTIPLIER,
    muon_lr_multiplier: float = MUON_LR_MULTIPLIER,
    adamw_clip: float = CLIP_NORM,
    muon_clip: float = CLIP_NORM,
    balanced: bool = BALANCED,
    router_lr_multiplier: float = ROUTER_LR_MULTIPLIER,
    balancer_multiplier: float = BALANCER_MULTIPLIER,
    balance_loss: float = BALANCE_LOSS,
    warmup_steps: int = WARMUP_STEPS,
    schedule: str = SCHEDULE,
    log_every: int = LOG_EVERY,
    eval_every: int = EVAL_EVERY,
    eval_batches: int = EVAL_BATCHES,
    checkpoint_every: int = CHECKPOINT_EVERY,
    seed: int = SEED,
    compiled: bool = False,
    recompute: int = 0,
    run_dir: Path | None = None,
    data_dir: Path = DATA_DIR,
    resume: bool = False,
) -> list[Record]:
    """Train for ``steps`` steps; write and return one metrics record per log step taken.

    The model shape and, unless given, the batch size come from the named tier. Both tuned rates are scaled by ``lr_multiplier``, and Muon's once more by ``muon_lr_multiplier``; ``adamw_clip`` and ``muon_clip`` are the thresholds each optimizer's gradient is clipped at, Muon's shared with its control; ``balancer_multiplier`` scales the balancer's update rate; ``balance_loss`` is the weight of the sequence-wise balance loss, zero for none; ``adamw_beta2`` is AdamW's second-moment decay, which only the check run changes. The run lives in ``run_dir``, by default ``runs/<tier>-<arm>-s<seed>``. With ``resume`` the run continues from the checkpoint there when there is one, and the settings must match the ones saved there, the number of training windows among them, since the data order depends on the shards present; a run that diverged is over and trains no further, and neither does a run logged to its last step, whose files a relaunch leaves as they are, even when its checkpoint is gone and it would otherwise start again over its own logs; only when its checkpoint is behind its log, the run having been killed while it wrote its last one, does it train its tail again. Without ``resume`` the run starts over in an emptied directory: the logs, saved states, and evaluations of the run it replaces are removed. Evaluation and checkpoints happen on log steps, so their intervals must be multiples of ``log_every``.
    """
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r} is not one of {sorted(TIERS)}")
    # Positive, which a NaN is not either.
    positive = (lr_multiplier, muon_lr_multiplier, adamw_clip, muon_clip, balancer_multiplier)
    if not all(s > 0 for s in positive):
        raise ValueError(
            "lr_multiplier, muon_lr_multiplier, adamw_clip, muon_clip, and balancer_multiplier "
            "must be positive"
        )
    if not 0 < adamw_beta2 < 1:
        raise ValueError(f"adamw_beta2 {adamw_beta2} must lie in (0, 1)")
    if not balance_loss >= 0:  # a NaN is not either
        raise ValueError(f"balance_loss {balance_loss} must be zero or positive")
    check_schedule(
        schedule,
        steps=steps,
        warmup_steps=warmup_steps,
        log_every=log_every,
        every=(eval_every, checkpoint_every),
    )
    shape = TIERS[tier]
    batch_size = batch_size or shape.batch_size
    name = arm(
        partition,
        balanced=balanced,
        router_lr_multiplier=router_lr_multiplier,
        lr_multiplier=lr_multiplier,
        balancer_multiplier=balancer_multiplier,
        adamw_beta2=adamw_beta2,
        muon_lr_multiplier=muon_lr_multiplier,
        balance_loss=balance_loss,
    )
    run_dir = run_dir or RUNS_DIR / f"{tier}-{name}-s{seed}"
    dev = device()
    torch.use_deterministic_algorithms(True)  # raises on a kernel without a deterministic path
    # Deterministic mode also fills every new tensor with NaN, in case a kernel reads memory nothing wrote; none does in this model (the H100 gave the same losses with and without the fill), and the fill cost about 5 percent of a step there.
    torch.utils.deterministic.fill_uninitialized_memory = False  # pyright: ignore[reportAttributeAccessIssue]  # a property torch gives the module at import
    torch.manual_seed(seed)
    model = LanguageModel(VOCAB_SIZE, shape, recompute=recompute, balance_loss=balance_loss > 0)
    model = model.to(dev)
    optimizers = build(
        model,
        partition,
        adamw_lr=adamw_lr * lr_multiplier,
        muon_lr=muon_lr * lr_multiplier * muon_lr_multiplier,
        router_lr_multiplier=router_lr_multiplier,
        adamw_beta2=adamw_beta2,
    )
    clip_norms = {"adamw": adamw_clip, "muon": muon_clip, "momentum": muon_clip}
    train_batches = load_batches(
        "train", batch_size, shape.context_len, seed=seed, data_dir=data_dir
    )
    val_batches = load_batches("validation", batch_size, shape.context_len, data_dir=data_dir)
    eval_batches = min(eval_batches, len(val_batches))

    run_dir.mkdir(parents=True, exist_ok=True)
    resuming = resume and (run_dir / checkpoint.FILE).exists()
    over = resume and finished(run_dir, steps=steps)
    record_settings(
        run_dir,
        {
            "tier": tier,
            "partition": partition,
            "steps": steps,
            "batch_size": batch_size,
            "adamw_lr": adamw_lr,
            "muon_lr": muon_lr,
            "adamw_beta2": adamw_beta2,
            "lr_multiplier": lr_multiplier,
            "muon_lr_multiplier": muon_lr_multiplier,
            "adamw_clip": adamw_clip,
            "muon_clip": muon_clip,
            "balanced": balanced,
            "router_lr_multiplier": router_lr_multiplier,
            "balancer_multiplier": balancer_multiplier,
            "balance_loss": balance_loss,
            "warmup_steps": warmup_steps,
            "log_every": log_every,
            "eval_every": eval_every,
            "eval_batches": eval_batches,
            "checkpoint_every": checkpoint_every,
            "schedule": schedule,
            "seed": seed,
            "compiled": compiled,
            "recompute": recompute,
            "train_windows": len(train_batches.windows),
            "device": dev.type,
        },
        resume=resuming,
        over=over,
    )
    if over:
        return []
    start = restore(run_dir, model, optimizers)

    records: list[Record] = []
    early = max(1, round(EARLY_CHECKPOINT * steps))  # the step the model-only save is taken at
    loss_of = loss_function(model, compiled=compiled)
    telemetry = Telemetry(model, run_dir / "routing.jsonl", resume=resuming)
    with (run_dir / "metrics.jsonl").open("a" if resuming else "w", encoding="utf-8") as metrics:
        clock, logged = time.perf_counter(), start
        losses: list[float] = []  # of every step since the last log step
        grad_norms: dict[str, list[float]] = {kind: [] for kind in optimizers}  # the same
        for step in range(start, steps):
            done = step + 1
            logging = done % log_every == 0 or done == steps
            factor = lr_factor(step, steps=steps, warmup_steps=warmup_steps, schedule=schedule)
            before = telemetry.snapshot() if logging else None
            loss, norms = train_step(
                model,
                optimizers,
                to_device(train_batches[step], dev),
                factor=factor,
                clip_norms=clip_norms,
                loss_of=loss_of,
                balanced=balanced,
                balancer_multiplier=balancer_multiplier,
                balance_loss=balance_loss,
            )
            over = not all(map(math.isfinite, (loss, *norms.values())))  # diverged: the last step
            if before is not None and not over:  # a model of NaNs has no readings
                telemetry.write(done, before)
            losses.append(loss)
            for kind, norm in norms.items():
                grad_norms[kind].append(norm)
            if done == early and not over:
                checkpoint.save_model(run_dir, model, step=done)
            if not (logging or over):
                continue
            record: Record = {
                "step": done,
                "loss": loss,
                "lr_factor": factor,
                "grad_norm": math.sqrt(sum(norm * norm for norm in norms.values())),
                "grad_norms": grad_norms,
                "losses": losses,
                "tokens_per_s": (done - logged)
                * batch_size
                * shape.context_len
                / (time.perf_counter() - clock),
            }
            losses, grad_norms = [], {kind: [] for kind in optimizers}
            if balance_loss:
                record["sequence_balance"] = model.sequence_balance().item()
            if over:
                record["val_loss"] = math.nan  # the mark of a diverged run
            elif done % eval_every == 0 or done == steps:
                record["val_loss"] = evaluate(model, val_batches, eval_batches, loss_of=loss_of)
            records.append(record)
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record))
            if over:
                break
            if done % checkpoint_every == 0 or done == steps:
                checkpoint.save(run_dir, model, optimizers, step=done)
            clock, logged = time.perf_counter(), done
    return records


def main(args: list[str] | None = None) -> list[Record]:
    """The command line: every setting of ``train`` as a flag, ``--help`` lists them."""
    return tyro.cli(train, args=args)


if __name__ == "__main__":
    main()
