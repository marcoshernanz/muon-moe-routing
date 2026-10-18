"""The plan's run program: every block of runs as one command, in the order the gates need them.

A block is the list of runs the plan fixes (``thesis/PLAN.md``, sections 7 and 12), trained one after another into one directory: the tuning sweeps with the control's sweep for the frontier and the β2 check, Gate 1's pair, the RQ2 stress grid with its clip calibration and the sixteen-fold level for the partitions that survive the eight-fold one, the RQ3 arms on seeds of their own, the confirmation tier with its rate re-check, the balancer block, (a) and (b) with the bias at its rate and four times faster, on seeds of its own again, and the balance-loss block, (a), (b) and (c) with the frontier recipes' sequence-wise balance loss beside the bias, on RQ3's seeds, each run read against RQ3's run of its arm and seed. Every run is an ordinary ``tfg.train`` run under its arm's name, resumed if it was interrupted and skipped if it finished, so a block is relaunched with the same command after a preemption and continues where it stopped; a run that diverged is over and is left as it is; a run whose settings changed since it was made is refused by the trainer rather than reused. A machine with several GPUs runs the same command once per GPU (``tfg.workers``): each worker trains the runs no other one holds, and where a block goes on to read what its runs produced, the rates or the clips, it first waits for the runs the others hold. What one block needs from another is read from files: the tuned rates from the tuning block's ``rates.json``, the stress clips from the calibration runs' endpoints. Steps follow from the plan's token counts and the tier's tokens per step. The ``evaluate`` block runs the specialization panel and the annex over every finished run of a block, which needs the checkpoints and so happens where the runs are, before they come home.
"""

import gc
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import torch
import tyro

from tfg import checkpoint, workers
from tfg.analysis import endpoints, run_dirs, status_of
from tfg.annex import FILE as ANNEX_FILE
from tfg.annex import evaluate as evaluate_annex
from tfg.data import DATA_DIR
from tfg.panel import DOMAINS, PANEL_DIR
from tfg.panel import evaluate as evaluate_panel
from tfg.sweep import best, beyond, sweep
from tfg.tier import TIERS
from tfg.train import (
    ADAMW_BETA2,
    BALANCE_LOSS,
    BALANCED,
    BALANCER_MULTIPLIER,
    CHECKPOINT_EVERY,
    CLIP_NORM,
    EVAL_BATCHES,
    EVAL_EVERY,
    LOG_EVERY,
    LR_MULTIPLIER,
    MUON_LR_MULTIPLIER,
    ROUTER_LR_MULTIPLIER,
    RUNS_DIR,
    SCHEDULE,
    WARMUP_STEPS,
    arm,
    train,
)

# The plan's numbers (section 6 and section 7).
TIER = "mechanism"  # Gate 1's, the stress grid's, and RQ3's tier
CONFIRMATION_TIER = "confirmation"
TOKENS = {  # per run, by block
    "tuning": 500_000_000,
    "rq1": 1_000_000_000,  # Gate 1's
    "rq2": 500_000_000,
    "rq3": 1_000_000_000,
    "recheck": 1_000_000_000,
    "confirmation": 3_000_000_000,
    "balancer": 1_000_000_000,
    "balance_loss": 1_000_000_000,
}
ADAMW_RATES = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)  # a factor of two apart, around the pilot's optima
MUON_RATES = (2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2)
CONTROL_RATES = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)  # the control's, for the frontier
BETA2_CHECK = 0.98  # the tuning block's one AdamW run at a higher second-moment decay
SEEDS = (0, 1, 2)
RQ3_SEEDS = (3, 4, 5)  # seeds no earlier run had: RQ3's predictions come from the runs at 0 to 2
BALANCER_SEEDS = (6, 7, 8)  # the same for the balancer block, whose predictions come from 3 to 5
GATE1 = ("adamw", "muon")  # partitions (a) and (b), Gate 1's pair
RQ3 = (  # (a) to (e), (g), (h), and (i)
    "adamw",
    "muon",
    "muon_except_router",
    "muon_experts",
    "muon_attention",
    "muon_router",
    "muon_dense",
    "muon_attention_dense",
)
UNBALANCED = ("adamw", "muon")  # the arms without the balancer, bridging to the no-balancer papers
STRESSED = ("adamw", "muon", "muon_except_router")  # the stress grid's partitions
MULTIPLIERS = (1.0, 2.0, 4.0, 8.0)  # of both tuned rates
EXTENSION = 16.0  # one more multiplier, for a partition none of whose runs at the top one diverged
CONTROL = ("muon", 4.0)  # the balancer-speed control: this partition at this multiplier
CLIP_FACTOR = 2.0  # of the post-warmup median pre-clip norm in the calibration run
# Of the mechanism tier's optimum, re-checked at the confirmation tier: at and below it, since the best
# rate falls as a run gets longer.
RECHECK = (0.125, 0.25, 0.5, 1.0)
HORIZON = 0.7  # on the re-check's winners for the 3B runs: 3 ** -0.32, the best rate's fall from 1B
FAST_BALANCER = 4.0  # the balancer block's faster bias, as the stress grid's control and the probe
CONFIRMED = ("adamw", "muon", "muon_except_router")
RECIPE_LOSS = 1e-4  # the balance-loss block's weight on the loss, DeepSeek-V3's and V4's
RATES_FILE = "rates.json"  # the tuned rates, written by the tuning block and the re-check
PANEL_FILE = "specialization.json"


@dataclass(frozen=True)
class Settings:
    """What every block shares: where the runs go, the intervals, the compiler, how many mixture blocks the backward pass recomputes (for a tier that does not fit otherwise), and the data."""

    runs_dir: Path = RUNS_DIR
    compiled: bool = True
    recompute: int = 0
    warmup_steps: int = WARMUP_STEPS
    log_every: int = LOG_EVERY
    eval_every: int = EVAL_EVERY
    eval_batches: int = EVAL_BATCHES
    checkpoint_every: int = CHECKPOINT_EVERY
    data_dir: Path = DATA_DIR

    @property
    def training(self) -> dict[str, Any]:
        """The trainer's keyword arguments: everything but where the runs go."""
        return {key: value for key, value in asdict(self).items() if key != "runs_dir"}


Shared = Annotated[Settings, tyro.conf.OmitArgPrefixes]  # the flags without a prefix
DEFAULTS = Settings()


@dataclass(frozen=True)
class Run:
    """One run of a block: its partition and seed, plus what departs from the trainer's defaults."""

    partition: str
    seed: int
    balanced: bool = BALANCED
    router_lr_multiplier: float = ROUTER_LR_MULTIPLIER
    lr_multiplier: float = LR_MULTIPLIER
    balancer_multiplier: float = BALANCER_MULTIPLIER
    adamw_beta2: float = ADAMW_BETA2
    muon_lr_multiplier: float = MUON_LR_MULTIPLIER
    adamw_clip: float = CLIP_NORM
    muon_clip: float = CLIP_NORM
    balance_loss: float = BALANCE_LOSS

    def name(self, tier: str) -> str:
        """The run's directory name, as the trainer would name it."""
        label = arm(
            self.partition,
            balanced=self.balanced,
            router_lr_multiplier=self.router_lr_multiplier,
            lr_multiplier=self.lr_multiplier,
            balancer_multiplier=self.balancer_multiplier,
            adamw_beta2=self.adamw_beta2,
            muon_lr_multiplier=self.muon_lr_multiplier,
            balance_loss=self.balance_loss,
        )
        return f"{tier}-{label}-s{self.seed}"


def steps_of(tier: str, tokens: int) -> int:
    """The steps that train ``tokens`` at the tier's batch."""
    return round(tokens / TIERS[tier].tokens_per_step)


def launch(
    shared: Settings,
    block_dir: Path,
    runs: list[Run],
    *,
    tier: str,
    steps: int,
    rates: dict[str, float],
    schedule: str = SCHEDULE,
    wait: bool = False,
) -> list[Path]:
    """Train the runs one after another under ``block_dir``, each resumed if it was interrupted and skipped if it finished or diverged; return their directories. A run another worker is training is skipped too, or, with ``wait``, waited for, so that every run is over when this returns, which a block needs before it reads them."""
    dirs = [block_dir / run.name(tier) for run in runs]
    run_of = dict(zip(dirs, runs))

    def run(run_dir: Path) -> None:
        print(f"run {dirs.index(run_dir) + 1} of {len(dirs)}: {run_dir}", flush=True)
        # A worker trains its runs one after another in one process; each starts on an empty cache, as its first did, since the last run's freed blocks would sit beside the new ones and bring AdamW's partition at the mechanism tier to a 32 GB card's edge.
        gc.collect()
        torch.cuda.empty_cache()
        records = train(
            tier=tier,
            steps=steps,
            adamw_lr=rates["adamw"],
            muon_lr=rates["muon"],
            schedule=schedule,
            run_dir=run_dir,
            resume=True,
            **asdict(run_of[run_dir]),
            **shared.training,
        )
        if not records:
            print(f"{run_dir.name}: {status_of(run_dir)} before", flush=True)

    workers.each(dirs, run, wait=wait)
    return dirs


def tune(
    shared: Settings,
    block_dir: Path,
    grids: dict[str, list[float]],
    *,
    tier: str,
    steps: int,
    stop_at_edge: bool = False,
) -> dict[str, float]:
    """Sweep AdamW's rate, then each other optimizer's with AdamW's tables and norms at the rate that won, each grid extended by one point when its best rate is an edge; write every winner to the block's ``rates.json`` and return them. With ``stop_at_edge``, when AdamW's or Muon's best rate is still an edge after that point, stop before the next sweep and before ``rates.json``, so that no later sweep or block trains at a rate the grid does not bracket: widen that grid and run the block again, which skips the finished rates. The control's winner is read by no later block, so its sweep goes on."""
    rates: dict[str, float] = {}
    for optimizer, grid in grids.items():
        losses = sweep(
            tier=tier,
            optimizer=optimizer,
            rates=grid,
            steps=steps,
            adamw_lr=rates.get("adamw", grid[0]),  # held at its winner during Muon's sweep
            sweep_dir=block_dir / optimizer,
            **shared.training,
        )
        rates[optimizer] = best(losses)
        if stop_at_edge and optimizer in ("adamw", "muon") and beyond(losses) is not None:
            raise SystemExit(
                f"{optimizer}'s best rate, {rates[optimizer]:g}, is still an edge of its grid "
                f"under {block_dir}: widen the grid and run the block again"
            )
    (block_dir / RATES_FILE).write_text(json.dumps(rates, indent=2), encoding="utf-8")
    print(f"tuned rates {rates}, written to {block_dir / RATES_FILE}", flush=True)
    return rates


def rates_of(shared: Settings, adamw_lr: float | None, muon_lr: float | None) -> dict[str, float]:
    """The tuned rates: the ones given, the tuning block's for the ones not."""
    given = {"adamw": adamw_lr, "muon": muon_lr}
    if all(rate is not None for rate in given.values()):
        return {optimizer: float(rate) for optimizer, rate in given.items() if rate is not None}
    path = shared.runs_dir / "tuning" / RATES_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the tuning block, or give both rates")
    tuned = json.loads(path.read_text(encoding="utf-8"))
    return {
        optimizer: float(tuned[optimizer] if rate is None else rate)
        for optimizer, rate in given.items()
    }


def tuning(
    shared: Shared = DEFAULTS,
    *,
    tier: str = TIER,
    adamw_rates: tuple[float, ...] = ADAMW_RATES,
    muon_rates: tuple[float, ...] = MUON_RATES,
    control_rates: tuple[float, ...] = CONTROL_RATES,
    tokens: int = TOKENS["tuning"],
) -> dict[str, float]:
    """Gate 1a's tuning block: both sweeps, the control's sweep for the frontier (``tfg.analysis frontier``), whose winner the later blocks do not read since the control runs at Muon's rate, then the AdamW check run at the higher β2. It stops before writing ``rates.json`` when AdamW's or Muon's best rate is still an edge of its grid after one more point (``tune``), so that a chain goes no further."""
    block_dir = shared.runs_dir / "tuning"
    steps = steps_of(tier, tokens)
    grids = {
        "adamw": list(adamw_rates),
        "muon": list(muon_rates),
        "momentum": list(control_rates),
    }
    rates = tune(shared, block_dir, grids, tier=tier, steps=steps, stop_at_edge=True)
    check = [Run("adamw", SEEDS[0], adamw_beta2=BETA2_CHECK)]
    launch(shared, block_dir, check, tier=tier, steps=steps, rates=rates)
    return rates


def gate1(
    shared: Shared = DEFAULTS,
    *,
    tier: str = TIER,
    adamw_lr: float | None = None,
    muon_lr: float | None = None,
    seeds: tuple[int, ...] = SEEDS,
    tokens: int = TOKENS["rq1"],
) -> list[Path]:
    """Gate 1: partitions (a) and (b) at every seed, into ``runs/rq1``."""
    runs = [Run(partition, seed) for partition in GATE1 for seed in seeds]
    rates = rates_of(shared, adamw_lr, muon_lr)
    return launch(
        shared, shared.runs_dir / "rq1", runs, tier=tier, steps=steps_of(tier, tokens), rates=rates
    )


def rq3_runs(seeds: tuple[int, ...], rates: dict[str, float]) -> list[Run]:
    """The RQ3 runs, seed by seed so that every arm has one seed before any has two: the partitions of ``RQ3``, then all-Muon with Muon's rate brought to AdamW's, which is all-Muon itself when the two rates are one, then the arms without the balancer."""
    equal = rates["adamw"] / rates["muon"]  # Muon's multiplier that gives it AdamW's rate
    runs: list[Run] = []
    for seed in seeds:
        runs += [Run(partition, seed) for partition in RQ3]
        runs += [Run("muon", seed, muon_lr_multiplier=equal)] if equal != 1 else []
        runs += [Run(partition, seed, balanced=False) for partition in UNBALANCED]
    return runs


def rq3(
    shared: Shared = DEFAULTS,
    *,
    tier: str = TIER,
    adamw_lr: float | None = None,
    muon_lr: float | None = None,
    seeds: tuple[int, ...] = RQ3_SEEDS,
    tokens: int = TOKENS["rq3"],
) -> list[Path]:
    """The RQ3 block, into ``runs/rq3``: ``rq3_runs`` at the tuned rates (the rates given, or the tuning block's)."""
    rates = rates_of(shared, adamw_lr, muon_lr)
    runs = rq3_runs(seeds, rates)
    return launch(
        shared, shared.runs_dir / "rq3", runs, tier=tier, steps=steps_of(tier, tokens), rates=rates
    )


def clips_of(run_dir: Path) -> dict[str, float]:
    """Each optimizer's clip for a family: ``CLIP_FACTOR`` times its post-warmup median pre-clip norm in the calibration run; an optimizer the partition does not use keeps the default."""
    values = endpoints(run_dir)
    clips: dict[str, float] = {}
    for kind in ("adamw", "muon"):
        median = values.get(f"grad_norm_median_{kind}")
        clips[kind] = CLIP_FACTOR * median if isinstance(median, float) else CLIP_NORM
    return clips


def rq2(
    shared: Shared = DEFAULTS,
    *,
    tier: str = TIER,
    adamw_lr: float | None = None,
    muon_lr: float | None = None,
    seeds: tuple[int, ...] = SEEDS,
    multipliers: tuple[float, ...] = MULTIPLIERS,
    extension: float | None = EXTENSION,
    tokens: int = TOKENS["rq2"],
) -> list[Path]:
    """The RQ2 stress grid in three stages, each over every partition: the calibration runs at the tuned rates on the constant schedule; the families at every multiplier, seed by seed, with both clips set from their partition's calibration run, and on the control partition, when its multiplier is one of them, the arm whose balancer runs as fast as its rates; and last the ``extension`` multiplier for each family none of whose runs diverged at the highest one. The calibration runs live in their own subdirectory, out of the stress analysis, and one that did not finish stops the block: there is nothing to set the clips from. Workers on the block wait for each other at the end of the first two stages, since the next one reads their runs; the directories come back grouped by partition."""
    block_dir = shared.runs_dir / "rq2"
    steps, rates = steps_of(tier, tokens), rates_of(shared, adamw_lr, muon_lr)
    stress = partial(launch, shared, tier=tier, steps=steps, rates=rates, schedule="constant")
    calibration = [Run(partition, seeds[0]) for partition in STRESSED]
    calibrated = stress(block_dir / "calibration", calibration, wait=True)
    for run_dir in calibrated:
        if status_of(run_dir) != "finished":  # its norms are what the clips are set from
            raise RuntimeError(f"the calibration run {run_dir} {status_of(run_dir)}")
    arms: list[tuple[list[Run], list[Run], list[Run]]] = []  # per partition
    for partition, run_dir in zip(STRESSED, calibrated):
        clips = clips_of(run_dir)
        clipped = Run(partition, seeds[0], adamw_clip=clips["adamw"], muon_clip=clips["muon"])
        beside = partition == CONTROL[0] and CONTROL[1] in multipliers  # the arm it is read against
        family = [
            replace(clipped, seed=seed, lr_multiplier=m) for seed in seeds for m in multipliers
        ]
        control = [
            replace(clipped, seed=seed, lr_multiplier=CONTROL[1], balancer_multiplier=CONTROL[1])
            for seed in (seeds if beside else ())
        ]
        further = [
            replace(clipped, seed=seed, lr_multiplier=extension)
            for seed in (seeds if extension is not None else ())
        ]
        arms.append((family, control, further))
    runs = [run for family, control, _ in arms for run in family + control]
    stress(block_dir, runs, wait=True)
    grouped: list[Run] = []  # per partition: the family, the control arm, the extension
    extended: list[Run] = []
    for family, control, further in arms:
        # The family's own runs at the highest multiplier, which the control arm is not one of.
        top = [
            block_dir / run.name(tier) for run in family if run.lr_multiplier == max(multipliers)
        ]
        if any(status_of(run_dir) == "diverged" for run_dir in top):
            further = []
        grouped += family + control + further
        extended += further
    stress(block_dir, extended)
    return [block_dir / run.name(tier) for run in grouped]


def confirmation(
    shared: Shared = DEFAULTS,
    *,
    tier: str = CONFIRMATION_TIER,
    adamw_lr: float | None = None,
    muon_lr: float | None = None,
    seeds: tuple[int, ...] = SEEDS,
    recheck_tokens: int = TOKENS["recheck"],
    tokens: int = TOKENS["confirmation"],
) -> list[Path]:
    """The confirmation tier: four rates per optimizer re-checked at and below the mechanism tier's optimum (the rates given, or the tuning block's), since the best rate falls as a run gets longer, then partitions (a), (b), (c) at every seed at the rates that won times ``HORIZON``, since those runs are three times as long as the re-check's."""
    block_dir = shared.runs_dir / "confirmation"
    center = rates_of(shared, adamw_lr, muon_lr)
    grids = {
        optimizer: [factor * center[optimizer] for factor in RECHECK]
        for optimizer in ("adamw", "muon")
    }
    rates = tune(
        shared, block_dir / "recheck", grids, tier=tier, steps=steps_of(tier, recheck_tokens)
    )
    runs = [Run(partition, seed) for seed in seeds for partition in CONFIRMED]
    longer = {optimizer: HORIZON * rate for optimizer, rate in rates.items()}
    return launch(shared, block_dir, runs, tier=tier, steps=steps_of(tier, tokens), rates=longer)


def balancer_runs(seeds: tuple[int, ...]) -> list[Run]:
    """The balancer block's runs, seed by seed: (a) and (b) with the bias at its rate, then both with it ``FAST_BALANCER`` times faster."""
    speeds = (1.0, FAST_BALANCER)
    return [
        Run(partition, seed, balancer_multiplier=speed)
        for seed in seeds
        for speed in speeds
        for partition in GATE1
    ]


def balancer(
    shared: Shared = DEFAULTS,
    *,
    tier: str = TIER,
    adamw_lr: float | None = None,
    muon_lr: float | None = None,
    seeds: tuple[int, ...] = BALANCER_SEEDS,
    tokens: int = TOKENS["balancer"],
) -> list[Path]:
    """The balancer block, into ``runs/balancer``: ``balancer_runs`` at the tuned rates (the rates given, or the tuning block's), on the RQ3 block's tier and schedule."""
    rates = rates_of(shared, adamw_lr, muon_lr)
    block_dir, steps = shared.runs_dir / "balancer", steps_of(tier, tokens)
    return launch(shared, block_dir, balancer_runs(seeds), tier=tier, steps=steps, rates=rates)


def balance_loss_runs(seeds: tuple[int, ...]) -> list[Run]:
    """The balance-loss block's runs, seed by seed: (a), (b) and (c) with the sequence-wise balance loss beside the bias at ``RECIPE_LOSS``."""
    return [
        Run(partition, seed, balance_loss=RECIPE_LOSS) for seed in seeds for partition in CONFIRMED
    ]


def balance_loss(
    shared: Shared = DEFAULTS,
    *,
    tier: str = TIER,
    adamw_lr: float | None = None,
    muon_lr: float | None = None,
    seeds: tuple[int, ...] = RQ3_SEEDS,
    tokens: int = TOKENS["balance_loss"],
) -> list[Path]:
    """The balance-loss block, into ``runs/balance_loss``: ``balance_loss_runs`` at the tuned rates (the rates given, or the tuning block's), on the RQ3 block's tier, schedule and seeds. The same arms without the loss are RQ3's runs of those seeds, which it does not train again: a run and its RQ3 run share the initialization, the data order and every setting but the loss, and its claims are read with them linked into its directory."""
    rates = rates_of(shared, adamw_lr, muon_lr)
    block_dir, steps = shared.runs_dir / "balance_loss", steps_of(tier, tokens)
    runs = balance_loss_runs(seeds)
    return launch(shared, block_dir, runs, tier=tier, steps=steps, rates=rates)


def evaluate(
    shared: Shared = DEFAULTS,
    *,
    block: str = "rq3",
    panel_dir: Path = PANEL_DIR,
    batch_size: int | None = None,
) -> list[Path]:
    """The specialization panel and the annex over every finished run of the block that lacks them, which leaves out the runs that diverged and a run whose checkpoint is behind its log, killed while it wrote its last one, which a relaunch of its block finishes; the panel must be built first (``tfg.panel build``). A run another worker holds, training it or evaluating it, is left to that worker. ``batch_size`` is the sequences per forward pass, by default the tier's training batch: the annex runs a backward pass too, without recomputation, so a tier that trains with ``--recompute`` above zero needs a smaller one here."""
    if not all((panel_dir / f"{domain}.npy").exists() for domain in DOMAINS):
        raise FileNotFoundError(f"no panel under {panel_dir}: build it with tfg.panel first")
    done: list[Path] = []

    def evaluate_run(run_dir: Path) -> None:
        if status_of(run_dir) != "finished":
            print(f"{run_dir.name}: {status_of(run_dir)}, skipped", flush=True)
            return
        missing = [file for file in (PANEL_FILE, ANNEX_FILE) if not (run_dir / file).exists()]
        steps = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))["steps"]
        if missing and checkpoint.saved_step(run_dir) not in (None, steps):  # none: raised below
            print(f"{run_dir.name}: checkpoint behind its log, relaunch its block", flush=True)
            return
        if not missing:
            print(f"{run_dir.name}: evaluated before", flush=True)
        if PANEL_FILE in missing:
            evaluate_panel(run_dir, panel_dir=panel_dir, batch_size=batch_size)
        if ANNEX_FILE in missing:
            evaluate_annex(run_dir, panel_dir=panel_dir, batch_size=batch_size)
        done.append(run_dir)

    workers.each(run_dirs(shared.runs_dir / block), evaluate_run, wait=False)
    return done


BLOCKS: dict[str, Callable[..., object]] = {
    "tuning": tuning,
    "gate1": gate1,
    "rq2": rq2,
    "rq3": rq3,
    "confirmation": confirmation,
    "balancer": balancer,
    "balance_loss": balance_loss,
    "evaluate": evaluate,
}


def main(args: list[str] | None = None) -> object:
    """The command line: one subcommand per block, ``--help`` lists their options."""
    return tyro.extras.subcommand_cli_from_dict(BLOCKS, args=args)


if __name__ == "__main__":
    main()
