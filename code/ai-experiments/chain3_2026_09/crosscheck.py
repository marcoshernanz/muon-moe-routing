"""An independent check of the third chain's runs and of the analysis's readings the new claims rest on, without tfg.

Every run's settings against the plan (the re-check's grids at an eighth, a quarter, a half and all of the tuned rates with one more rate past a winning edge, the confirmation runs at 0.7 times the re-check's winners, the balancer block at the tuned rates with the bias at 1 and 4 times its rate), one data order per block (train_windows), a routing reading at every tenth step and at the last and no step logged twice, no loss that is not a number, and, from routing.jsonl, the first mixture's integrated bias span (the largest bias minus the smallest, the mean over the logged steps after the warmup), its router-input cosine and its router update's common share, and its busiest expert's load over its fair share over the early window (the logged steps after the warmup up to 5 percent of the run's steps further), each against the analysis's endpoints.csv. Run from code/: uv run python ai-experiments/chain3_2026_09/crosscheck.py > ai-experiments/chain3_2026_09/outputs/crosscheck.txt
"""

import csv
import glob
import json
import os
from statistics import mean

TUNED = {"adamw": 0.002, "muon": 0.004}  # runs/tuning/rates.json
RECHECK = (0.125, 0.25, 0.5, 1.0)
HORIZON = 0.7
COMMON = dict(
    warmup_steps=500,
    adamw_clip=1.0,
    muon_clip=1.0,
    schedule="cosine",
    compiled=True,
    lr_multiplier=1.0,
    router_lr_multiplier=1.0,
    muon_lr_multiplier=1.0,
    batch_size=64,
    log_every=10,
    eval_every=250,
    adamw_beta2=0.95,
    balanced=True,
)
READINGS = {  # endpoint: (reading, window)
    "bias_span_first": ("bias_span", "run"),
    "router_input_similarity_first": ("router_input_similarity", "run"),
    "max_load_ratio_first_early": ("max_load_ratio", "early"),
    "router_update_common_share_first": ("router_update_common_share", "run"),
}
problems: list[tuple] = []
worst = dict.fromkeys(READINGS, 0.0)


def check(run: str, want: dict, endpoints: dict | None, key: tuple | None) -> int:
    """Check one run against ``want`` and, given the analysis's ``endpoints``, its readings against the row at ``key``; print one line and return its train_windows."""
    settings = json.load(open(f"{run}/settings.json"))
    name = os.path.basename(run)
    for k, v in dict(COMMON, **want).items():
        if abs(settings[k] - v) > 1e-12 * abs(v) if isinstance(v, float) else settings[k] != v:
            problems.append((name, k, settings[k], v))
    steps, first = [], []
    for line in open(f"{run}/routing.jsonl"):
        record = json.loads(line)
        steps.append(record["step"])
        first.append(record["mixtures"][0])
    total, warmup = settings["steps"], settings["warmup_steps"]
    if steps != list(range(10, total - total % 10 + (1 if total % 10 else -9), 10)) + [total]:
        problems.append((name, "routing steps", len(steps), len(set(steps)), steps[-1]))
    metrics = [json.loads(line) for line in open(f"{run}/metrics.jsonl")]
    if len({m["step"] for m in metrics}) != len(metrics):
        problems.append((name, "a metrics step logged twice"))
    if any(x != x or abs(x) == float("inf") for m in metrics for x in m.get("losses", [])):
        problems.append((name, "a loss that is not a number"))
    evaluated = [m for m in metrics if "val_loss" in m]
    ours = {}
    for endpoint, (reading, scope) in READINGS.items():
        high = warmup + 0.05 * total if scope == "early" else float("inf")
        values = [
            (max(r["bias"]) - min(r["bias"])) if reading == "bias_span" else r[reading]
            for s, r in zip(steps, first)
            if warmup < s <= high and (reading == "bias_span" or r.get(reading) is not None)
        ]
        ours[endpoint] = mean(values)
    line = f"{name:40} {len(steps)} readings to step {steps[-1]}, val {evaluated[-1]['val_loss']:.4f} at {evaluated[-1]['step']}"
    if endpoints is not None:
        row = endpoints[key]
        for endpoint, value in ours.items():
            if endpoint in row and row[endpoint] != "":
                worst[endpoint] = max(worst[endpoint], abs(value - float(row[endpoint])))
        line += ", " + ", ".join(f"{e.replace('_first', '')} {v:.4f}" for e, v in ours.items())
    print(line)
    return settings["train_windows"]


def main() -> None:
    print("The re-check")
    recheck_rates = json.load(open("runs/confirmation/recheck/rates.json"))
    print("winners", recheck_rates)
    windows = set()
    for optimizer in ("adamw", "muon"):
        sweep = json.load(open(f"runs/confirmation/recheck/{optimizer}/sweep.json"))
        grid = [factor * TUNED[optimizer] for factor in RECHECK]
        print(optimizer, json.dumps(sweep))
        for run in sorted(glob.glob(f"runs/confirmation/recheck/{optimizer}/lr*")):
            rate = float(os.path.basename(run)[2:])
            if not any(abs(rate - g) < 1e-12 for g in grid) and not any(
                abs(rate - x) < 1e-12 for x in (grid[0] / 2, grid[-1] * 2)
            ):
                problems.append((run, "a rate off the grid and its edges"))
            adamw = recheck_rates["adamw"] if optimizer == "muon" else rate
            want = dict(
                tier="confirmation",
                steps=15259,
                recompute=9,
                partition=optimizer,
                seed=0,
                balancer_multiplier=1.0,
                adamw_lr=adamw,
            )
            if optimizer == "muon":
                want["muon_lr"] = rate
            windows.add(check(run, want, None, None))
    print("\nThe confirmation runs")
    endpoints = {
        (r["arm"], r["seed"]): r
        for r in csv.DictReader(open("runs/confirmation/analysis/endpoints.csv"))
    }
    seen = set()
    for run in sorted(glob.glob("runs/confirmation/confirmation-*")):
        name = os.path.basename(run)
        arm, seed = (
            name[len("confirmation-") : name.rindex("-s")],
            int(name[name.rindex("-s") + 2 :]),
        )
        seen.add((arm, seed))
        want = dict(
            tier="confirmation",
            steps=45776,
            recompute=9,
            partition=arm,
            seed=seed,
            balancer_multiplier=1.0,
            adamw_lr=HORIZON * recheck_rates["adamw"],
            muon_lr=HORIZON * recheck_rates["muon"],
        )
        windows.add(check(run, want, endpoints, (arm, str(seed))))
    missing = sorted(
        {(a, s) for a in ("adamw", "muon", "muon_except_router") for s in (0, 1, 2)} - seen
    )
    print(f"runs {len(seen)}, missing {missing}; train_windows {sorted(windows)}")
    print("\nThe balancer block")
    endpoints = {
        (r["arm"], r["seed"]): r
        for r in csv.DictReader(open("runs/balancer/analysis/endpoints.csv"))
    }
    seen, windows = set(), set()
    for run in sorted(glob.glob("runs/balancer/mechanism-*")):
        name = os.path.basename(run)
        arm, seed = name[len("mechanism-") : name.rindex("-s")], int(name[name.rindex("-s") + 2 :])
        seen.add((arm, seed))
        want = dict(
            tier="mechanism",
            steps=15259,
            recompute=1,
            partition=arm.split("-")[0],
            seed=seed,
            balancer_multiplier=4.0 if arm.endswith("balancer4x") else 1.0,
            adamw_lr=TUNED["adamw"],
            muon_lr=TUNED["muon"],
        )
        windows.add(check(run, want, endpoints, (arm, str(seed))))
    missing = sorted(
        {
            (a, s)
            for a in ("adamw", "muon", "adamw-balancer4x", "muon-balancer4x")
            for s in (6, 7, 8)
        }
        - seen
    )
    print(f"runs {len(seen)}, missing {missing}; train_windows {sorted(windows)}")
    print(
        "\nlargest difference from the analyses: "
        + ", ".join(f"{e} {v:.1e}" for e, v in worst.items())
    )
    print(f"problems: {problems or 'none'}")


if __name__ == "__main__":
    main()
