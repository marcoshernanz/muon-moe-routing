"""The third chain's numbers for the evidence log, from the analyses' tables and the host's logs.

The re-check's sweeps and the rates that won; each block's claims, and every contrast of the claims Amendment 3 registered (O, O-router and U on the confirmation block, S-fast, L and O-fast on the balancer block); per run, the registered endpoints; per arm, the step time; the three readings of (a), (b) and (c) side by side (the confirmation tier, the RQ3 block and the balancer block); (c) against (b) on the validation loss; the cards and the disk. Run from code/ after tfg.analysis (both blocks, and the confirmation block once more with --reference muon --out-dir runs/confirmation/analysis-muon): uv run python ai-experiments/chain3_2026_09/summary.py > ai-experiments/chain3_2026_09/outputs/summary.txt
"""

import csv
import glob
import json
import re
from statistics import mean, median

import numpy as np

BLOCKS = {  # block: (arms, seeds, run name prefix)
    "confirmation": (["adamw", "muon", "muon_except_router"], (0, 1, 2), "confirmation"),
    "balancer": (["adamw", "muon", "adamw-balancer4x", "muon-balancer4x"], (6, 7, 8), "mechanism"),
}
NEW = {"confirmation": ("O", "O-router", "U"), "balancer": ("S-fast", "L", "O-fast")}
ENDPOINTS = [  # name in endpoints.csv: short label
    ("val_loss", "loss"),
    ("bias_span_first", "S"),
    ("router_input_similarity_first", "G"),
    ("bias_span", "R"),
    ("load_churn_warmup", "M"),
    ("specialization", "C4"),
    ("update_load_correlation", "C6"),
    ("max_load_ratio_first_early", "O"),
    ("router_update_common_share_first", "U"),
    ("bias_span_first_early", "S early"),
    ("bias_span_first_final", "S final"),
    ("collapsed_experts_final", "collapsed"),
    ("spikes", "spikes"),
]
TOKENS_PER_STEP = 65536  # both tiers
LOGS = "ai-experiments/chain3_2026_09/outputs/host-logs"


def endpoints(block: str) -> dict[tuple[str, int], dict[str, str]]:
    return {(r["arm"], int(r["seed"])): r for r in csv.DictReader(open(f"runs/{block}/analysis/endpoints.csv"))}


def number(row: dict[str, str], name: str) -> float:
    return float(row[name]) if row.get(name, "") != "" else float("nan")


print("The re-check at 1B tokens, final validation loss per rate")
for optimizer in ("adamw", "muon"):
    sweep = json.load(open(f"runs/confirmation/recheck/{optimizer}/sweep.json"))
    losses = ", ".join(f"{rate} {loss:.4f}" for rate, loss in sweep["val_loss"].items())
    print(f"{optimizer}: {losses}; best {sweep['best']}, extension {sweep['extension']}, edge {sweep['edge']}")
print(f"rates.json {json.load(open('runs/confirmation/recheck/rates.json'))}")

for block, (arms, seeds, prefix) in BLOCKS.items():
    print(f"\nThe {block} block: claims")
    for row in csv.DictReader(open(f"runs/{block}/analysis/claims.csv")):
        print(f"  {row['claim']:9} {row['endpoint']:34} contrasts {row['contrasts']:>2}  tokens {row['tokens']:10} loss {row['loss']:10} {row['verdict']}{'' if row['confirmatory'] == 'True' else ' (exploratory)'}")
    print(f"The {block} block: the contrasts of {', '.join(NEW[block])} (lower minus higher, per view)")
    for row in csv.DictReader(open(f"runs/{block}/analysis/contrasts.csv")):
        if row["claim"] in NEW[block]:
            print(f"  {row['claim']:8} {row['lower']:18} - {row['higher']:18} {row['view']:6} {float(row['difference']):+.4f} [{float(row['low']):+.4f}, {float(row['high']):+.4f}] effect {float(row['effect']):+.2f} signs {row['signs']} {row['verdict']}")
    rows = endpoints(block)
    print(f"The {block} block: per run, " + ", ".join(label for _, label in ENDPOINTS))
    for arm in arms:
        for seed in seeds:
            print(f"  {arm:20} {seed} " + " ".join(f"{number(rows[(arm, seed)], name):8.4f}" for name, _ in ENDPOINTS))
    print(f"The {block} block: step time per arm (ms, the median of each run's logged intervals after its first 500 steps) and hours a run")
    for arm in arms:
        times = []
        for seed in seeds:
            rates = [json.loads(line).get("tokens_per_s") for line in open(f"runs/{block}/{prefix}-{arm}-s{seed}/metrics.jsonl")]
            times.append(TOKENS_PER_STEP / median([r for r in rates if r][50:]) * 1000)
        steps = json.load(open(f"runs/{block}/{prefix}-{arm}-s{seeds[0]}/settings.json"))["steps"]
        print(f"  {arm:20} {min(times):.0f} to {max(times):.0f} ms, {mean(times) * steps / 3.6e6:.2f} h a run")

print("\nThe three readings of (a), (b), (c): mean over the seeds (confirmation tier at 3B tokens, seeds 0-2; RQ3 at the mechanism tier, seeds 3-5; the balancer block's (a) and (b), seeds 6-8)")
readings = {"confirmation": (endpoints("confirmation"), (0, 1, 2)), "rq3": (endpoints("rq3"), (3, 4, 5)), "balancer": (endpoints("balancer"), (6, 7, 8))}
for name, label in ENDPOINTS[:9]:
    line = []
    for block, (rows, seeds) in readings.items():
        for arm in ("adamw", "muon", "muon_except_router"):
            if (arm, seeds[0]) in rows:
                line.append(f"{block[:4]} {arm[:6]:6} {mean(number(rows[(arm, s)], name) for s in seeds):.4f}")
    print(f"  {label:5} " + "  ".join(line))

print("\n(c) minus (b) on the validation loss at 3B tokens, the second pass's comparison")
for row in csv.DictReader(open("runs/confirmation/analysis-muon/comparisons.csv")):
    if row["arm"] == "muon_except_router" and row["endpoint"] == "val_loss":
        print(f"  {row['matching']:6} {float(row['difference']):+.4f} [{float(row['low']):+.4f}, {float(row['high']):+.4f}] effect {float(row['effect']):+.2f} signs {row['signs']}")

cards = list(csv.reader(open(f"{LOGS}/gpus.csv")))
busy = [r for r in cards if float(r[4].strip(" %")) >= 90]
watts = np.array([float(r[2].strip(" W")) for r in busy])
print(f"\nCards: {len(cards)} readings, {len(busy)} at 90 percent or more; there {np.median(watts):.0f} W median, {np.percentile(watts, 5):.0f} W at the 5th percentile, {median(float(r[7].strip(' MHz')) for r in busy):.0f} MHz median; at most {max(float(r[6]) for r in cards):.0f} C; limits {sorted({r[3].strip() for r in cards})}")
free = [int(m.group(1)) for m in (re.search(r"disk_avail=(\d+)G", line) for line in open(f"{LOGS}/host.txt")) if m]
print(f"Free disk: {free[0]} GB at the start, {min(free)} GB at the least")
for block in BLOCKS:
    annexes = glob.glob(f"runs/{block}/*/annex.json")
    print(f"{block}: run directories {len(glob.glob(f'runs/{block}/{BLOCKS[block][2]}-*'))}, panels {len(glob.glob(f'runs/{block}/*/specialization.json'))}, annexes {len(annexes)} at batch sizes {sorted({json.load(open(f))['batch_size'] for f in annexes})}")
print(f"Re-check run directories: {len(glob.glob('runs/confirmation/recheck/*/lr*'))}")
print(f"Phases: {open(f'{LOGS}/phases.txt').read().strip()}")
print(f"Diverged: {open(f'{LOGS}/diverged.count').read().strip()}")
