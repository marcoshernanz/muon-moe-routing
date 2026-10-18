"""What the offset probe's runs say about the step between the shared direction and the span.

Per run, the first mixture (index 0 of every [B, E] reading) over windows of steps: the bias span; the spread across experts of the batch-mean logit, which is the offset the router's scores carry (on the router's own inputs, centred where the run centres them), and of the offset an uncentred input would give, w_e . x_bar; the length of the batch-mean input over the mean length of an input (the shared direction's strength) and the spread of the rows' lean along it (how unevenly the router's rows point along it), whose product is that second offset; the busiest expert's load over the even share; and, per logged step, the correlation across experts between the bias and minus the batch-mean logit, averaged over the window (+1 when the bias exactly mirrors the offset). Then per arm, seed means: the offset's two factors against all-AdamW's, and the centred runs against their plain twins on the span, the early load and the final validation loss. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/offset_probe/analyze.py [RUNS] > ai-experiments/rq3_reading_2026_09/offset_probe/outputs/analysis.txt
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "ai-experiments/rq3_reading_2026_09/offset_probe/outputs/mechanism/runs")
WINDOWS = [(0, 300), (300, 500), (500, 800), (800, 1500), (1500, 3000)]
AFTER = (500, 3000)  # after the warmup, to the end of a probe run
EARLY = (500, 800)
ORDER = ["adamw", "muon", "muon-muonlr0.5x", "muon_experts", "muon_except_router-muonlr0.5x", "muon-balancer4x", "muon-centred", "adamw-centred"]


def spread(values: np.ndarray) -> np.ndarray:  # [T, E] -> [T]
    return values.max(1) - values.min(1)


def mirror(bias: np.ndarray, logit: np.ndarray) -> np.ndarray:  # [T, E] each -> [T]
    b = bias - bias.mean(1, keepdims=True)
    o = -(logit - logit.mean(1, keepdims=True))
    return (b * o).sum(1) / np.sqrt((b * b).sum(1) * (o * o).sum(1) + 1e-30)


def readings(run: Path) -> dict[str, np.ndarray]:
    data = np.load(run / "offsets.npz")
    first = {name: data[name][:, 0] for name in data.files if name != "steps"}  # [T, E] or [T, 2]
    experts = first["bias"].shape[1]
    length, mean_length = first["lengths"][:, 0], first["lengths"][:, 1]
    return {
        "steps": data["steps"],
        "span": spread(first["bias"]),
        "logit_spread": spread(first["mean_logit"]),
        "offset_spread": spread(first["offset"]),
        "strength": length / mean_length,
        "lean_spread": spread(first["lean"]) * mean_length,  # in the units of an input of mean length
        "max_load": first["load"].max(1) * experts,
        "mirror": mirror(first["bias"], first["mean_logit"]),
        "mirror_uncentred": mirror(first["bias"], first["offset"]),
    }


def mean_over(values: np.ndarray, steps: np.ndarray, window: tuple[int, int]) -> float:
    keep = (steps > window[0]) & (steps <= window[1])
    return float(np.nanmean(values[keep])) if keep.any() else float("nan")


runs = sorted(p for p in RUNS.glob("*") if (p / "offsets.npz").exists())
if not runs:
    sys.exit(f"no runs under {RUNS}")
table: dict[str, dict[int, dict]] = defaultdict(dict)
for run in runs:
    name = run.name.split("-", 1)[1]  # drop the tier
    arm, seed = name.rsplit("-s", 1)
    info = json.loads((run / "probe.json").read_text()) if (run / "probe.json").exists() else {}
    r = readings(run)
    table[arm][int(seed)] = {"r": r, "val_loss": info.get("val_loss"), "last": int(r["steps"][-1])}

keys = ["span", "logit_spread", "offset_spread", "strength", "lean_spread", "max_load", "mirror", "mirror_uncentred"]
print("First mixture per run and window of steps (" + ", ".join(f"{a + 1}-{b}" for a, b in WINDOWS) + "):")
print("span = bias span; logit = spread of the batch-mean logit (the offset the scores carry); offset = spread of w_e . x_bar (what an uncentred input would give);")
print("strength = |mean input| / mean |input|; lean = spread of the rows' components along the mean input, times mean |input|; load = busiest expert / even share;")
print("mirror = correlation across experts of the bias with minus the mean logit, per step; mirror_u = the same with minus w_e . x_bar\n")
for arm in [a for a in ORDER if a in table] + [a for a in table if a not in ORDER]:
    for seed, row in sorted(table[arm].items()):
        r = row["r"]
        print(f"{arm} s{seed} (to step {row['last']}, final val loss {row['val_loss'] if row['val_loss'] is None else round(row['val_loss'], 4)})")
        for key in keys:
            print(f"  {key:16}" + " ".join(f"{mean_over(r[key], r['steps'], w):9.4f}" for w in WINDOWS))

print("\nPer arm, seed means over steps 501 to 3,000 (after the warmup), and the early load over steps 501 to 800:")
print(f"{'arm':18}{'seeds':>6}{'span':>9}{'logit':>9}{'offset':>9}{'strength':>10}{'lean':>9}{'mirror':>9}{'mirror_u':>9}{'early load':>12}{'val loss':>10}")
summary = {}
for arm in [a for a in ORDER if a in table] + [a for a in table if a not in ORDER]:
    seeds = sorted(table[arm])
    after = {k: np.mean([mean_over(table[arm][s]["r"][k], table[arm][s]["r"]["steps"], AFTER) for s in seeds]) for k in keys}
    early = np.mean([mean_over(table[arm][s]["r"]["max_load"], table[arm][s]["r"]["steps"], EARLY) for s in seeds])
    losses = [table[arm][s]["val_loss"] for s in seeds if table[arm][s]["val_loss"] is not None]
    loss = np.mean(losses) if losses else float("nan")
    summary[arm] = {**after, "early_load": early, "val_loss": loss, "seeds": seeds}
    print(f"{arm:18}{len(seeds):>6}{after['span']:9.4f}{after['logit_spread']:9.4f}{after['offset_spread']:9.4f}{after['strength']:10.4f}{after['lean_spread']:9.4f}{after['mirror']:9.3f}{after['mirror_uncentred']:9.3f}{early:12.3f}{loss:10.4f}")

if "adamw" in summary:
    print("\nThe offset's two factors against all-AdamW (ratios of the seed means over steps 501 to 3,000):")
    for arm, row in summary.items():
        if arm == "adamw" or arm.endswith("centred"):
            continue
        base = summary["adamw"]
        print(f"  {arm:18} offset x{row['offset_spread'] / base['offset_spread']:.2f} = strength x{row['strength'] / base['strength']:.2f} and lean spread x{row['lean_spread'] / base['lean_spread']:.2f}; span x{row['span'] / base['span']:.2f}")

print("\nCentred against plain, per seed (span over steps 501 to 3,000; busiest expert over 501 to 800; final validation loss):")
for plain in ["muon", "adamw"]:
    centred = f"{plain}-centred"
    if plain not in table or centred not in table:
        continue
    for seed in sorted(set(table[plain]) & set(table[centred])):
        p, c = table[plain][seed], table[centred][seed]
        values = [(mean_over(x["r"]["span"], x["r"]["steps"], AFTER), mean_over(x["r"]["max_load"], x["r"]["steps"], EARLY), x["val_loss"]) for x in (p, c)]
        loss = f"{values[0][2]:.4f} -> {values[1][2]:.4f} ({values[1][2] - values[0][2]:+.4f})" if None not in (values[0][2], values[1][2]) else "n/a"
        print(f"  {plain:6} s{seed}: span {values[0][0]:.4f} -> {values[1][0]:.4f}; early load {values[0][1]:.3f} -> {values[1][1]:.3f}; val loss {loss}")
