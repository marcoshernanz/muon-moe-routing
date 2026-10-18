"""The offset probe's runs at the mechanism tier against the second chain's runs of the same seeds, over the same first 3,000 steps.

Each probe run is the second chain's run of its arm and seed (the tag's code, compiled, the first mixture block recomputed, the schedule of 1B tokens, every shard present, so the same data order) stopped after step 3,000, with an intervention or a setting the chain did not have: the first mixture's router input centred, Muon at half its rate with the router on AdamW, or the bias moving four times faster. Per run, from the thesis telemetry's routing log, the first mixture's bias span over steps 501 to 3,000, the busiest expert's load over its fair share over steps 501 to 1,262 (the plan's early window) and 301 to 500, the router-input cosine and the score offset (`mean_score_span`) over steps 501 to 3,000, the router update's common share over steps 301 to 1,262, and the validation loss at step 3,000; beside it the same readings of the chain's runs it is read against. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/offset_probe/versus_chain.py [PROBE_RUNS] > ai-experiments/rq3_reading_2026_09/offset_probe/outputs/versus_chain.txt
"""

import json
import sys
from pathlib import Path

import numpy as np

PROBE = Path(sys.argv[1] if len(sys.argv) > 1 else "ai-experiments/rq3_reading_2026_09/offset_probe/outputs/mechanism/runs")
CHAIN = Path("runs/rq3")
STOP = 3000
READINGS = [  # label, routing key, window of steps (low, high]
    ("span", "bias_span", (500, STOP)),
    ("load 301-500", "max_load_ratio", (300, 500)),
    ("load early", "max_load_ratio", (500, 1262)),
    ("cosine", "router_input_similarity", (500, STOP)),
    ("offset", "mean_score_span", (500, STOP)),
    ("common", "router_update_common_share", (300, 1262)),
]
AGAINST = {  # probe arm: the chain's arms it is read against
    "muon-centred": ["muon", "adamw"],
    "muon": ["muon"],
    "adamw": ["adamw"],
    "muon-muonlr0.5x": ["muon-muonlr0.5x"],
    "muon_except_router-muonlr0.5x": ["muon-muonlr0.5x", "muon_except_router"],
    "muon-balancer4x": ["muon", "adamw"],
}


def read(run: Path) -> dict[str, float]:
    at, rows = [], []
    with (run / "routing.jsonl").open() as log:
        for line in log:
            record = json.loads(line)
            if record["step"] > STOP:
                break
            at.append(record["step"])
            rows.append([record["mixtures"][0][key] for _, key, _ in READINGS])
    at, values = np.array(at), np.array(rows, dtype=float)
    out = {label: float(np.nanmean(values[(at > low) & (at <= high), i])) for i, (label, _, (low, high)) in enumerate(READINGS)}
    loss = [json.loads(line) for line in (run / "metrics.jsonl").open()]
    out["val loss"] = next((r["val_loss"] for r in loss if r["step"] == STOP and "val_loss" in r), float("nan"))
    out["last step"] = int(at[-1]) if len(at) else 0
    return out


def row(name: str, values: dict[str, float]) -> str:
    return f"  {name:44}" + "".join(f"{values[label]:>13.4f}" for label, _, _ in READINGS) + f"{values['val loss']:>11.4f}"


probes = sorted(p for p in PROBE.glob("mechanism-*") if (p / "probe.json").exists())
if not probes:
    sys.exit(f"no finished probe runs under {PROBE}")
print("First mixture; span, cosine and offset over steps 501-3,000, the busiest expert's load over 301-500 and 501-1,262 (8 is every token on one expert), the router update's common share over 301-1,262, validation loss at step 3,000.")
print(f"  {'run':44}" + "".join(f"{label:>13}" for label, _, _ in READINGS) + f"{'val loss':>11}")
by_arm: dict[str, dict[int, dict[str, float]]] = {}
for run in probes:
    arm, seed = run.name.split("-", 1)[1].rsplit("-s", 1)
    values = read(run)
    by_arm.setdefault(arm, {})[int(seed)] = values
    print(f"probe {arm} s{seed} (to step {values['last step']})")
    print(row("probe", values))
    for twin in AGAINST.get(arm, []):
        chain = CHAIN / f"mechanism-{twin}-s{seed}"
        if chain.exists():
            print(row(f"chain {twin}", read(chain)))

print("\nPer probe arm, seed means, and the chain's arms at the same seeds:")
print(f"  {'arm (seeds)':44}" + "".join(f"{label:>13}" for label, _, _ in READINGS) + f"{'val loss':>11}")
for arm, seeds in by_arm.items():
    keys = [label for label, _, _ in READINGS] + ["val loss"]
    print(row(f"probe {arm} ({','.join(map(str, sorted(seeds)))})", {k: float(np.mean([seeds[s][k] for s in seeds])) for k in keys}))
    for twin in AGAINST.get(arm, []):
        chain = [read(CHAIN / f"mechanism-{twin}-s{s}") for s in sorted(seeds) if (CHAIN / f"mechanism-{twin}-s{s}").exists()]
        if chain:
            print(row(f"chain {twin}", {k: float(np.mean([c[k] for c in chain])) for k in keys}))
