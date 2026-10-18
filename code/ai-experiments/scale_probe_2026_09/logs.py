"""The saturation reading of the program's logs: how far the first mixture's router logits grow, how thin the selection margins get against the bias's step, and how much the load churns, per arm and tier, which the scale probe then tests.

Per group of runs (tier, block, arm, with the rate multiplier or the balancer's where they depart from the default), the mean over its seeds of the first mixture's readings over the steps after the warmup and over the early window (the 5 percent of steps after it): the router logits' norm (the mean over tokens of the logit vector's length), the router entropy, the mean selection margin and its 10 percent quantile (the last chosen expert's score plus bias minus the first rejected one's), both over the bias's step u, the load churn between readings ten steps apart (the share of slots that changed expert), the busiest expert's load over its share with the bias and without it, the router-input cosine, the bias span, and how often the busiest expert is the same at two consecutive readings. Then, over every balanced run of the mechanism tier and of the confirmation tier, the rank correlations between those run means; at the confirmation tier the same readings per mixture, the first against the ten after it; and (c) minus (b) per seed at both tiers. Post hoc and exploratory. Run from code/: PYTHONPATH=. uv run python ai-experiments/scale_probe_2026_09/logs.py > ai-experiments/scale_probe_2026_09/outputs/logs.txt
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

U = 1e-3  # the bias's step at the default rate (tfg.model.BIAS_UPDATE_RATE)
BLOCKS = ("rq1", "rq2", "rq3", "balancer", "confirmation")
READINGS = [  # label, routing key
    ("logit norm", "router_logit_norm"),
    ("entropy", "router_entropy"),
    ("margin/u", "margin_mean"),
    ("low/u", "margin_low"),
    ("churn", "load_churn"),
    ("busiest", "max_load_ratio"),
    ("raw busiest", "raw_max_load_ratio"),
    ("cosine", "router_input_similarity"),
    ("span", "bias_span"),
]


def run_dirs() -> list[Path]:
    dirs = []
    for block in BLOCKS:
        dirs += sorted(p.parent for p in Path("runs", block).glob("*/routing.jsonl"))
    return dirs


def read(run: Path) -> dict:
    """The run's settings and its first mixture's readings, whole and early, and every mixture's over the whole run."""
    settings = json.loads((run / "settings.json").read_text())
    warmup, steps = settings["warmup_steps"], settings["steps"]
    early_end = warmup + round(0.05 * steps)
    rate = U * settings.get("balancer_multiplier", 1.0)
    rows = [json.loads(line) for line in open(run / "routing.jsonl")]
    after = [r for r in rows if r["step"] > warmup]
    early = [r for r in after if r["step"] <= early_end]

    def means(records: list[dict], mixture: int) -> dict[str, float]:
        out = {}
        for label, key in READINGS:
            values = np.array([r["mixtures"][mixture].get(key, np.nan) for r in records], dtype=float)  # the first chain logged no geometry
            if key.startswith("margin"):
                values = values / rate
            out[label] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        busiest = [int(np.argmax(r["mixtures"][mixture]["load"])) for r in records]
        out["same busiest"] = float(np.mean([a == b for a, b in zip(busiest, busiest[1:])]))
        return out

    count = len(rows[0]["mixtures"])
    return {
        "settings": settings,
        "whole": means(after, 0),
        "early": means(early, 0),
        "per_mixture": [means(after, m) for m in range(count)],
    }


def group_of(run: Path, settings: dict) -> str:
    arm = run.name.rsplit("-s", 1)[0].split("-", 1)[1]
    balanced = "" if settings.get("balanced", True) else " (no bias)"
    return f"{settings['tier'][:4]} {run.parent.name:12} {arm}{balanced}"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    runs = {run: read(run) for run in run_dirs()}
    groups: dict[str, list[dict]] = defaultdict(list)
    for run, data in runs.items():
        groups[group_of(run, data["settings"])].append(data)

    labels = [label for label, _ in READINGS] + ["same busiest"]
    for window in ("whole", "early"):
        print(f"The first mixture, {window} ({'every step after the warmup' if window == 'whole' else 'the early window, the 5 percent of steps after the warmup'}), mean over seeds; margins over the bias's step u of the run")
        print(f"  {'group':58} {'n':>2} " + " ".join(f"{label:>12}" for label in labels))
        for name, members in groups.items():
            values = [np.mean([m[window][label] for m in members]) for label in labels]
            print(f"  {name:58} {len(members):>2} " + " ".join(f"{v:12.3f}" for v in values))
        print()

    print("Rank correlations over every balanced run of a tier, the first mixture over the whole run (n runs)")
    pairs = [("logit norm", "margin/u"), ("margin/u", "churn"), ("logit norm", "churn"), ("cosine", "logit norm"), ("churn", "busiest"), ("margin/u", "busiest"), ("cosine", "span"), ("logit norm", "span")]
    for tier in ("mechanism", "confirmation"):
        members = [d for d in runs.values() if d["settings"]["tier"] == tier and d["settings"].get("balanced", True)]
        cells = []
        for a, b in pairs:
            both = [d for d in members if np.isfinite(d["whole"][a]) and np.isfinite(d["whole"][b])]
            x = np.array([d["whole"][a] for d in both])
            y = np.array([d["whole"][b] for d in both])
            cells.append(f"{a} ~ {b} {spearman(x, y):+.2f} ({len(both)})")
        print(f"  {tier} (n = {len(members)}): " + "; ".join(cells))
    print()

    print("The confirmation tier per mixture over the whole run, mean over seeds: logit norm / margin over u / churn / busiest")
    for arm in ("adamw", "muon", "muon_except_router"):
        members = [d for run, d in runs.items() if run.parent.name == "confirmation" and run.name.startswith(f"confirmation-{arm}-s")]
        count = len(members[0]["per_mixture"])
        cells = []
        for m in range(count):
            v = [np.mean([d["per_mixture"][m][label] for d in members]) for label in ("logit norm", "margin/u", "churn", "busiest")]
            cells.append(f"{v[0]:.1f}/{v[1]:.1f}/{v[2]:.3f}/{v[3]:.2f}")
        print(f"  {arm:20} " + "  ".join(cells))
    print()

    print("The router on AdamW against the router on Muon at each tier, (c) minus (b) per seed on the first mixture, in the early window and over the whole run (O-router's contrast is the busiest expert in the early window)")
    for block, seeds in (("rq3", (3, 4, 5)), ("confirmation", (0, 1, 2))):
        tier = "confirmation" if block == "confirmation" else "mechanism"
        for window in ("early", "whole"):
            cells = []
            for label in ("busiest", "churn", "margin/u", "logit norm"):
                diffs = [runs[Path("runs", block, f"{tier}-muon_except_router-s{s}")][window][label] - runs[Path("runs", block, f"{tier}-muon-s{s}")][window][label] for s in seeds]
                cells.append(f"{label} " + " ".join(f"{d:+.3f}" for d in diffs))
            print(f"  {block:12} {window:5}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
