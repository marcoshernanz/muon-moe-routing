"""Gate 1's integrated bias span and load spread per mixture, from the runs' routing records: each reading after the warmup, averaged over the logged steps. Run from `code/` once the runs are home."""

import json
from pathlib import Path

import numpy as np

for arm in ("adamw", "muon"):
    for seed in range(3):
        run = Path(f"runs/rq1/mechanism-{arm}-s{seed}")
        warmup = json.loads((run / "settings.json").read_text(encoding="utf-8"))["warmup_steps"]
        spans, spreads = [], []
        for line in (run / "routing.jsonl").open(encoding="utf-8"):
            reading = json.loads(line)
            if reading["step"] <= warmup:
                continue
            spans.append([m["bias_span"] for m in reading["mixtures"]])
            spreads.append([np.std(m["load"]) / np.mean(m["load"]) for m in reading["mixtures"]])
        span, spread = np.mean(spans, axis=0), np.mean(spreads, axis=0)
        rest = span[1:].mean()
        print(
            f"{arm} s{seed}  span by mixture {' '.join(f'{x:.3f}' for x in span)}  mean {span.mean():.4f}  without mixture 0 {rest:.4f}"
            f"  |  load cv by mixture {' '.join(f'{x:.3f}' for x in spread)}"
        )
