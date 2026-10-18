"""The first mixture's realized load early in training, in every block of the program that logged it: how far the busiest expert runs over its fair share while the bias is still catching up with the router's offset.

The bias moves by a fixed 0.001 a step (loss-free balancing's sign update), so an offset that opens faster than that is carried by the load until the bias arrives. Read here: the second chain (seeds 3 to 5, 1B tokens); Gate 1 (seeds 0 to 2, 1B tokens); the stress grid (0.5B tokens, seeds 0 to 2, every arm at its tuned rate and at two to sixteen times it, and Muon at four times with the balancer's rate four times too, the control of the balancer's speed); and the tuning block (0.5B tokens, seed 0, every rate). Per run, the first mixture's span, the busiest expert's load over the fair share (`max_load_ratio`), the usage perplexity (32 is even) and the churn, each averaged over windows of steps; and the stress grid's final loss with the balancer at its rate and at four times it. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/overload.py > ai-experiments/rq3_reading_2026_09/outputs/overload.txt
"""

import glob
import json
import os

import numpy as np
from reading import ARMS, LABEL, SEEDS, first_mixture, seed_mean, window

KEYS = ["bias_span", "max_load_ratio", "usage_perplexity", "load_churn"]


def table(title: str, paths: list[str], windows: list[tuple[int, int]]) -> None:
    print(f"\n{title}\nper run and window " + " ".join(f"{a + 1}-{b}" for a, b in windows) + ": " + " | ".join(KEYS))
    for path in paths:
        at, values = first_mixture(path, KEYS)
        parts = []
        for k in range(len(KEYS)):
            parts.append(" ".join(f"{np.nanmean(values[(at > a) & (at <= b), k]):6.3f}" for a, b in windows))
        print(f"{os.path.relpath(path, 'runs'):48} " + " | ".join(parts))


LONG = [(0, 300), (300, 500), (500, 1262), (1262, 3000), (3000, 15259)]
print("The second chain (seeds 3 to 5, 1B tokens): seed means per arm, windows " + " ".join(f"{a + 1}-{b}" for a, b in LONG) + ": " + " | ".join(KEYS))
for arm in ARMS:
    print(f"{LABEL[arm]:10} " + " | ".join(" ".join(f"{seed_mean(arm, k, a, b):6.3f}" for a, b in LONG) for k in KEYS))
print("\nThe same, max_load_ratio per seed over steps 301 to 1,262")
for arm in ARMS:
    print(f"{LABEL[arm]:10} " + " ".join(f"{window(arm, s, 'max_load_ratio', 300, 1262):6.3f}" for s in SEEDS))

table("Gate 1 (seeds 0 to 2, 1B tokens)", sorted(glob.glob("runs/rq1/mechanism-*")), LONG)
SHORT = [(0, 100), (100, 300), (300, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7629)]
stress = [f"runs/rq2/mechanism-{name}-s{s}" for name in ["adamw", "muon", "muon_except_router", "adamw-lr4x", "muon-lr4x", "muon-lr4x-balancer4x", "muon_except_router-lr4x"] for s in (0, 1, 2)]
table("The stress grid (seeds 0 to 2, 0.5B tokens): the tuned rates, four times them, and Muon at four times with the balancer's rate four times too", stress, SHORT)
tuning = sorted(glob.glob("runs/tuning/*/lr*"), key=lambda p: (p.split("/")[2], float(p.split("lr")[-1])))
table("The tuning block (seed 0, 0.5B tokens), every optimizer and rate", tuning, SHORT)


def final_loss(path: str) -> float:
    return [json.loads(line) for line in open(f"{path}/metrics.jsonl") if "val_loss" in line][-1]["val_loss"]


print("\nWhat the faster balancer does to the loss: the stress grid's final validation loss, Muon at four times its rate with the balancer at its rate and at four times it, per seed 0 1 2")
for name in ["muon-lr4x", "muon-lr4x-balancer4x"]:
    print(f"{name:24} " + " ".join(f"{final_loss(f'runs/rq2/mechanism-{name}-s{s}'):.4f}" for s in (0, 1, 2)))
print("difference                " + " ".join(f"{final_loss(f'runs/rq2/mechanism-muon-lr4x-balancer4x-s{s}') - final_loss(f'runs/rq2/mechanism-muon-lr4x-s{s}'):+.4f}" for s in (0, 1, 2)))
