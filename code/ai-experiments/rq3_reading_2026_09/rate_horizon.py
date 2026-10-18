"""Where each optimizer's best rate should sit at 1B and 3B tokens, from the tuning sweep at 0.5B and the published fall of the best rate with the horizon.

Per optimizer, a parabola in log2 of the rate through the sweep's best rate and its two neighbours gives the best rate at 0.5B tokens; moving every loss by 0.003 either way (about the seed spread of Muon's final loss at 1B tokens) gives its range. Bjorck et al. (arXiv 2409.19913) find the best rate scales as the tokens to the power -0.32, and HyperP (arXiv 2603.28743) finds the same exponent for a Muon variant, so the rate at 1B is the one at 0.5B times 2^-0.32 and at 3B times 6^-0.32. Beside it, what the second chain measured at 1B: Muon at 2e-3 against 4e-3, per seed. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/rate_horizon.py > ai-experiments/rq3_reading_2026_09/outputs/rate_horizon.txt
"""

import itertools
import json

import numpy as np
import pandas as pd

EXPONENT = 0.32
NOISE = 0.003


def vertex(rates: list[float], losses: list[float]) -> float:
    x = np.log2(rates)
    a, b, _ = np.polyfit(x, losses, 2)
    return float(2 ** (-b / (2 * a)))


for optimizer in ["adamw", "muon", "momentum"]:
    sweep = json.load(open(f"runs/tuning/{optimizer}/sweep.json"))
    rates = sorted(float(r) for r in sweep["val_loss"])
    losses = [sweep["val_loss"][str(r) if str(r) in sweep["val_loss"] else f"{r:g}"] for r in rates]
    best = int(np.argmin(losses))
    three = slice(max(best - 1, 0), max(best - 1, 0) + 3)
    r3, l3 = rates[three], losses[three]
    centre = vertex(r3, l3)
    spread = [vertex(r3, [l + d for l, d in zip(l3, ds)]) for ds in itertools.product((-NOISE, NOISE), repeat=3)]
    print(f"{optimizer}: sweep at 0.5B " + ", ".join(f"{r:g} {l:.4f}" for r, l in zip(rates, losses)))
    for tokens, label in [(1, "0.5B"), (2, "1B"), (6, "3B")]:
        factor = tokens ** -EXPONENT
        print(f"  best rate at {label:4}: {centre * factor:.2e} (range {min(spread) * factor:.2e} to {max(spread) * factor:.2e}), factor {factor:.2f}")

table = pd.read_csv("runs/rq3/analysis/endpoints.csv")
half = table[table["arm"] == "muon-muonlr0.5x"].set_index("seed")["val_loss"]
tuned = table[table["arm"] == "muon"].set_index("seed")["val_loss"]
adamw = table[table["arm"] == "adamw"].set_index("seed")["val_loss"]
print("\nMeasured at 1B tokens (the second chain), final validation loss per seed 3, 4, 5:")
print("  Muon 2e-3 minus Muon 4e-3: " + " ".join(f"{d:+.4f}" for d in (half - tuned)))
print("  AdamW 2e-3 minus Muon 2e-3: " + " ".join(f"{d:+.4f}" for d in (adamw - half)))
print("  AdamW 2e-3 minus Muon 4e-3: " + " ".join(f"{d:+.4f}" for d in (adamw - tuned)))
