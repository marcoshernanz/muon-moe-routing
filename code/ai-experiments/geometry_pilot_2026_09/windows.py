"""The pilot's first-mixture readings by window of training, which the second registration's parts claims and their margins rest on. Four tables: per arm and window of 500 steps up to 3,000 (where (d), (h) and (i) were stopped), the bias span, the cosine between the router's inputs, the mean score span and the raw max-load ratio. Then the cosine per mixture over steps 2,001 to 3,000, to see whether a part on Muon moves the shared direction to another mixture. Then, for (d) beside (b) and (a), the first mixture's lowest bias against the rest and the span without it, to see whether (d)'s span is one expert the router favours. Last, the stress grid's all-Muon and Muon-except-the-router runs at 1x (constant rate, three seeds), first mixture by window, to see whether the order of (b) and (c) late in the pilot holds across seeds.

Run from ``code/`` with ``PYTHONPATH=. uv run python ai-experiments/geometry_pilot_2026_09/windows.py > ai-experiments/geometry_pilot_2026_09/outputs/windows.txt``. Reads ``runs/pilot`` and ``runs/rq2``.
"""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from tfg.analysis import first_mixture, routing_frame

PILOT = Path("runs/pilot")
ARMS = {
    "(a)": "mechanism-adamw-s0",
    "(g)": "mechanism-muon_router-s0",
    "(h)": "mechanism-muon_dense-s0",
    "(e)": "mechanism-muon_attention-s0",
    "(i)": "mechanism-muon_attention_dense-s0",
    "(d)": "mechanism-muon_experts-s0",
    "(c)": "mechanism-muon_except_router-s0",
    "(b)": "mechanism-muon-s0",
}
EDGES = (0, 500, 1000, 1500, 2000, 2500, 3000)
LATE = (2000, 3000)  # the last window every arm has
STRESS = {
    "(b)": "runs/rq2/mechanism-muon-s{seed}",
    "(c)": "runs/rq2/mechanism-muon_except_router-s{seed}",
}
STRESS_EDGES = (500, 1000, 2000, 4000, 7629)


def window(series, low: int, high: int) -> float:
    return float(series[(series.index > low) & (series.index <= high)].mean())


def tables() -> None:
    names = [f"{low}-{high}" for low, high in pairwise(EDGES)] + ["501-3000"]
    frames = {arm: first_mixture(routing_frame(PILOT / run)) for arm, run in ARMS.items()}
    for reading in (
        "bias_span",
        "router_input_similarity",
        "mean_score_span",
        "raw_max_load_ratio",
    ):
        print(reading)
        print("     " + " ".join(f"{n:>9s}" for n in names))
        for arm, first in frames.items():
            values = [window(first[reading], low, high) for low, high in pairwise(EDGES)]
            values.append(window(first[reading], EDGES[1], EDGES[-1]))
            print(f"{arm:4s} " + " ".join(f"{v:9.3f}" for v in values))
        print()


def per_mixture() -> None:
    print(f"router_input_similarity per mixture, steps {LATE[0] + 1} to {LATE[1]}")
    for arm, run in ARMS.items():
        readings = routing_frame(PILOT / run)
        late = readings[(readings["step"] > LATE[0]) & (readings["step"] <= LATE[1])]
        values = late.groupby("mixture")["router_input_similarity"].mean()
        print(f"{arm:4s} " + " ".join(f"{v:.3f}" for v in values))
    print()


def dominant_expert() -> None:
    """The first mixture's biases over steps 2,001 to 3,000 in (d), (b) and (a). The balancer lowers the bias of an expert the router over-uses, so an expert the router favours on every token shows as one bias far below the rest; the span without it says how much of the span is that one expert."""
    for arm in ("(d)", "(b)", "(a)"):
        lines = [
            json.loads(line)
            for line in (PILOT / ARMS[arm] / "routing.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        late = [line for line in lines if LATE[0] < line["step"] <= LATE[1]]
        biases = np.array([line["mixtures"][0]["bias"] for line in late])
        low = int(biases.mean(axis=0).argmin())
        rest = np.delete(biases, low, axis=1)
        span = float((biases.max(axis=1) - biases.min(axis=1)).mean())
        without = float((rest.max(axis=1) - rest.min(axis=1)).mean())
        print(
            f"{arm} first mixture, steps {LATE[0] + 1} to {LATE[1]}: lowest bias expert {low} at {biases[:, low].mean():+.3f},"
            f" the others' mean {rest.mean():+.3f}; span {span:.3f}, without it {without:.3f}"
        )
    print()


def stress() -> None:
    names = [f"{low}-{high}" for low, high in pairwise(STRESS_EDGES)]
    print("stress grid at 1x (constant rate), first mixture, bias span by window")
    print("        " + " ".join(f"{n:>10s}" for n in names))
    for seed in range(3):
        for arm, pattern in STRESS.items():
            span = first_mixture(routing_frame(Path(pattern.format(seed=seed))))["bias_span"]
            values = [window(span, low, high) for low, high in pairwise(STRESS_EDGES)]
            print(f"{arm} s{seed}  " + " ".join(f"{v:10.3f}" for v in values))


if __name__ == "__main__":
    tables()
    per_mixture()
    dominant_expert()
    stress()
