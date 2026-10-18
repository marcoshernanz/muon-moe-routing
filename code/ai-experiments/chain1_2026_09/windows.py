"""The first mixture's bias span by window of training in the tuning's AdamW run at 2e-3 and Muon's at 2e-3 and 4e-3 (0.5B tokens) and in Gate 1's six runs (1B tokens), to see how much of Muon's first-mixture span its rate carries and how the span changes from 0.5B tokens to 1B; the plan's power paragraph reads the margin of S-rate from it.

Run from ``code/`` with ``PYTHONPATH=. uv run python ai-experiments/chain1_2026_09/windows.py > ai-experiments/chain1_2026_09/outputs/windows.txt``. Reads ``runs/tuning`` and ``runs/rq1`` (from the Hub dataset ``marcoshernanz/tfg-runs``).
"""

from itertools import pairwise
from pathlib import Path

from tfg.analysis import first_mixture, routing_frame

RUNS = {
    "tuning adamw 2e-3": "runs/tuning/adamw/lr0.002",
    "tuning muon 2e-3": "runs/tuning/muon/lr0.002",
    "tuning muon 4e-3": "runs/tuning/muon/lr0.004",
    **{
        f"gate1 {arm} s{seed}": f"runs/rq1/mechanism-{arm}-s{seed}"
        for arm in ("adamw", "muon")
        for seed in range(3)
    },
}
EDGES = (
    500,
    1000,
    2000,
    4000,
    7629,
    15259,
)  # the warmup's end, then windows to the tuning's end and Gate 1's


def main() -> None:
    windows = [f"{low}-{high}" for low, high in pairwise(EDGES)]
    print(f"{'run':20s}", " ".join(f"{w:>10s}" for w in windows), f"{'to 7629':>8s} {'all':>6s}")
    for name, run in RUNS.items():
        span = first_mixture(routing_frame(Path(run)))["bias_span"]
        means = [
            span[(span.index > low) & (span.index <= high)].mean() for low, high in pairwise(EDGES)
        ]
        half = span[(span.index > EDGES[0]) & (span.index <= 7629)].mean()
        print(
            f"{name:20s}",
            " ".join(f"{m:10.3f}" for m in means),
            f"{half:8.3f} {span[span.index > EDGES[0]].mean():6.3f}",
        )


if __name__ == "__main__":
    main()
