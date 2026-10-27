"""The one data file of the arXiv paper's figures that the memoria's figures do not already have: the first MoE layer over the first 3,000 steps of the offset probe's runs with the first router's input centered (seeds 3 to 5).

The probe's runs are in ``../rq3_reading_2026_09/offset_probe/outputs/mechanism/runs/``. The curves are binned and summarized as in ``../thesis_figures_2026_09/extract.py`` (bins of 30 steps, the mean of the seeds and their lowest and highest), so the file has the columns of ``thesis/tex/figures/data/rq3-early-*.csv`` and the figure can draw it beside them. As a check, the probe's plain runs at seed 3 (Muon and AdamW), which reproduce the second chain's runs bit for bit, are binned the same way and must fall inside the band of their arm's seeds in those files. Run from ``code/`` with ``.venv/bin/python ai-experiments/paper_figures_2026_10/extract.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))

from tfg import analysis

PROBE = CODE / "ai-experiments" / "rq3_reading_2026_09" / "offset_probe" / "outputs" / "mechanism" / "runs"
THESIS = CODE.parent / "thesis" / "tex" / "figures" / "data"
DATA = CODE.parent / "paper" / "figures" / "data"
NAMES = ("router_input_similarity", "mean_score_span", "max_load_ratio", "bias_span")
WIDTH = 30
UNTIL = 3000


def binned(series: pd.Series, width: int) -> pd.Series:
    """Means over consecutive bins of ``width`` steps, indexed by the bin's center, as in the memoria's extract.py."""
    bins = ((series.index - 1) // width) * width + width / 2
    return series.groupby(bins).mean()


def first_layer(run: Path, name: str) -> pd.Series:
    values = analysis.first_mixture(analysis.routing_frame(run))[name]
    return binned(values[values.index <= UNTIL], WIDTH)


def main() -> None:
    runs = [PROBE / f"mechanism-muon-centred-s{seed}" for seed in (3, 4, 5)]
    columns: dict[str, object] = {}
    for name in NAMES:
        stacked = pd.concat([first_layer(run, name) for run in runs], axis=1)
        columns["step"] = stacked.index.to_numpy()
        columns[name] = stacked.mean(axis=1).to_numpy()
        columns[f"{name}_low"] = stacked.min(axis=1).to_numpy()
        columns[f"{name}_high"] = stacked.max(axis=1).to_numpy()
    DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(DATA / "rq3-early-muon-centered.csv", index=False, float_format="%.5f")

    for arm in ("muon", "adamw"):
        band = pd.read_csv(THESIS / f"rq3-early-{arm}.csv").set_index("step")
        for name in NAMES:
            mine = first_layer(PROBE / f"mechanism-{arm}-s3", name)
            low = band.loc[mine.index, f"{name}_low"]
            high = band.loc[mine.index, f"{name}_high"]
            outside = int(((mine < low - 1e-5) | (mine > high + 1e-5)).sum())
            print(f"{arm} s3 {name}: {outside} of {len(mine)} bins outside the band of seeds 3 to 5")

    frame = pd.DataFrame(columns).set_index("step")
    after = frame[(frame.index > 500) & (frame.index <= 1262)]
    print(f"centered, bins in steps 501 to 1,262: overload {after['max_load_ratio'].mean():.3f}, span {after['bias_span'].mean():.3f}")


if __name__ == "__main__":
    main()
