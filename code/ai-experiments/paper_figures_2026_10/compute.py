"""The GPU-hours of the training steps of every run, the compute that Appendix B of the arXiv paper reports.

Lists every ``metrics.jsonl`` of the public Hub dataset ``marcoshernanz/tfg-runs`` (cached in ``outputs/hub/``) and sums, for each run, the time of its training steps. ``tfg.train`` logs every tenth step the tokens trained since the last log over the wall-clock time they took, which runs from the end of the last log, evaluation and save to the end of the step logged, so each record's steps times the tokens of a step over its ``tokens_per_s`` is that time without the evaluations and the saves. The program is every block except the pilot and the two probes, which are exploratory. Run from ``code/`` with ``.venv/bin/python ai-experiments/paper_figures_2026_10/compute.py > ai-experiments/paper_figures_2026_10/outputs/compute.txt``.
"""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "outputs" / "hub"
API = "https://huggingface.co/api/datasets/marcoshernanz/tfg-runs"
HUB = "https://huggingface.co/datasets/marcoshernanz/tfg-runs/resolve/main"
TOKENS_PER_STEP = 64 * 1024  # 64 sequences of 1,024 tokens in every run, on both models
EXPLORATORY = ("pilot", "offset-probe", "scale-probe")


def listed() -> list[str]:
    with urllib.request.urlopen(API) as response:
        files = [entry["rfilename"] for entry in json.load(response)["siblings"]]
    return sorted(name for name in files if name.endswith("/metrics.jsonl"))


def hours(name: str) -> float:
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{HUB}/{name}", path)
    total, logged = 0.0, 0
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            record = json.loads(line)
            total += (record["step"] - logged) * TOKENS_PER_STEP / record["tokens_per_s"]
            logged = record["step"]
    return total / 3600


def main() -> None:
    runs: defaultdict[str, int] = defaultdict(int)
    spent: defaultdict[str, float] = defaultdict(float)
    for name in listed():
        block = name.split("/")[0]
        runs[block] += 1
        spent[block] += hours(name)
    print("== GPU-hours of the training steps, by block")
    for block in sorted(runs):
        print(f"  {block:14s} {runs[block]:4d} runs {spent[block]:9.2f} h")
    program = [block for block in runs if block not in EXPLORATORY]
    count, total = sum(runs[block] for block in program), sum(spent[block] for block in program)
    print(f"  program        {count:4d} runs {total:9.2f} h")
    print(f"  all            {sum(runs.values()):4d} runs {sum(spent.values()):9.2f} h")


if __name__ == "__main__":
    main()
