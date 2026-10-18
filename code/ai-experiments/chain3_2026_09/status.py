"""Where the third chain stands, in one screen: in the confirmation block (its re-check included) and the balancer block, or the directories given, the runs by status as ``tfg.analysis`` counts them, then every run under way with its step of its steps, its last training loss, its milliseconds per step over the last logged interval, and its last validation loss, and the re-check's winners once they are written.

Run from ``code/``: ``PYTHONPATH=. uv run python ai-experiments/chain3_2026_09/status.py [runs/confirmation runs/balancer]``.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from tfg.analysis import status_of
from tfg.tier import TIERS

counts: dict[str, Counter] = {}
under_way = []
roots = [Path(root) for root in sys.argv[1:]] or [Path("runs/confirmation"), Path("runs/balancer")]
for settings in sorted(path for root in roots for path in root.rglob("settings.json")):
    run_dir = settings.parent
    block = str(run_dir.parent.relative_to("runs"))
    status = status_of(run_dir)
    counts.setdefault(block, Counter())[status] += 1
    if status == "unfinished":
        lines = (
            (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if (run_dir / "metrics.jsonl").exists()
            else []
        )
        if lines:
            made = json.loads(settings.read_text(encoding="utf-8"))
            last = json.loads(lines[-1])
            val = next(
                (
                    json.loads(line)["val_loss"]
                    for line in reversed(lines)
                    if "val_loss" in json.loads(line)
                ),
                None,
            )
            ms = 1000 * TIERS[made["tier"]].tokens_per_step / last["tokens_per_s"]
            under_way.append(
                f"  {block}/{run_dir.name}: step {last['step']}/{made['steps']}, loss {last['loss']:.3f}, {ms:.0f} ms/step, val {val if val is None else round(val, 3)}"
            )
for block, counter in counts.items():
    print(block, dict(counter))
print("\n".join(under_way))
rates = Path("runs/confirmation/recheck/rates.json")
if rates.exists():
    print("re-check rates", json.loads(rates.read_text(encoding="utf-8")))
