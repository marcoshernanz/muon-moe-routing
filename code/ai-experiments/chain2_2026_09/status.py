"""Where the second chain stands, in one screen: in the RQ3 block (or the directory given), its runs by status as ``tfg.analysis`` counts them, then every run under way with its step of its steps, its last training loss, its milliseconds per step over the last logged interval, and its last validation loss.

Run from ``code/``: ``PYTHONPATH=. uv run python ai-experiments/chain2_2026_09/status.py [runs/rq3]``.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from tfg.analysis import status_of
from tfg.tier import TIERS

counts: dict[str, Counter] = {}
under_way = []
for settings in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/rq3").rglob("settings.json")):
    run_dir = settings.parent
    block = str(run_dir.parent.relative_to("runs"))
    status = status_of(run_dir)
    counts.setdefault(block, Counter())[status] += 1
    if status == "unfinished":
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if (run_dir / "metrics.jsonl").exists() else []
        if lines:
            steps = json.loads(settings.read_text(encoding="utf-8"))["steps"]
            last = json.loads(lines[-1])
            val = next((json.loads(line)["val_loss"] for line in reversed(lines) if "val_loss" in json.loads(line)), None)
            tier = json.loads(settings.read_text(encoding="utf-8"))["tier"]
            ms = 1000 * TIERS[tier].tokens_per_step / last["tokens_per_s"]
            under_way.append(f"  {block}/{run_dir.name}: step {last['step']}/{steps}, loss {last['loss']:.3f}, {ms:.0f} ms/step, val {val if val is None else round(val, 3)}")
for block, counter in counts.items():
    print(block, dict(counter))
print("\n".join(under_way))
rates = Path("runs/tuning/rates.json")
if rates.exists():
    print("rates", json.loads(rates.read_text(encoding="utf-8")))
