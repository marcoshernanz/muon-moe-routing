"""The balance-loss block's check on the host: whether this code, without the loss, trains RQ3's run of all-Muon at seed 3 as RQ3 did, bit for bit, over the steps the check ran, so that each run of the block differs from its RQ3 run by the loss alone. Compares every metrics record to step 250, the time per step left out, the routing readings of the same steps, and the settings RQ3's run recorded, the device aside. Usage, from ~/code on the host: uv run python ai-experiments/balance_loss_2026_09/check.py CHECK_DIR REFERENCE_DIR; the last line says IDENTICAL or DIFFERENT."""

import json
import sys
from pathlib import Path

LAST = 250  # the check's last step, its first evaluation

check, reference = map(Path, sys.argv[1:3])


def records(path: Path) -> list[dict]:
    """The records to the check's last step; a line the kill cut short is left out."""
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record["step"] <= LAST:
            record.pop("tokens_per_s", None)
            kept.append(record)
    return kept


same = True
for name in ("metrics.jsonl", "routing.jsonl"):
    mine, theirs = records(check / name), records(reference / name)
    differ = [a["step"] for a, b in zip(mine, theirs) if a != b]
    first = f", the first at step {differ[0]}" if differ else ""
    print(f"{name}: {len(mine)} records against {len(theirs)}, {len(differ)} differ{first}")
    same = same and bool(mine) and len(mine) == len(theirs) and not differ
settings = json.loads((check / "settings.json").read_text(encoding="utf-8"))
recorded = json.loads((reference / "settings.json").read_text(encoding="utf-8"))
changed = sorted(key for key in recorded if key != "device" and recorded[key] != settings.get(key))
print(f"settings that differ: {', '.join(changed) or 'none'}")
same = same and not changed
print("IDENTICAL" if same else "DIFFERENT")
