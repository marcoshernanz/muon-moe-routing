"""Compares the runs baseline.sh trained: for each mode, every metrics and routing record of the run before the change with the run after it, the time per step left out, then every tensor of the saved model and optimizer state; the settings must differ by the added balance_loss alone. Usage: uv run python baseline.py DIR; prints one line per mode and SAME or CHANGED last."""

import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])


def records(path: Path) -> list[dict]:
    kept = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in kept:
        record.pop("tokens_per_s", None)
    return kept


def tensors(value: object, prefix: str = "") -> dict[str, torch.Tensor]:
    """Every tensor in a nested state, by its path."""
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    items = (
        value.items()
        if isinstance(value, dict)
        else enumerate(value)
        if isinstance(value, (list, tuple))
        else ()
    )
    found: dict[str, torch.Tensor] = {}
    for key, item in items:
        found |= tensors(item, f"{prefix}/{key}")
    return found


same = True
for mode in ("eager", "compiled"):
    before, after = root / f"before-{mode}", root / f"after-{mode}"
    counts = []
    for name in ("metrics.jsonl", "routing.jsonl"):
        a, b = records(before / name), records(after / name)
        counts.append(
            f"{name} {len(a)} records, {sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))} differ"
        )
        same = same and a == b
    old = tensors(torch.load(before / "checkpoint.pt", weights_only=False))
    new = tensors(torch.load(after / "checkpoint.pt", weights_only=False))
    unequal = [k for k in old if k not in new or not torch.equal(old[k], new[k])]
    counts.append(f"checkpoint {len(old)} tensors, {len(unequal)} differ")
    same = same and not unequal and old.keys() == new.keys()
    settings = [
        json.loads((run / "settings.json").read_text(encoding="utf-8")) for run in (before, after)
    ]
    added = sorted(set(settings[1]) - set(settings[0]))
    changed = sorted(k for k in settings[0] if settings[0][k] != settings[1].get(k))
    counts.append(f"settings added {added}, changed {changed}")
    same = same and added == ["balance_loss"] and not changed
    print(f"{mode}: " + "; ".join(counts))
print("SAME" if same else "CHANGED")
