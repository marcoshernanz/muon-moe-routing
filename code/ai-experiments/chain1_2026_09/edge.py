"""Between the tuning and Gate 1 in the first chain: exit 1 when AdamW's or Muon's sweep still has its best rate at an edge of its grid after the sweep's one extra rate, so that Gate 1 and the stress grid do not train at a rate the grid does not bracket (RUNBOOK section 5: widen the grid by hand and rerun). The control's sweep is not read, since the later blocks run the control at Muon's rate.

Run from ``code/``: ``uv run python ai-experiments/chain1_2026_09/edge.py``.
"""

import json
import sys
from pathlib import Path

edges = []
for optimizer in ("adamw", "muon"):
    table = json.loads((Path("runs/tuning") / optimizer / "sweep.json").read_text(encoding="utf-8"))
    print(f"{optimizer}: val_loss {table['val_loss']}, best {table['best']}, extension {table['extension']}, edge {table['edge']}", flush=True)
    if table["edge"]:
        edges.append(optimizer)
if edges:
    print(f"still at an edge: {edges}; the chain stops before Gate 1", flush=True)
    sys.exit(1)
