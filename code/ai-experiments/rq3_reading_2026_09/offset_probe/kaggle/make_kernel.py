"""Write the one-file Kaggle kernel ``offset-probe-t4.py`` from ``launcher.py`` and ``../probe.py``, with the hashes of the thesis modules the probe uses, so that the kernel's log says whether the code on the dataset ``marcoshernanz/tfg-code`` is the laptop's.

Usage, from this directory: python make_kernel.py && kaggle kernels push -p .
"""

import hashlib
from pathlib import Path

HERE = Path(__file__).parent
CODE = HERE.parents[3]
MODULES = ("model", "optim", "telemetry", "tier", "train", "data", "checkpoint")

hashes = {name: hashlib.sha256((CODE / "tfg" / f"{name}.py").read_bytes()).hexdigest() for name in MODULES}
launcher = (HERE / "launcher.py").read_text(encoding="utf-8")
probe = (HERE.parent / "probe.py").read_text(encoding="utf-8")
assert "'''" not in probe
for marker in ("HASHES = {}", 'PROBE = ""'):
    assert launcher.count(marker) == 1, marker
kernel = launcher.replace("HASHES = {}", f"HASHES = {hashes!r}").replace('PROBE = ""', f"PROBE = r'''{probe}'''")
(HERE / "offset-probe-t4.py").write_text(kernel, encoding="utf-8")
print("written", len(kernel), "characters")
