"""The offset probe's runs (``probe.py``) on a Kaggle session of two T4s at the ``kaggle_t4`` tier, one worker per GPU, the jobs in order of importance so that a session cut short keeps the ones that matter: at seed 0 and then seed 1, all Muon, all AdamW, all Muon with the first mixture's router input centred, all Muon at half its rate, Muon on the experts alone, and all AdamW centred; at seed 2 the first three. Written by ``make_kernel.py``, which puts ``probe.py`` and the hashes of the thesis modules it uses in."""

import hashlib
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

HASHES = {}
PROBE = ""

CODE = Path("/tmp/code")  # outside /kaggle/working, so the venv is not saved as output
OUT = Path("/kaggle/working")
STEPS = 3000
DEADLINE = 9.5 * 3600  # seconds after which no new run starts, to stay inside the session
ARMS = [  # partition, Muon's rate multiplier, centred
    ("muon", 1.0, False),
    ("adamw", 1.0, False),
    ("muon", 1.0, True),
    ("muon", 0.5, False),
    ("muon_experts", 1.0, False),
    ("adamw", 1.0, True),
]
JOBS = [(*arm, seed) for seed in (0, 1) for arm in ARMS] + [(*arm, 2) for arm in ARMS[:3]]
start = time.perf_counter()


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=CODE if CODE.exists() else None, check=True)


sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv")
root = next(Path("/kaggle/input").rglob("pyproject.toml")).parent
shutil.copytree(root, CODE, dirs_exist_ok=True)
for name, expected in HASHES.items():
    found = hashlib.sha256((CODE / "tfg" / f"{name}.py").read_bytes()).hexdigest()
    print(f"tfg/{name}.py: {'the same as on the laptop' if found == expected else 'DIFFERENT from the laptop'}", flush=True)
(CODE / "probe.py").write_text(PROBE, encoding="utf-8")
sh("pip install -q uv && uv --version && uv sync --frozen")
sh('uv run python -c "from tfg.data import download; download(train_shards=3, validation_shards=1)" && ls data/fineweb-edu')

gpus = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True, check=True).stdout)
print(f"{gpus} GPUs, {len(JOBS)} runs", flush=True)
queue, lock = list(JOBS), threading.Lock()


def worker(gpu: int) -> None:
    while True:
        with lock:
            if not queue or time.perf_counter() - start > DEADLINE:
                return
            partition, multiplier, centre, seed = queue.pop(0)
        began = time.perf_counter()
        label = f"{partition}-{multiplier:g}-{'centred' if centre else 'plain'}-s{seed}"
        log = OUT / f"log-{label}.txt"
        flags = f"--partition {partition} --seed {seed} --muon-lr-multiplier {multiplier} --steps {STEPS}{' --centre' if centre else ''}"
        cmd = f"PYTHONPATH=. uv run python probe.py {OUT}/runs {flags} > {log} 2>&1"
        code = subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=CODE, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=False).returncode
        tail = log.read_text(encoding="utf-8").splitlines()[-1:] if log.exists() else []
        print(f"gpu {gpu}: {label}: exit {code} in {time.perf_counter() - began:.0f}s {str(tail)[:300]}", flush=True)


workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(max(gpus, 1))]
for thread in workers:
    thread.start()
for thread in workers:
    thread.join()
print(f"done in {time.perf_counter() - start:.0f}s; not run: {queue}", flush=True)
