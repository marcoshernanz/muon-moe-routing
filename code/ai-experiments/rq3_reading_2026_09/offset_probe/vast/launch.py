"""The offset probe's runs at the mechanism tier on a rented host of four RTX 5090s, one worker per card, each the second chain's run of its arm and seed (compiled, the first mixture block recomputed, the schedule of 1B tokens) stopped after step 3,000, with the probe's readings. The jobs, in order of importance so that a host stopped early keeps the ones that matter, a round of four at a time: the backbone on Muon with the first mixture's router input centred (seeds 3, 4, 5); the same arm plain at seed 3; the router on AdamW with Muon at half its rate, 2e-3, the first arm of the block proposed in `context/decision/03-after-the-second-chain.md` (seeds 3, 4, 5); the backbone on Muon with the bias moving at 0.004 a step, its second arm (seeds 3, 4, 5); all AdamW and all Muon at half its rate, at seed 3. No new job starts after DEADLINE seconds, to keep the host's cost inside its cap. Run from ~/code on the host: uv run python ai-experiments/rq3_reading_2026_09/offset_probe/vast/launch.py
"""

import os
import subprocess
import threading
import time
from pathlib import Path

OUT = Path("runs/offset-probe")
LOGS = Path("logs")
DEADLINE = 80 * 60
FLAGS = "--tier mechanism --steps 15259 --stop-after 3000 --checkpoint-every 10000 --compiled --recompute 1"
JOBS = [  # partition, Muon's rate multiplier, the balancer's multiplier, centred, seed
    ("muon", 1, 1, True, 3),
    ("muon", 1, 1, False, 3),
    ("muon_except_router", 0.5, 1, False, 3),
    ("muon", 1, 4, False, 3),
    ("muon", 1, 1, True, 4),
    ("muon_except_router", 0.5, 1, False, 4),
    ("muon", 1, 4, False, 4),
    ("adamw", 1, 1, False, 3),
    ("muon", 1, 1, True, 5),
    ("muon_except_router", 0.5, 1, False, 5),
    ("muon", 1, 4, False, 5),
    ("muon", 0.5, 1, False, 3),
]
start = time.perf_counter()
queue, lock = list(JOBS), threading.Lock()


def worker(gpu: int) -> None:
    while True:
        with lock:
            if not queue or time.perf_counter() - start > DEADLINE:
                return
            partition, multiplier, balancer, centre, seed = queue.pop(0)
        began = time.perf_counter()
        label = f"{partition}-{multiplier:g}-{balancer:g}-{'centred' if centre else 'plain'}-s{seed}"
        log = LOGS / f"probe-{label}.txt"
        flags = f"{FLAGS} --partition {partition} --seed {seed} --muon-lr-multiplier {multiplier} --balancer-multiplier {balancer}{' --centre' if centre else ''}"
        cmd = f"PYTHONPATH=. uv run python ai-experiments/rq3_reading_2026_09/offset_probe/probe.py {OUT} {flags} > {log} 2>&1"
        code = subprocess.run(cmd, shell=True, executable="/bin/bash", env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, check=False).returncode
        with (LOGS / "jobs.txt").open("a") as jobs:
            jobs.write(f"{time.strftime('%H:%M:%S')} gpu {gpu}: {label}: exit {code} in {time.perf_counter() - began:.0f}s\n")


LOGS.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
gpus = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True, check=True).stdout)
workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(gpus)]
for thread in workers:
    thread.start()
for thread in workers:
    thread.join()
(LOGS / "jobs.txt").open("a").write(f"done in {time.perf_counter() - start:.0f}s; not run: {queue}\n")
