"""The scale probe's runs at the confirmation tier on a rented host, one worker per card, each the third chain's run of its arm and seed (compiled, the first nine mixture blocks recomputed, the schedule of 3B tokens, the chain's rates 7e-4 and 1.4e-3) stopped after step 3,000, with the probe's readings. The jobs, in order of importance so that a host stopped early keeps the ones that matter: the backbone on Muon with the first mixture's router input centred and plain at seed 0 (the plain run is the chain's own, with the new readings added, and checks that the host reproduces it); the router on AdamW, centred and plain, at seed 0; the backbone on Muon centred at seeds 1 and 2; all AdamW plain at seed 0, the new readings' baseline; the router on AdamW centred at seeds 1 and 2; the backbone on Muon with the bias moving at 0.004 a step at seed 0. No new job starts after DEADLINE, a time in seconds since the epoch that host.sh sets from its own start, so that a slow setup costs jobs and not the cap. Run from ~/code on the host: uv run python ai-experiments/scale_probe_2026_09/launch.py
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT = Path("runs/scale-probe")
LOGS = Path("logs")
DEADLINE = float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 150 * 60
FLAGS = "--tier confirmation --steps 45776 --stop-after 3000 --adamw-lr 7e-4 --muon-lr 1.4e-3 --checkpoint-every 10000 --compiled --recompute 9"
JOBS = [  # partition, the balancer's multiplier, centred, seed
    ("muon", 1, True, 0),
    ("muon", 1, False, 0),
    ("muon_except_router", 1, True, 0),
    ("muon_except_router", 1, False, 0),
    ("muon", 1, True, 1),
    ("muon", 1, True, 2),
    ("adamw", 1, False, 0),
    ("muon_except_router", 1, True, 1),
    ("muon_except_router", 1, True, 2),
    ("muon", 4, False, 0),
]
start = time.perf_counter()
queue, lock = list(JOBS), threading.Lock()


def worker(gpu: int) -> None:
    while True:
        with lock:
            if not queue or time.time() > DEADLINE:
                return
            partition, balancer, centre, seed = queue.pop(0)
        began = time.perf_counter()
        label = f"{partition}-{balancer:g}-{'centred' if centre else 'plain'}-s{seed}"
        log = LOGS / f"probe-{label}.txt"
        flags = f"{FLAGS} --partition {partition} --seed {seed} --balancer-multiplier {balancer}{' --centre' if centre else ''}"
        cmd = f"PYTHONPATH=. uv run python ai-experiments/scale_probe_2026_09/probe.py {OUT} {flags} > {log} 2>&1"
        with (LOGS / "jobs.txt").open("a") as jobs:
            jobs.write(f"{time.strftime('%H:%M:%S')} gpu {gpu}: {label}: started\n")
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
