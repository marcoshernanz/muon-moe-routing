"""Tests of several workers on one block: the claim on a run, and two processes sharing the tuning and stress blocks at the laptop tier; under a minute on a laptop."""

import json
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path

from tfg import workers
from tfg.program import main
from tfg.tests.synthetic import write_shards

CODE = Path(__file__).parents[2]  # where ``python -m tfg...`` runs from
TOKENS = 5 * 8 * 64  # five laptop steps
HOLD = """
import sys, time
from pathlib import Path
from tfg import workers
with workers.claim(Path(sys.argv[1]), wait=False) as mine:
    print("held" if mine else "not held", flush=True)
    time.sleep(60)
"""


def test_a_run_another_worker_holds_is_skipped_or_waited_for(tmp_path: Path) -> None:
    run_dirs = [tmp_path / "block" / name for name in ("a", "b", "c")]
    trained: list[str] = []
    held, release = threading.Event(), threading.Event()

    def other_worker() -> None:
        with workers.claim(run_dirs[1], wait=False) as mine:
            assert mine
            held.set()
            release.wait()
            trained.append("b, by the other worker")

    other = threading.Thread(target=other_worker)
    other.start()
    held.wait()
    workers.each(run_dirs, lambda run_dir: trained.append(run_dir.name), wait=False)
    assert trained == ["a", "c"]
    threading.Timer(0.2, release.set).start()
    workers.each(run_dirs, lambda run_dir: trained.append(run_dir.name), wait=True)
    other.join()
    assert trained == ["a", "c", "a", "c", "b, by the other worker", "b"]
    assert sorted(path.name for path in (tmp_path / "block").iterdir()) == [
        ".a.lock",
        ".b.lock",
        ".c.lock",
    ]


def test_the_system_frees_the_claim_of_a_worker_that_dies(tmp_path: Path) -> None:
    run_dir = tmp_path / "block" / "run"
    command = [sys.executable, "-c", HOLD, str(run_dir)]
    with subprocess.Popen(command, cwd=CODE, stdout=subprocess.PIPE, text=True) as holder:
        assert holder.stdout is not None and holder.stdout.readline() == "held\n"
        with workers.claim(run_dir, wait=False) as mine:
            assert not mine
        holder.kill()  # no chance to let go of anything itself
        holder.wait()
    with workers.claim(run_dir, wait=False) as mine:
        assert mine


def lines(path: Path) -> str:
    """A log's records without the throughput, which depends on the machine's load; as text, where a NaN equals itself."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return json.dumps([{k: v for k, v in r.items() if k != "tokens_per_s"} for r in records])


def test_two_workers_share_a_block_and_train_every_run_once_as_one_would(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    flags = [
        "--data-dir", str(data_dir), "--no-compiled", "--warmup-steps", "2", "--log-every", "5",
        "--eval-every", "10", "--eval-batches", "1", "--checkpoint-every", "10",
        "--tier", "laptop", "--tokens", str(TOKENS),
    ]  # fmt: skip
    grids = ["--adamw-rates", "3e-3", "1e-5", "--muon-rates", "3e-3", "--control-rates", "1e-3"]
    stress = ["--seeds", "0", "--multipliers", "1", "4"]
    blocks = [["tuning", *flags, *grids], ["rq2", *flags, *stress]]
    together, alone = tmp_path / "together", tmp_path / "alone"
    chain = " && ".join(
        subprocess.list2cmdline(
            [sys.executable, "-m", "tfg.program", *block, "--runs-dir", str(together)]
        )
        for block in blocks
    )
    run = partial(
        subprocess.Popen,
        chain,
        shell=True,
        cwd=CODE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with run() as first, run() as second:
        outputs = [first.communicate(timeout=600)[0], second.communicate(timeout=600)[0]]
    assert first.returncode == second.returncode == 0, outputs
    assert all('{"step": 5' in output for output in outputs)  # each trained some of the runs
    for block in blocks:  # the same commands, one worker
        main([*block, "--runs-dir", str(alone)])

    runs = sorted(path.parent.relative_to(alone) for path in alone.rglob("metrics.jsonl"))
    assert runs == sorted(
        path.parent.relative_to(together) for path in together.rglob("metrics.jsonl")
    )
    assert len(runs) == 6 + 3 + 10  # the tuning block, the calibration, the grid with 16x
    for run_dir in runs:
        for log in ("metrics.jsonl", "routing.jsonl"):
            assert lines(together / run_dir / log) == lines(alone / run_dir / log), run_dir / log
    for table in ("tuning/rates.json", "tuning/adamw/sweep.json"):
        assert (together / table).read_bytes() == (alone / table).read_bytes()
