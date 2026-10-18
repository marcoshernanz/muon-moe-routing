"""Several workers on one block: a machine with several GPUs runs the same command once per GPU, and each worker trains the runs no other one is training.

A worker claims a run by locking a file beside the run's directory, not inside it, since a run that starts over empties its directory. It holds the lock while it trains the run and loses it when it stops, however it stops: the system drops the locks of a process that dies. A run another worker holds is skipped, or, where the block goes on to read what the run produced, waited for; once its lock is free the run is trained again, which returns at once if it finished and resumes it if its worker died.
"""

import fcntl
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def claim(run_dir: Path, *, wait: bool) -> Generator[bool]:
    """Hold the run's lock inside, if it is free; with ``wait``, wait until it is. Yields whether this worker holds it."""
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with (run_dir.parent / f".{run_dir.name}.lock").open("w") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        yield True


def each(run_dirs: list[Path], train: Callable[[Path], object], *, wait: bool) -> None:
    """Train, in order, every run no other worker holds, and skip the rest; with ``wait``, then wait for each skipped run and train it once it is free, so that every run is over when this returns."""
    held: list[Path] = []
    for run_dir in run_dirs:
        with claim(run_dir, wait=False) as mine:
            if mine:
                train(run_dir)
            else:
                print(f"{run_dir}: another worker is on it", flush=True)
                held.append(run_dir)
    for run_dir in held if wait else ():
        with claim(run_dir, wait=True):
            train(run_dir)
