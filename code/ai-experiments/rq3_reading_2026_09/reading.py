"""What the reading's scripts share: the second chain's arrays from extract.py, the arms and their labels, and readings over windows of steps; and the first mixture's scalars from any run's routing log, for the runs of the first chain and the tuning block.

Every script of this folder runs from code/ after extract.py.
"""

import json

import numpy as np

READINGS = np.load("runs/rq3-reading/readings.npz")
SCALARS = list(READINGS["scalars"])
GROUPS = list(READINGS["groups"])
ARMS = ["adamw", "muon_router", "muon_dense", "muon_attention", "muon_attention_dense", "muon_experts", "muon_except_router", "muon", "muon-muonlr0.5x", "adamw-unbalanced", "muon-unbalanced"]
BALANCED = ARMS[:9]
LABEL = {"adamw": "(a)", "muon": "(b)", "muon_except_router": "(c)", "muon_router": "(g)", "muon_experts": "(d)", "muon_attention": "(e)", "muon_dense": "(h)", "muon_attention_dense": "(i)", "muon-muonlr0.5x": "(b) half", "adamw-unbalanced": "(a) unbal", "muon-unbalanced": "(b) unbal"}
SEEDS = (3, 4, 5)
WARMUP = 500
LAST = 15259


def array(arm: str, seed: int, name: str) -> np.ndarray:
    return READINGS[f"mechanism-{arm}-s{seed}/{name}"]


def steps(arm: str, seed: int) -> np.ndarray:
    return array(arm, seed, "steps")


def scalar(arm: str, seed: int, key: str, block: int = 0) -> np.ndarray:  # [T]
    return array(arm, seed, "scalars")[:, block, SCALARS.index(key)]


def group(arm: str, seed: int, name: str, reading: int = 0) -> np.ndarray:  # [T]; reading 0 weight norm, 1 update norm, 2 angular step
    return array(arm, seed, "groups")[:, GROUPS.index(name), reading]


def window(arm: str, seed: int, key: str, low: int = WARMUP, high: int = LAST, block: int = 0) -> float:
    """The mean of a scalar over the logged steps in (low, high]; over the steps after the warmup by default, the plan's integration."""
    at = steps(arm, seed)
    return float(np.nanmean(scalar(arm, seed, key, block)[(at > low) & (at <= high)]))


def seed_mean(arm: str, key: str, low: int = WARMUP, high: int = LAST, block: int = 0) -> float:
    return float(np.mean([window(arm, s, key, low, high, block) for s in SEEDS]))


def first_mixture(path: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The steps and the first mixture's scalars of any run's routing log: [T] and [T, K]."""
    at, rows = [], []
    for line in open(f"{path}/routing.jsonl"):
        record = json.loads(line)
        at.append(record["step"])
        rows.append([record["mixtures"][0][k] for k in keys])
    return np.array(at), np.array(rows, dtype=float)
