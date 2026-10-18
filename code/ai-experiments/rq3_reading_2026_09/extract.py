"""Every logged reading of the second chain's runs as arrays, for the readings that follow.

From each run's routing.jsonl and metrics.jsonl: per logged step and mixture, the scalar routing readings; the first mixture's per-expert biases, loads and expert update norms; every group's weight norm, update norm and angular step; the training loss per logged step and the validation loss per evaluation. Written once to runs/rq3-reading/readings.npz (runs/ is not in git; the file is derived and rebuilt by this script). Run from code/: uv run python ai-experiments/rq3_reading_2026_09/extract.py
"""

import glob
import json
import os

import numpy as np

SCALARS = ["bias_span", "usage_perplexity", "max_load_ratio", "load_churn", "router_entropy", "router_logit_norm", "router_input_similarity", "mean_score_span", "raw_usage_perplexity", "raw_max_load_ratio", "margin_mean", "margin_low", "update_load_correlation", "router_row_dispersion", "router_update_common_share", "router_svd_entropy", "router_row_similarity", "residual_max", "router_logit_max", "expert_output_max", "expert_kurtosis_max"]
GROUPS = ["router", "experts", "shared", "attention", "dense", "tables_and_norms", "gains"]
GROUP_READINGS = ["weight_norm", "update_norm", "angular_step"]
OUT = "runs/rq3-reading/readings.npz"

arrays = {}
names = []
for run in sorted(glob.glob("runs/rq3/mechanism-*")):
    name = os.path.basename(run)
    names.append(name)
    steps, scalars, bias, load, update, groups = [], [], [], [], [], []
    for line in open(f"{run}/routing.jsonl"):
        record = json.loads(line)
        steps.append(record["step"])
        scalars.append([[np.nan if m[k] is None else m[k] for k in SCALARS] for m in record["mixtures"]])  # [B, S]
        first = record["mixtures"][0]
        bias.append(first["bias"])
        load.append(first["load"])
        update.append(first["expert_update_norm"])
        groups.append([[np.nan if record["groups"].get(g, {}).get(r) is None else record["groups"][g][r] for r in GROUP_READINGS] for g in GROUPS])
    metrics = [json.loads(line) for line in open(f"{run}/metrics.jsonl")]
    arrays[f"{name}/steps"] = np.array(steps)
    arrays[f"{name}/scalars"] = np.array(scalars, dtype=np.float32)  # [T, B, S]
    arrays[f"{name}/bias"] = np.array(bias, dtype=np.float32)  # [T, E]
    arrays[f"{name}/load"] = np.array(load, dtype=np.float32)  # [T, E]
    arrays[f"{name}/update"] = np.array(update, dtype=np.float32)  # [T, E]
    arrays[f"{name}/groups"] = np.array(groups, dtype=np.float32)  # [T, G, R]
    arrays[f"{name}/loss_steps"] = np.array([m["step"] for m in metrics])
    arrays[f"{name}/loss"] = np.array([m["loss"] for m in metrics], dtype=np.float32)
    arrays[f"{name}/val_steps"] = np.array([m["step"] for m in metrics if "val_loss" in m])
    arrays[f"{name}/val_loss"] = np.array([m["val_loss"] for m in metrics if "val_loss" in m], dtype=np.float32)
    print(name, len(steps), "readings", arrays[f"{name}/scalars"].shape)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, names=np.array(names), scalars=np.array(SCALARS), groups=np.array(GROUPS), group_readings=np.array(GROUP_READINGS), **arrays)
print("written", OUT, os.path.getsize(OUT) // 2**20, "MB")
