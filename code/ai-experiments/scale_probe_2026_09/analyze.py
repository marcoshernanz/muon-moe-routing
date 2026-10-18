"""The scale probe's runs against the third chain's runs of the same arm and seed, over the same first 3,000 steps at the confirmation tier.

Each probe run is the third chain's run of its arm and seed (the tag's code, compiled, the first nine mixture blocks recomputed, the schedule of 3B tokens, the chain's rates, every shard present, so the same data order) stopped after step 3,000; a plain run adds the probe's readings alone, a centred one feeds the first mixture's router its inputs minus their batch mean, and one runs the bias four times faster. First, whether each plain run reproduces the chain's logs: every training loss and every routing reading of the first 3,000 steps. Then, per run beside the chain's runs of its seed, the first mixture from the thesis telemetry's routing log: the busiest expert's load over its share over steps 301 to 500 and over the early window (501 to 2,789), the span, the router-input cosine, the mean selection margin and its 10 percent quantile over the bias's step, the load churn, the logit norm and the router entropy over steps 501 to 3,000, and the validation loss at step 3,000. Last, the probe's own readings of the first mixture by window (the share of scores in the sigmoid's flat range, the share of tokens one bias step can move, the share of chosen experts the scores alone would not have chosen, the router's own margin without the bias) and the offset the mean input puts in each expert's logit against the bias that cancels it. Run from code/: uv run python ai-experiments/scale_probe_2026_09/analyze.py > ai-experiments/scale_probe_2026_09/outputs/analysis.txt
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PROBE = HERE / "outputs" / "runs"
CHAIN = Path("runs/confirmation")
STOP = 3000
U = 1e-3
WINDOWS = [(501, 1000), (1001, 2000), (2001, 3000)]
ROUTING = [  # label, routing key, window (low, high], divided by the bias's step
    ("busiest 301-500", "max_load_ratio", (300, 500), False),
    ("busiest early", "max_load_ratio", (500, 2789), False),
    ("span", "bias_span", (500, STOP), False),
    ("cosine", "router_input_similarity", (500, STOP), False),
    ("margin/u", "margin_mean", (500, STOP), True),
    ("low/u", "margin_low", (500, STOP), True),
    ("churn", "load_churn", (500, STOP), False),
    ("logit norm", "router_logit_norm", (500, STOP), False),
    ("entropy", "router_entropy", (500, STOP), False),
]


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def first_mixture(run: Path, rate: float) -> dict[str, float]:
    rows = lines(run / "routing.jsonl")
    out = {}
    for label, key, (low, high), per_step in ROUTING:
        values = [r["mixtures"][0][key] for r in rows if low < r["step"] <= high]
        out[label] = float(np.mean(values)) / (rate if per_step else 1)
    loss = [r for r in lines(run / "metrics.jsonl") if r["step"] == STOP and "val_loss" in r]
    out["val loss"] = loss[0]["val_loss"] if loss else float("nan")
    return out


def reproduces(probe: Path, chain: Path) -> str:
    """Whether the probe run's training losses and routing readings up to step 3,000 are the chain's, bit for bit."""
    def upto(rows: list[dict]) -> list[dict]:
        return [r for r in rows if r["step"] <= STOP]

    ours, theirs = upto(lines(probe / "metrics.jsonl")), upto(lines(chain / "metrics.jsonl"))
    losses = all(a["losses"] == b["losses"] for a, b in zip(ours, theirs)) and len(ours) == len(theirs)
    ours, theirs = upto(lines(probe / "routing.jsonl")), upto(lines(chain / "routing.jsonl"))
    routing = len(ours) == len(theirs) and all(a["mixtures"] == b["mixtures"] for a, b in zip(ours, theirs))
    return f"{len(ours)} logged steps; training losses {'identical' if losses else 'DIFFER'}, routing readings {'identical' if routing else 'DIFFER'}"


def main() -> None:
    probes = sorted(p for p in PROBE.glob("*") if (p / "probe.json").exists())
    labels = [label for label, *_ in ROUTING] + ["val loss"]

    print("Do the plain runs reproduce the chain's? Steps up to 3,000")
    for probe in probes:
        info = json.loads((probe / "probe.json").read_text())
        if not info["centre"] and info["balancer_multiplier"] == 1:
            chain = CHAIN / f"confirmation-{info['partition']}-s{info['seed']}"
            print(f"  {probe.name}: {reproduces(probe, chain)}")

    print("\nThe first mixture, probe runs beside the chain's runs of their seed (first 3,000 steps); busiest is the busiest expert's load over its share (8 is every token), margins over the bias's step of the run")
    print(f"  {'run':46} " + " ".join(f"{label:>15}" for label in labels))
    seeds = sorted({json.loads((p / "probe.json").read_text())["seed"] for p in probes})
    for seed in seeds:
        for arm in ("adamw", "muon", "muon_except_router"):
            values = first_mixture(CHAIN / f"confirmation-{arm}-s{seed}", U)
            print(f"  {'chain ' + arm + ' s' + str(seed):46} " + " ".join(f"{values[label]:15.4f}" for label in labels))
        for probe in probes:
            info = json.loads((probe / "probe.json").read_text())
            if info["seed"] != seed:
                continue
            values = first_mixture(probe, U * info["balancer_multiplier"])
            print(f"  {'probe ' + probe.name.removeprefix('confirmation-'):46} " + " ".join(f"{values[label]:15.4f}" for label in labels))
        print()

    print("The probe's readings of the first mixture by window (mean over logged steps): flat is the share of scores within 0.01 of 1 / of 0; flippable the share of tokens whose margin is under two bias steps; bias-decided the share of chosen experts the scores alone would not choose; raw margin the k-th score minus the next without the bias, over u, mean / 10 percent quantile")
    for probe in probes:
        readings = np.load(probe / "readings.npz")
        names = list(readings["summary_names"])
        steps, summary = readings["steps"], readings["summary"][:, 0]  # [T], the first mixture [T, 6]
        cells = []
        for low, high in WINDOWS:
            rows = summary[(steps >= low) & (steps <= high)].mean(0)
            v = dict(zip(names, rows))
            cells.append(f"{v['saturated_high']:.3f}/{v['saturated_low']:.3f} {v['flippable']:.3f} {v['bias_decided']:.3f} {v['raw_margin_mean'] / U:.1f}/{v['raw_margin_low'] / U:.2f}")
        print(f"  {probe.name.removeprefix('confirmation-'):36} " + "  |  ".join(f"{low}-{high}: {cell}" for (low, high), cell in zip(WINDOWS, cells)))

    print("\nThe offset (w_e . x_bar, the mean logit the uncentred mean input gives each expert) of the first mixture against its bias, by window: the offset's spread across experts, the scores' spread (mean score, max minus min), the bias span, the correlation across experts between offset and bias (the bias cancelling the offset reads near -1), and the mean input's length over an input's mean length (the shared direction's strength)")
    for probe in probes:
        readings = np.load(probe / "readings.npz")
        steps = readings["steps"]
        offset, bias, score, lengths = (readings[k][:, 0] for k in ("offset", "bias", "mean_score", "lengths"))  # [T, E], [T, 2]
        cells = []
        for low, high in WINDOWS:
            keep = (steps >= low) & (steps <= high)
            spread = np.ptp(offset[keep], axis=1).mean()
            score_spread = np.ptp(score[keep], axis=1).mean()
            span = np.ptp(bias[keep], axis=1).mean()
            corr = np.mean([np.corrcoef(o, b)[0, 1] for o, b in zip(offset[keep], bias[keep])])
            ratio = (lengths[keep, 0] / lengths[keep, 1]).mean()
            cells.append(f"{spread:.2f} {score_spread:.3f} {span:.3f} {corr:+.2f} {ratio:.2f}")
        print(f"  {probe.name.removeprefix('confirmation-'):36} " + "  |  ".join(f"{low}-{high}: {cell}" for (low, high), cell in zip(WINDOWS, cells)))


if __name__ == "__main__":
    main()
