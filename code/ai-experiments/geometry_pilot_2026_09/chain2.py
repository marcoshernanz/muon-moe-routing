"""Wall time and cost of the second chain of the plan's second registration on one host of four RTX 5090s, one worker per card, with the cost model of ``host_checks_2026_09/chains.py`` (the step measured on the host of EVIDENCE-LOG 15, a 90-second start per run, the waits of ``tfg.program``), which put the first chain at 15.8 hours before the 16x level against the 18.2 it took with it (EVIDENCE-LOG 16).

Run from this folder: ``python3 chain2.py``. The first two lines cost the chain as the second registration first had it: the confirmation tier (its re-check, a stage the final runs wait for, then (a), (b), (c) at three seeds on 3B tokens), then the RQ3 block (``tfg.program.rq3_runs``: eleven arms at each of three seeds, 1B tokens each at the mechanism tier), then the evaluation of both blocks' runs. The last two lines cost the second chain as it stands, the RQ3 block and its evaluation alone, and the confirmation tier with its evaluation as the optional chain after it. The dollars are at the price of the first chain's host and at the two cheapest qualifying hosts of 2026-09-23, with 400 GB of disk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "host_checks_2026_09"))
from chains import PRICES, simulate, step_ms

PRICES = {"first chain's host (Taiwan)": 2.21, **{k: v for k, v in PRICES.items() if k != "Taiwan"}}
RQ3 = (
    "adamw",
    "muon",
    "muon_except_router",
    "muon_experts",
    "muon_attention",
    "muon_router",
    "muon_dense",
    "muon_attention_dense",
    "muon",
    "adamw",
    "muon",
)  # the eight partitions, (b) at AdamW's rate, and the two arms without the balancer
EVALUATION_H = (
    (40 * 60 + 9 * 115) / 3600 / 4
)  # the panel and annex per run (EVIDENCE-LOG 15, item 9: about 60 s a mechanism run, 115 s a confirmation run) on four cards
RQ3_EVALUATION_H = 33 * 60 / 3600 / 4  # the same for the RQ3 block alone
CONFIRMATION_EVALUATION_H = 9 * 115 / 3600 / 4  # and for the confirmation tier alone


def second_chain(rq3_arms: tuple[str, ...], seeds: int = 3) -> list:
    m, c = "mechanism", "confirmation"
    return [
        ([(c, p, 1.0) for p in ("adamw", "muon") for _ in range(3)], True),  # the re-check
        ([(c, p, 3.0) for p in ("adamw", "muon", "muon_except_router") for _ in range(3)], False),
        ([(m, p, 1.0) for _ in range(seeds) for p in rq3_arms], False),
    ]


def main() -> None:
    steps = step_ms()
    for label, arms in (
        ("RQ3 with (i)", RQ3),
        ("RQ3 without (i)", tuple(p for p in RQ3 if p != "muon_attention_dense")),
    ):
        wall, busy = simulate(second_chain(arms), steps, 90.0)
        mechanism = simulate(
            [([("mechanism", p, 1.0) for _ in range(3) for p in arms], False)], steps, 90.0
        )
        confirmation = simulate(second_chain(arms)[:2], steps, 90.0)
        wall += EVALUATION_H
        print(
            f"{label}: {3 * len(arms)} RQ3 runs ({mechanism[1]:.1f} card-hours) and 15 confirmation runs ({confirmation[1]:.1f}); "
            f"the chain {wall:.1f} h wall, {busy:.1f} card-hours busy ({100 * busy / (4 * wall):.0f} percent), dollars "
            + " / ".join(f"{name} {wall * price:.0f}" for name, price in PRICES.items())
        )
    for label, blocks, evaluation in (
        (f"RQ3 alone: {3 * len(RQ3)} runs", second_chain(RQ3)[2:], RQ3_EVALUATION_H),
        ("The confirmation tier alone: 15 runs", second_chain(RQ3)[:2], CONFIRMATION_EVALUATION_H),
    ):
        wall, busy = simulate(blocks, steps, 90.0)
        wall += evaluation
        print(
            f"{label}, the chain {wall:.1f} h wall, {busy:.1f} card-hours busy ({100 * busy / (4 * wall):.0f} percent), dollars "
            + " / ".join(f"{name} {wall * price:.0f}" for name, price in PRICES.items())
        )


if __name__ == "__main__":
    main()
