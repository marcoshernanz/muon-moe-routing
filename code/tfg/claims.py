"""The plan's frozen predictions and the rule that decides them, on the table of endpoints ``tfg.analysis`` collects.

Each of the ``CLAIMS`` names an endpoint, the arms predicted lower and the arms predicted higher on it, and the views it is read in: at equal tokens, and at matched loss; a claim registered for one tier is read at that tier alone. Every pair of arms a claim sets against each other is a contrast. A contrast is a comparison of two arms paired by seed: the mean difference with a bootstrap interval over the seeds, the effect size, and each seed's sign. It passes when every seed's sign is the predicted one and the interval excludes zero, on the finished runs of at least the plan's three seeds and of every seed both arms have; a claim passes when every one of its contrasts does, in every view it is read in. A contrast whose mean has the predicted sign while a seed disagrees is a ``direction``, which does not pass, and so is a claim with such a contrast and none that failed.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY = "bias_span_first"  # integrated over the run in the first mixture: the primary endpoint
VIEWS = {"tokens": "equal tokens", "loss": "matched loss"}  # the two views' titles
RESAMPLES = 10_000  # of the bootstrap over seeds
MIN_SEEDS = 3  # a contrast is decided on at least the plan's three seeds
RESULT = ("seeds", "difference", "low", "high", "effect", "signs", "agreeing")  # of a comparison


@dataclass(frozen=True)
class Claim:
    """One prediction of the plan: on ``endpoint``, every arm of ``lower`` reads below every arm of ``higher``, in each of ``views``, at each of ``tiers`` (every tier when it is empty); the ``confirmatory`` ones are the registration's claims, the others are read by the same rule and reported apart."""

    name: str
    endpoint: str
    lower: tuple[str, ...]
    higher: tuple[str, ...]
    views: tuple[str, ...] = tuple(VIEWS)
    confirmatory: bool = True
    tiers: tuple[str, ...] = ()

    def pairs(self, arms: set[str], tiers: set[str]) -> list[tuple[str, str]]:
        """The contrasts of the claim among ``arms`` of runs at ``tiers``: every lower arm against every higher one, none at a tier the claim is not registered for."""
        if self.tiers and not tiers <= set(self.tiers):
            return []
        return [(low, high) for low in self.lower for high in self.higher if {low, high} <= arms]


# The two-by-two of the router's optimizer against the rest's.
BACKBONE_ON_ADAMW = ("adamw", "muon_router")  # (a), (g)
BACKBONE_ON_MUON = ("muon", "muon_except_router")  # (b), (c)
ROUTER_ON_MUON = ("muon", "muon_router")  # (b), (g)
EXPERTS_ON_MUON = ("muon", "muon_except_router", "muon_experts")  # (b), (c), (d)
EXPERTS_ON_ADAMW = (  # (a), (e), (g), (h), (i)
    "adamw",
    "muon_attention",
    "muon_router",
    "muon_dense",
    "muon_attention_dense",
)
# Muon on one part of the backbone: the experts (d), attention (e), the dense block 0 (h), and the
# two groups that feed the first mixture together (i).
PARTS = ("muon_experts", "muon_attention", "muon_dense", "muon_attention_dense")
RAISING_PARTS = ("muon_experts", "muon_attention_dense")  # (d), (i): above all-Muon in the pilot
EQUAL_RATE = "muon-muonlr0.5x"  # (b) with Muon at AdamW's tuned rate, half its own
FAST = ("adamw-balancer4x", "muon-balancer4x")  # (a) and (b) with the bias moving four times faster
RECIPE = (  # (a), (b) and (c) with the recipes' sequence-wise balance loss, at 1e-4, beside the bias
    "adamw-balanceloss0.0001",
    "muon-balanceloss0.0001",
    "muon_except_router-balanceloss0.0001",
)
OVERLOAD = "max_load_ratio_first_early"  # the first mixture's busiest expert over its fair share
CLAIMS = [  # sections 7.2 and 7.5 of the plan; the specialization and the loss at equal tokens only
    Claim("S", PRIMARY, BACKBONE_ON_ADAMW, BACKBONE_ON_MUON),
    Claim("S-rate", PRIMARY, ("adamw",), (EQUAL_RATE,)),
    Claim("S-parts", PRIMARY, ("adamw",), RAISING_PARTS),
    Claim("G", "router_input_similarity_first", BACKBONE_ON_ADAMW, BACKBONE_ON_MUON),
    Claim("G-parts", "router_input_similarity_first", PARTS, BACKBONE_ON_MUON),
    Claim("R", "bias_span", ("adamw",), ("muon",)),  # Gate 1's reversal, on fresh seeds
    Claim("M", "load_churn_warmup", ("adamw", "muon"), ("muon_except_router",)),
    Claim("C3", "usage_perplexity_final", ("adamw-unbalanced",), ("muon-unbalanced",)),
    Claim("C4", "specialization", ("adamw",), ("muon",), views=("tokens",)),
    Claim("C5", "val_loss", ("muon",), ("adamw",), views=("tokens",)),
    Claim("C6", "update_load_correlation", EXPERTS_ON_MUON, EXPERTS_ON_ADAMW),
    # Amendment 3: the second chain's reading, which found them on its seeds, tested at twice the size.
    Claim("O", OVERLOAD, ("adamw",), BACKBONE_ON_MUON, tiers=("confirmation",)),
    Claim("O-router", OVERLOAD, ("muon",), ("muon_except_router",), tiers=("confirmation",)),
    Claim(
        "U",
        "router_update_common_share_first",
        ("adamw", "muon"),
        ("muon_except_router",),
        tiers=("confirmation",),
    ),
    # Amendment 3: the balancer's speed (H9), on seeds of its own at the mechanism tier.
    Claim("S-fast", PRIMARY, FAST[:1], FAST[1:]),
    Claim("L", PRIMARY, FAST[1:], ("muon",)),
    Claim("O-fast", OVERLOAD, FAST[1:], ("muon",)),
    # Amendment 5: the recipes' balance loss (H10), on RQ3's seeds, against RQ3's runs without it.
    Claim("S-loss", PRIMARY, RECIPE[:1], RECIPE[1:]),
    Claim("G-loss", "router_input_similarity_first", RECIPE[:1], RECIPE[1:]),
    Claim("L-loss", PRIMARY, RECIPE[1:2], ("muon",)),
    Claim("O-loss", OVERLOAD, RECIPE[1:2], ("muon",), confirmatory=False),
    Claim("L-loss-router", PRIMARY, RECIPE[2:], ("muon_except_router",), confirmatory=False),
    Claim(  # the router-row geometry of the first registration, read again: exploratory
        "C7",
        "router_row_dispersion_early",
        ROUTER_ON_MUON,
        ("adamw", "muon_except_router", *PARTS),
        confirmatory=False,
    ),
]
CONFIRMATORY = list(dict.fromkeys(c.endpoint for c in CLAIMS if c.confirmatory))
HEADLINE = ("muon", "adamw")  # the pair the views figure sets side by side


def compare(table: pd.DataFrame, name: str, reference: str, *, names: list[str]) -> pd.DataFrame:
    """The difference of every endpoint in ``names``, the arm ``name`` minus the reference arm, paired by seed.

    With a 95% percentile bootstrap interval over the seeds, the effect size (the difference over the pooled seed spread of the two arms), each seed's sign, and the number of seeds whose sign agrees with the mean's. When every seed's sign agrees the interval excludes zero by construction, since every resample's mean has that sign: the signs decide, the interval shows the size.
    """
    ours = table[table["arm"] == name].set_index("seed")
    theirs = table[table["arm"] == reference].set_index("seed")
    seeds = ours.index.intersection(theirs.index)
    if seeds.empty:
        raise ValueError(f"{name} and {reference} share no seed")
    rows: list[dict[str, object]] = []
    for endpoint in names:
        if endpoint not in table.columns:
            continue
        mine, yours = ours.loc[seeds, endpoint], theirs.loc[seeds, endpoint]
        paired = (mine - yours).dropna()
        if paired.empty:  # neither arm has the endpoint, like a panel never evaluated
            continue
        diffs = paired.to_numpy(dtype=float)
        mean = float(diffs.mean())
        spread = float(np.sqrt((mine.var(ddof=1) + yours.var(ddof=1)) / 2))  # NaN with one seed
        low, high = mean, mean
        if len(diffs) > 1:
            interval = stats.bootstrap(
                (diffs,),
                np.mean,
                n_resamples=RESAMPLES,
                confidence_level=0.95,
                method="percentile",
                rng=0,
            )
            low, high = (
                float(interval.confidence_interval.low),
                float(interval.confidence_interval.high),
            )
        rows.append(
            {
                "arm": name,
                "reference": reference,
                "endpoint": endpoint,
                "seeds": len(diffs),
                "difference": mean,
                "low": low,
                "high": high,
                "effect": mean / spread if spread > 0 else np.nan,
                "signs": "".join("+" if d > 0 else "-" if d < 0 else "0" for d in diffs),
                "agreeing": int((np.sign(diffs) == np.sign(mean)).sum()),
            }
        )
    return pd.DataFrame(rows)


def pair_key(mine: str, yours: str) -> tuple[str, str]:
    """The two arms in name order: what a matched-loss cut is kept under, whichever is lower."""
    low, high = sorted((mine, yours))
    return low, high


def contrast(
    table: pd.DataFrame, claim: Claim, pair: tuple[str, str], *, view: str
) -> dict[str, object]:
    """One contrast of a claim in one view, decided by the plan's rule on the finished runs of the two arms in ``table``: ``pass`` when every seed's difference, the lower arm minus the higher one, is negative and the interval lies below zero; ``direction`` when the mean is negative but a seed disagrees; ``fail`` otherwise; ``incomplete`` while a seed both arms have lacks a finished run or the endpoint, or fewer than ``MIN_SEEDS`` have both."""
    lower, higher = pair
    done = table[table["status"] == "finished"]
    expected, finished = (
        len(set(t.loc[t["arm"] == lower, "seed"]) & set(t.loc[t["arm"] == higher, "seed"]))
        for t in (table, done)
    )
    paired = compare(done, lower, higher, names=[claim.endpoint]) if finished else pd.DataFrame()
    result: dict[Any, Any] = paired.iloc[0].to_dict() if len(paired) else {"seeds": 0}
    if result["seeds"] < max(expected, MIN_SEEDS):
        verdict = "incomplete"
    elif set(result["signs"]) == {"-"} and result["high"] < 0:
        verdict = "pass"
    else:
        verdict = "direction" if result["difference"] < 0 else "fail"
    return {
        "claim": claim.name,
        "confirmatory": claim.confirmatory,
        "endpoint": claim.endpoint,
        "lower": lower,
        "higher": higher,
        "view": view,
        "expected_seeds": expected,
        **{name: result.get(name) for name in RESULT},
        "verdict": verdict,
    }


def contrasts(table: pd.DataFrame, cuts: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Every contrast of every claim the runs' arms and tier allow, in each of the claim's views: at equal tokens from ``table``, at matched loss from the pair's own cut in ``cuts``."""
    rows = [
        contrast(
            table if view == "tokens" else cuts.get(pair_key(*pair), table.iloc[:0]),
            claim,
            pair,
            view=view,
        )
        for claim in CLAIMS
        for pair in claim.pairs(set(table["arm"]), set(table["tier"]))
        for view in claim.views
    ]
    return pd.DataFrame(rows)


def decide(verdicts: pd.Series) -> str:
    """A claim's verdict from its contrasts': ``pass`` when all pass; once one can no longer pass, ``fail`` when one failed and ``direction`` when the ones that cannot pass all have the predicted sign in the mean; else ``incomplete``; empty when it has none."""
    if verdicts.empty:
        return ""
    for verdict in ("fail", "direction"):  # neither passes
        if (verdicts == verdict).any():
            return verdict
    return "pass" if (verdicts == "pass").all() else "incomplete"


def claims(decided: pd.DataFrame) -> pd.DataFrame:
    """One row per claim with a contrast among the runs: how many contrasts it has, its verdict in each view, and its verdict, which takes every contrast in every view."""
    rows: list[dict[str, object]] = []
    for claim in CLAIMS:
        mine = decided[decided["claim"] == claim.name] if len(decided) else decided
        if mine.empty:
            continue
        rows.append(
            {
                "claim": claim.name,
                "confirmatory": claim.confirmatory,
                "endpoint": claim.endpoint,
                "contrasts": len(mine),
                **{view: decide(mine.loc[mine["view"] == view, "verdict"]) for view in VIEWS},
                "verdict": decide(mine["verdict"]),
            }
        )
    return pd.DataFrame(rows)
