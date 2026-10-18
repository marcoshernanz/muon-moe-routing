"""The claims Amendment 3 adds to the confirmation tier, read on the second chain's runs by the plan's rule, as a check of what the confirmation tier would test and of its margin, not as a test: the readings were found on these seeds.

`O`, (a) below (b) and (c) on the first mixture's early overload (`max_load_ratio_first_early`); `O-router`, (b) below (c) on the same; and `U`, (a) and (b) below (c) on the share of the first mixture's router update that acts on the mean direction of its inputs (`router_update_common_share_first`). Beside them, read the same way and not proposed as a claim, the first mixture's score offset (`mean_score_span_first`) as `S` orders the arms. Uses `tfg.claims` on the analysis's tables in `runs/rq3/analysis/`, at equal tokens and at matched loss. Run from code/: uv run python ai-experiments/rq3_reading_2026_09/claims_as_if.py > ai-experiments/rq3_reading_2026_09/outputs/claims_as_if.txt
"""

import pandas as pd
from tfg.claims import BACKBONE_ON_ADAMW, BACKBONE_ON_MUON, Claim, contrast, pair_key

TABLES = "runs/rq3/analysis"
AS_IF = [
    Claim("O", "max_load_ratio_first_early", ("adamw",), BACKBONE_ON_MUON),
    Claim("O-router", "max_load_ratio_first_early", ("muon",), ("muon_except_router",)),
    Claim("offset", "mean_score_span_first", BACKBONE_ON_ADAMW, BACKBONE_ON_MUON, confirmatory=False),
    Claim("U", "router_update_common_share_first", ("adamw", "muon"), ("muon_except_router",)),
]

table = pd.read_csv(f"{TABLES}/endpoints.csv")
matched = pd.read_csv(f"{TABLES}/endpoints_matched.csv")
cuts = {pair_key(first, second): cut.drop(columns=["first", "second"]) for (first, second), cut in matched.groupby(["first", "second"])}
rows = []
for claim in AS_IF:
    for pair in claim.pairs(set(table["arm"]), set(table["tier"])):
        for view in claim.views:
            rows.append(contrast(table if view == "tokens" else cuts.get(pair_key(*pair), table.iloc[:0]), claim, pair, view=view))
result = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(result[["claim", "endpoint", "lower", "higher", "view", "seeds", "difference", "low", "high", "effect", "signs", "verdict"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nPer arm and seed, max_load_ratio_first_early:")
for arm in ["adamw", "muon_router", "muon", "muon_except_router", "muon-muonlr0.5x"]:
    print(f"{arm:20} " + " ".join(f"{v:.3f}" for v in table.loc[table["arm"] == arm].sort_values("seed")["max_load_ratio_first_early"]))
