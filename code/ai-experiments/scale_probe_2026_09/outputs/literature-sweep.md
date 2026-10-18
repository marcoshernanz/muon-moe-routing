# Literature sweep, 2026-09-28

Third chain's results (EVIDENCE-LOG §19–20) against published work. VERIFIED = source opened and claim seen; UNVERIFIED = search snippet only. Items already in `context/decision/LITERATURE-MAP.md` appear only where this sweep found something new in them.

---

## A. Dynamics of loss-free / bias-based balancing

**A1. Kimi K3 (Kimi Team), "Kimi K3: Open Frontier Intelligence", [arXiv 2607.24653](https://arxiv.org/abs/2607.24653), §2.3.3, Table 1 §3.2 — VERIFIED (HTML).**
States the sign rule as the fixed-step baseline — "The original method updates b with the fixed-step rule b_j^(t+1)=b_j^(t)+γ sign(l̄−l_j^(t))" — and names the dilemma behind results 1 and 3: γ "trades off slow adaptation against load oscillation". QB is an exact per-step solve, b̂_j^(t+1) ← −quantile_{1−k/n}(s_{:,j} − α^(t)) (Eq. 14), no rate. Sigmoid router (Eq. 13), no z-loss or logit cap, router optimizer not stated, 1 dense layer.
New vs map: the explicit γ rule and the oscillation sentence.

**A2. Arcee, "Arcee Trinity Large Technical Report" (Singh, Krauss, Jaghouar et al.), [arXiv 2602.17004](https://arxiv.org/abs/2602.17004), §2.3, §3.4.1, §6 — VERIFIED (HTML).**
SMEBU: v_i=(n̄−n_i)/n̄ (Eq. 23), ṽ_i=tanh(κv_i) (Eq. 24), m_i=βm_i+(1−β)Δb_i (Eq. 27), b_i += m_i (Eq. 28); λ=5e-4, β=0.5, κ=2. Sigmoid router (Eq. 16). **Six dense layers before the first MoE layer** (§3.4.1). Motivation (§2.3): the standard rule's "update step is biased towards oscillation". §6: "our initial runs faced router instability issues...MaxVio would remain stable then experience sudden climb as experts collapsed."
A Muon-trained frontier run that abandoned the fixed-step sign rule for the reason the thesis measures. New vs map (which had only "soft-clamped momentum bias updates, zero loss spikes").

**A3. Motif Technologies, "Motif 3: Technical Report", [arXiv 2608.09119](https://arxiv.org/abs/2608.09119), 2026-08-10, §4.2–4.3 — VERIFIED (HTML).**
Sigmoid routing, aux-loss-free bias at 1e-3, plus a sequence-wise balance loss at 1e-4 by default but **2.4e-4 for layer 2 and 2.0e-4 for layers 3–5**, because "the earliest MoE layers were more susceptible to expert-load imbalance than the remaining layers". Two dense first layers. Decaying Gaussian noise on the router logits: it "reduces the risk of premature routing lock-in". Routing-collapse handling: "we immediately replaced the affected layer's router parameters and expert-selection biases with those from an adjacent layer".
The closest independent confirmation of result 1's localisation: a production Muon run that had to give the earliest MoE layers extra balancing. Entirely new.

**A4. Ling Team, "Every Activation Boosted..." (Ling 2.0 / Ling-1T), [arXiv 2510.22115](https://arxiv.org/abs/2510.22115), §2.1–2.2, §3.2.1 — VERIFIED (HTML).**
Bias update "b_i=b_i+u×(sign(e_i)−mean(sign(e)))", with the stated modification "keeping the bias centered around zero". Router gate scaling 2.5. One dense layer (mini/flash), four (Ling-1T). AdamW.
Directly on result 1's common-mode drift: a frontier recipe removes exactly that drift by construction. New.

**A5. Larry Dial (Open Athena), "Mixture of Experts Quantile Balancing: Validated at 32B-A5B (1e22 FLOPs) Scale", <https://openathena.ai/blog/quantile-balancing/>, 2026-04-10 — VERIFIED.**
With QB, "supplementary methods to assist with balancing—such as leading dense layers, auxiliary losses, or extra capacity overload factors—were unnecessary". The earlier fixed-increment (0.001) run had "a harder time achieving balancing in the first 5,000 steps", with "experts completely dead for 50+ steps"; QB gave "zero loss spikes over the 326 billion token run" (Fig. 3 caption). Optimizer not stated. Grey literature, but an independent early-window report. New.

**A6. Han and Zhong, [arXiv 2512.03915](https://arxiv.org/abs/2512.03915) (rev. 2026-04-26) — VERIFIED (abs only).** Primal-dual; "a logarithmic expected regret bound under certain step-size choices". No oscillation theorem, no bias-growth bound, no layer dependence. Already mapped; the new point is that no oscillation theory exists to cite.

**A7. Neitemeier, Li, Serra, Scholl, Maskey, "Exact Quantile Balancing and Load-Error Injection for MoE", [arXiv 2609.28053](https://arxiv.org/abs/2609.28053), 2026-09-23, §E.1 — VERIFIED (HTML).** 7.5B MoEs; "50 Transformer layers, with two dense layers followed by 48 MoE layers". No sign-rule, early-training or per-layer analysis; optimizer not stated. Marginal.

**A8. UNVERIFIED (snippets only):** GLM-5 dense FFN in the first three layers; Nemotron 3 sigmoid router with expert biasing, aux-loss-free, rate 1e-3. Need a direct read before citing.

*Summary.* Supports: the fixed step's early lag is a known failure mode (A1, A5); two 2026 recipes replaced or damped it because it oscillates (A2) or lags (A5); the earliest MoE layers need extra help (A2 six dense layers, A3 per-layer aux weights, A5 "leading dense layers"); the common-mode drift is removed by construction in Ling 2.0 (A4). Contradicts: nothing. New vs map: A2's full rule and argument, A3 entirely, A4, A5.

---

## B. Muon and the MoE router

**B1. DeepSeek-V4, [arXiv 2606.19348](https://arxiv.org/abs/2606.19348), §2.1 — VERIFIED (HTML).** See F1. Caution: the fetched §2.1 text on dense-vs-MoE first blocks read ambiguously ("replaced the dense FFN layers in the initial several Transformer blocks with MoE layers"); **do not cite V4's dense-block count from this sweep without a direct read.**

**B2. Motif 3, §3.3 — VERIFIED.** "Muon for matrix-shaped parameters; AdamW for embeddings, output projection, and vector-shaped parameters"; the router is not excluded, so it sits on Muon by exclusion — a third data point on DeepSeek-V4/Moonlight's side. New.

**B3. NVIDIA Megatron-LM, "[ROADMAP][2026 Q3] Megatron Core MoE Roadmap", issue [#6757](https://github.com/NVIDIA/Megatron-LM/issues/6757), 2026-08-21 — VERIFIED.** Under Optimizer: "Add option to skip Muon for MoE routers #6504", within "explicit optimizer assignment for MoE routers and other parameters excluded from Muon". No reason given. Useful as a one-line motivation: the main training framework is adding a switch for the disagreement. New.

**B4. Varun Srivastava, "Loss-Free MoE Routing", <https://varunneal.github.io/essays/loss-free-moe>, 2026-07-24 — VERIFIED.** 12 layers, d=768, 64 experts top-2, 1.465B params, ~2.7B tokens; router on Adam (3e-3) vs Muon (2e-2), backbone fixed. "Muon produces a better-conditioned router than Adam throughout the duration of training" (Fig. 5); lower mean pairwise row cosine under Muon (Fig. 6); validation loss "nearly identical" (Fig. 4); MaxVio comparable (Fig. 3).
The only published head-to-head of the router's optimizer under loss-free balancing, and it finds no load difference — but its backbone is Adam-trained, so the shared direction the thesis's mechanism needs is absent. That confound is the decisive distinction. (The map's "Srivastava" attribution is correct; full name Varun Srivastava.)

**B5. Trinity Large §6 — VERIFIED (see A2).** Router collapse and MaxVio "sudden climb" under a Muon recipe; no optimizer contrast.

*Summary.* Nobody has published the router-on-AdamW vs router-on-Muon load contrast under loss-free balancing, so result 2 stands. B4 is the nearest null and is confounded by its Adam backbone — which the thesis's own (g) arm independently predicts. New vs map: B2, B3, and the B4 confound.

---

## C. Hidden-state anisotropy / a shared direction

**C1. Queipo-de-Llano, Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv, "Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin", [arXiv 2510.06477](https://arxiv.org/abs/2510.06477), rev. 2026-02-09 — VERIFIED (abs).**
Proves "massive activations necessarily produce representational compression"; both arise when the BOS token "develops extreme activation norms"; located in **middle** layers; 410M–120B; Mix–Compress–Refine. No optimizer comparison. (The "p1 > 0.9" anisotropy figure came from a snippet — UNVERIFIED.)
The mechanism candidate for the shared direction, and a depth caveat: their compression sits mid-network while the thesis's router-input cosine is highest at the first mixture and falls with depth. State it as an open difference; the thesis reads one layer that follows a dense block. New vs map.

**C2. "Evidence for Shared Routing Geometry and Dynamics in Sparse MoE", [arXiv 2609.02404](https://arxiv.org/abs/2609.02404), §3.1 — VERIFIED (HTML).**
"Since softmax probabilities and top-k selection are invariant to a common shift of all E expert logits, we remove this common-logit direction by centering the router across experts" (Eq. 1), W̃_ℓ = W_ℓ − 1_E w̄_ℓ^T. No layer-position analysis, no optimizer comparison, no link to load or to the bias.
Two uses: it centres the router **weights across experts** while the thesis's probe centres the router **input across tokens** — different nuisance directions, worth separating explicitly; and its invariance statement is the published version of the thesis's reading that the common-mode bias drift changes no routing. New detail in a mapped paper.

**C3. Vlassis et al. (ETH Zurich, ISTA, Red Hat AI), "Beyond Outliers: A Study of Optimizers Under Quantization", [arXiv 2509.23500](https://arxiv.org/abs/2509.23500), §3.1, Figs. 1–2, 4 — VERIFIED (HTML).**
"across all optimizers in our experiments, Muon consistently achieves the lowest MMR"; "increasing the learning rate consistently leads to higher MMR values, independent of the optimizer used"; Muon nonetheless has "the highest gain for linear layers"; MMR and kurtosis correlate poorly with accuracy.
Supports the caution that Muon does not simply make bigger outliers, and supports the rate dependence (the first mixture's span grows with Muon's rate, §19 item 5). New.

*Summary.* Supports: a single dominant residual direction is a proven consequence of massive activations (C1); the common-logit shift is provably routing-irrelevant (C2). Contradicts nothing head-on, but C1's middle-layer location, plus C3 and Ruan et al. (2606.09658, mapped), tell against a general "Muon makes representations anisotropic" claim — the finding must stay specific to the first mixture's router input. No published measurement of a shared router-input direction by optimizer was found.

---

## D. Muon learning-rate transfer across width

**D1. Kim, Lee, Bak, Park, Kim (Kakao), "Let's Scale Step by Step", [arXiv 2608.20061](https://arxiv.org/abs/2608.20061), §2.1, §3.2, App. A — VERIFIED (HTML).**
μP factor fan_in_base/fan_in for matrix-like weights (Table 2); "the optimal learning rate identified in the base proxy consistently transfers across models scaled to 2×, 4×, and 8× the base width" (§3.2, Fig. 3b); "We match Muon optimizer's update RMS to that of AdamW by scaling the learning rate by 0.2·√max(A,B)" (App. A). The router is matrix-like and takes the same scaling.
The citation for result 4: transfer holds **under μP**; the thesis trains in standard parameterization, where the optimal rate is expected to fall with width, so the observed halving is expected SP behaviour and is partly confounded with the 3× longer horizon. New vs map.

**D2. Małaśnicki et al., "μ-Parametrization for Mixture of Experts", [arXiv 2508.09752](https://arxiv.org/abs/2508.09752), §4.2–4.3 — VERIFIED (HTML).**
Under SP, "optimal learning rate is different for different model sizes" (Fig. 1, left); under μP it is stable; "optimal learning rate is preserved across varied number of experts" (Fig. 2a) but "not preserved across different granularities" (Fig. 2b). AdamW only. The SP baseline for result 4, plus a warning that top-k/granularity changes break transfer. New.

**D3. Tian et al., "Hyperparameter Scaling Laws Across MoE Sparsity", [arXiv 2609.08690](https://arxiv.org/abs/2609.08690), 2026-09-08 — VERIFIED (abs).**
"the optimal learning rate scales with training compute C and remains robust to the allocation between model size and data"; the activation ratio A "enters both relationships as an additional multiplicative power-law factor"; 1,800 runs to 6B. Optimizer not stated in the abstract. A cleaner predictor for result 4 than the token-horizon exponent alone, because the confirmation tier changed both size and tokens. New.

**D4. Ren et al., HyperP, [arXiv 2603.28743](https://arxiv.org/abs/2603.28743), §3.2, §3.4 — VERIFIED (HTML).** "η* = 24.27·T^(−0.320)" (Eq. 8); hypersphere optimization "preserves width transfer without explicit 1/w learning rate scaling as in standard μP" (§3.2). The exponent is mapped; the equation and the width statement are new.

*Summary.* All sources agree the optimal rate transfers across width only under μP or an equivalent; in SP it moves, which is the thesis's setting. Nothing contradicts result 4. New vs map: D1's App. A and Fig. 3b, D2, D3.

---

## E. Novelty check

Searches on 2026-09-28: arXiv API `all:"Muon" AND all:"mixture-of-experts"` (22 entries back to Moonlight 2502.16982); `abs:"load balancing" AND abs:"Muon"` (5); `abs:"router" AND abs:"Muon"` (4 ML entries); `abs:"mixture-of-experts" AND abs:"optimizer"` from 2026-08-01 (8); plus web searches on balancing effort / bias span by optimizer, router-optimizer ablations, router saturation, and the 2026 flagship reports.

**No collision.** Nothing measures the loss-free bias span, or any balancing-effort proxy, as a function of the backbone's optimizer; nothing reports router-on-AdamW worsening early expert overload. The nearest items are unchanged from the map: the HiLD paper (no balancer, opposite direction, single seed), Li [2607.19771](https://arxiv.org/abs/2607.19771) (its abstract confirms only "a router collapsing to a single expert" and a cap that projects out "the first-order growth of the single top singular direction"; the 0.995 figure is in the body, not re-verified here), Lau and Su [2605.18106](https://arxiv.org/abs/2605.18106), and SSO [2601.08393](https://arxiv.org/abs/2601.08393).

Two **partial overlaps**, both new, worth citing pre-emptively so the framing is not scooped: Motif 3 (A3) reports the earliest MoE layers being the worst for load imbalance under a Muon backbone; Trinity (A2) reports the fixed-step rule oscillating and routers collapsing under Muon. Neither decomposes by parameter group, measures router-input geometry, or contrasts optimizers.

---

## F. Router logit growth and selection-margin saturation (added in-task)

**F1. Score function.** DeepSeek-V4 §2.1 — VERIFIED: "Different from DeepSeek-V3, we change the activation function that computes the affinity scores from Sigmoid(·) into Sqrt(Softplus(·))." **No reason is given there**; do not attribute one. Sqrt(Softplus) is unbounded above, so sigmoid's saturation ceiling and the 1+2u span bound do not apply to it — that is the thesis's inference and must be written as such. Kimi K3 keeps sigmoid (Eq. 13) with no z-loss or cap — VERIFIED. HyperP's SqrtGate (§3.7) rescales gate *weights* for RMS preservation, not selection scores — VERIFIED, so not a precedent.

**F2. Router logit control in 2026 recipes.** Ling 2.0 §2.2: gate scaling 2.5 "to stabilize the root mean square of the gate outputs" — VERIFIED. Motif 3 §4.2: "We keep the routing computation, including router scores and expert-selection biases, entirely in FP32 to avoid numerical errors in the discrete expert-selection decisions" — VERIFIED; the production answer to the thesis's ~1e-4 margins, which sit at or below bf16 resolution near a sigmoid score of 1. Motif 3 also adds decaying router noise and replaces collapsed routers from an adjacent layer. **No frontier recipe found uses a z-loss or a logit cap on a sigmoid router.**

**F3. Saturation under Muon / large rates.** No peer-reviewed source found. Two items. `acupof-ai/aupai` PR [#726](https://github.com/acupof-ai/aupai/pull/726), `cklxx`, 2026-09-26 — VERIFIED but a code PR on an undisclosed model: "router row norm grows init ~1 → ~11 in 3000 steps" at lr 0.01, cross-expert logit std 50–100, 93–96% of logits below −6 "where sigmoid ≈ 0"; its probe reports "cross-expert std after removing each token's common mode"; fix z := C·tanh(z/C), C=12, with the caveat that "the cap bounds the SCORE but does not shrink raw router weights". Informal corroboration only, never a citation. "Z-Loss Backward Geometry in Dense Output Heads and Sparse Routers", [arXiv 2609.16179](https://arxiv.org/abs/2609.16179), 2026-07-31 — VERIFIED: softmax routers, AdamW only, no sigmoid-saturation or margin analysis. Not useful.

**F4. The sign update as a bang-bang controller.** No paper found analyses chattering or limit cycles when selection margins fall below u. The two closest are production hypotheses: Trinity §2.3 (oscillation, fixed by tanh soft clamping plus momentum) and Kimi K3 §2.3.3 (γ's trade-off, fixed by the exact quantile solve). Han and Zhong give regret, not oscillation.

*Summary.* Supports: the saturation regime is real enough that two 2026 recipes act on it (FP32 routing, gate scaling), and both recipes that replaced the fixed step did so for the oscillation reason. Contradicts nothing. New vs map: V4's Sqrt(Softplus) change, Motif 3's FP32 rationale, Ling's gate scaling, and the finding that no published analysis of router saturation under Muon or of bang-bang chatter exists — a clean novelty claim for the group-F reading.

---

## Suggested citations (ranked, max 10)

1. **Motif 3, [2608.09119](https://arxiv.org/abs/2608.09119), §4.2–4.3** — earliest MoE layers hardest to balance; FP32 routing; router noise.
2. **Trinity Large, [2602.17004](https://arxiv.org/abs/2602.17004), §2.3, §6** — SMEBU's rule and oscillation hypothesis; six dense leading layers; router collapse under Muon. For H9 and the balancer-rate threat.
3. **Kimi K3, [2607.24653](https://arxiv.org/abs/2607.24653), §2.3.3** — the fixed-step rule and γ's trade-off; QB Eq. 14. For Background's theory of the balancer.
4. **Kim et al., [2608.20061](https://arxiv.org/abs/2608.20061), §3.2, App. A** — μP transfer under Muon and the 0.2·√max(A,B) match. For result 4 and tuning fairness.
5. **Ling 2.0, [2510.22115](https://arxiv.org/abs/2510.22115), §2.2** — the zero-centred bias update, for the common-mode drift.
6. **DeepSeek-V4, [2606.19348](https://arxiv.org/abs/2606.19348), §2.1** — Sigmoid → Sqrt(Softplus), no reason given. For the saturation reading.
7. **Queipo-de-Llano et al., [2510.06477](https://arxiv.org/abs/2510.06477)** — massive activations provably induce a dominant direction; middle-layer caveat. Discussion.
8. **Shared routing geometry, [2609.02404](https://arxiv.org/abs/2609.02404), §3.1** — top-k invariance to a common logit shift; router-weight vs router-input centring.
9. **Małaśnicki et al., [2508.09752](https://arxiv.org/abs/2508.09752), §4.2–4.3** — SP vs μP transfer; non-transfer across granularity.
10. **Dial, Open Athena QB blog, 2026-04-10** — the 0.001 step balancing poorly in the first 5,000 steps; leading dense layers as a balancing aid. Grey literature.

*Not for the thesis:* the `aupai` PR (F3) and the Vlassis quantization study (C3) belong in `context/`.
