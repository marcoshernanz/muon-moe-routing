"""Tests of the routing telemetry: hand-computed values, then a watched training step."""

import json
import math
from pathlib import Path

import pytest
import torch

from tfg.data import VOCAB_SIZE
from tfg.model import FeedForward, LanguageModel
from tfg.optim import GROUPS, build
from tfg.telemetry import (
    GAINS,
    Snapshot,
    Telemetry,
    angle,
    churn,
    common_share,
    copy,
    correlation,
    dispersion,
    expert_update_norms,
    input_similarity,
    kurtosis,
    load_fractions,
    raw_fractions,
    router_entropy,
    row_similarity,
    selection_margins,
    spectral_norms,
    svd_entropy,
    total_norm,
    update_norm,
    usage_perplexity,
)
from tfg.tier import LAPTOP
from tfg.train import train_step

MIXTURES = LAPTOP.num_blocks - LAPTOP.dense_blocks


def model() -> LanguageModel:
    torch.manual_seed(0)
    return LanguageModel(VOCAB_SIZE, LAPTOP)


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(VOCAB_SIZE, (2, LAPTOP.context_len + 1))  # [B, T+1]
    return tokens[:, :-1], tokens[:, 1:]


def readings(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_usage_perplexity_counts_the_experts_in_use() -> None:
    balanced = load_fractions(torch.tensor([3, 3, 3, 3]))
    assert balanced.tolist() == [0.25] * 4
    assert usage_perplexity(balanced) == pytest.approx(4.0)
    assert usage_perplexity(load_fractions(torch.tensor([12, 0, 0, 0]))) == pytest.approx(1.0)
    assert usage_perplexity(load_fractions(torch.zeros(4, dtype=torch.long))) == pytest.approx(1.0)


def test_churn_is_the_share_of_slots_that_changed_expert() -> None:
    half = torch.tensor([0.5, 0.5])
    assert churn(None, half) == 0.0
    assert churn(half, half) == 0.0
    assert churn(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(1.0)
    assert churn(half, torch.tensor([0.75, 0.25])) == pytest.approx(0.25)


def test_input_similarity_is_the_mean_cosine_between_distinct_tokens() -> None:
    inputs = torch.randn(6, 5)  # [N, D]
    units = torch.nn.functional.normalize(inputs, dim=-1)  # [N, D]
    cosines = units @ units.T  # [N, N]
    distinct = (cosines.sum() - cosines.diagonal().sum()) / (6 * 5)
    assert input_similarity(units.sum(0), 6) == pytest.approx(distinct.item(), abs=1e-6)
    same = torch.nn.functional.normalize(torch.ones(4, 3), dim=-1)  # every token alike
    assert input_similarity(same.sum(0), 4) == pytest.approx(1.0)
    opposed = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])  # [N, D]
    assert input_similarity(opposed.sum(0), 2) == pytest.approx(-1.0)


def test_common_share_is_the_part_of_the_update_along_the_mean_input() -> None:
    units = torch.tensor([3.0, 0.0, 0.0, 0.0])  # the inputs' mean direction: the first axis
    along = torch.zeros(2, 4)
    along[:, 0] = torch.tensor([1.0, -2.0])  # every row moves along that direction alone
    assert common_share(along, units) == pytest.approx(1.0)
    across = torch.zeros(2, 4)
    across[0, 1], across[1, 2] = 1.0, 1.0  # the rows move across it
    assert common_share(across, units) == pytest.approx(0.0)
    orthogonal = torch.eye(4)[
        :2
    ]  # two equal singular values: at most one in the square root of two
    assert common_share(orthogonal, units) == pytest.approx(1 / math.sqrt(2))
    assert common_share(torch.zeros(2, 4), units) is None
    assert common_share(along, torch.zeros(4)) is None


def test_raw_load_is_the_top_k_of_the_scores_without_the_bias() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [3.0, 1.0, 2.0, 0.0], [0.0, 3.0, 2.0, 1.0]])
    assert raw_fractions(logits, 2).tolist() == pytest.approx([2 / 6, 2 / 6, 2 / 6, 0.0])
    assert raw_fractions(logits, 1).tolist() == pytest.approx([2 / 3, 1 / 3, 0.0, 0.0])


def test_router_entropy_peaks_when_the_router_is_indifferent() -> None:
    assert router_entropy(torch.zeros(3, 4)) == pytest.approx(math.log(4))
    peaked = router_entropy(torch.tensor([[10.0, -10.0, -10.0, -10.0]]))
    assert 0 < peaked < math.log(4)


def test_selection_margin_compares_the_last_chosen_with_the_first_rejected() -> None:
    logits = torch.linspace(-2, 2, LAPTOP.num_experts)[None]  # [1, E], increasing with the index
    scores = torch.sigmoid(logits[0])  # [E]
    last, first_out = (
        LAPTOP.num_experts - LAPTOP.num_active_experts,
        LAPTOP.num_experts - LAPTOP.num_active_experts - 1,
    )
    unbiased = selection_margins(logits, torch.zeros(LAPTOP.num_experts), LAPTOP.num_active_experts)
    assert unbiased.item() == pytest.approx((scores[last] - scores[first_out]).item())
    bias = torch.zeros(LAPTOP.num_experts)
    bias[0] = 10.0  # the weakest expert is now forced in, pushing the old last choice out
    forced = selection_margins(logits, bias, LAPTOP.num_active_experts)
    assert forced.item() == pytest.approx((scores[last + 1] - scores[last]).item())


def test_update_statistics_on_hand_made_matrices() -> None:
    before = [torch.zeros(2, 2), torch.zeros(2, 2)]
    after = [torch.tensor([[3.0, 0.0], [0.0, 0.0]]), torch.tensor([[0.0, 4.0], [0.0, 0.0]])]
    assert update_norm(before, after).item() == pytest.approx(5.0)
    assert total_norm(after).item() == pytest.approx(5.0)

    counts = torch.tensor([1, 2, 3])
    assert correlation(torch.tensor([2.0, 4.0, 6.0]), counts) == pytest.approx(1.0)
    assert correlation(torch.tensor([3.0, 2.0, 1.0]), counts) == pytest.approx(-1.0)
    assert correlation(torch.ones(3), counts) is None

    assert dispersion(torch.ones(3)) == 0.0
    assert dispersion(torch.tensor([0.0, 2.0])) == pytest.approx(math.sqrt(2))
    assert dispersion(torch.zeros(3)) is None


def test_spectral_norm_is_the_largest_singular_value_and_never_overshoots() -> None:
    diagonal = torch.zeros(2, 3, 2)  # [E, M, N]
    diagonal[0, 0, 0], diagonal[0, 1, 1] = 3.0, 4.0  # singular values 4 and 3; the second is zero
    assert spectral_norms(diagonal).tolist() == pytest.approx([4.0, 0.0])
    generator = torch.Generator().manual_seed(0)
    u, v = torch.randn(4, 5, 1, generator=generator), torch.randn(4, 1, 3, generator=generator)
    rank_one = u @ v  # [E, M, N]: the one singular value is the product of the norms
    expected = (u.flatten(1).norm(dim=1) * v.flatten(1).norm(dim=1)).tolist()
    assert spectral_norms(rank_one).tolist() == pytest.approx(expected)
    matrices = torch.randn(8, 64, 32, generator=generator)  # [E, M, N]
    # pylint: disable-next=not-callable
    exact = torch.linalg.matrix_norm(matrices, ord=2)  # [E]
    estimate = spectral_norms(matrices)
    assert torch.allclose(estimate, exact, rtol=1e-3)
    assert (estimate <= exact * (1 + 1e-6)).all()  # the power method approaches from below
    assert torch.equal(spectral_norms(matrices), estimate)  # a fixed start: deterministic


def test_expert_update_norms_take_an_experts_matrices_as_one_block_diagonal_matrix() -> None:
    experts = [FeedForward(4, 3), FeedForward(4, 3)]
    before = [[torch.zeros_like(p) for p in expert.parameters()] for expert in experts]
    with torch.no_grad():
        for expert in experts:
            for p in expert.parameters():
                p.zero_()
        experts[0].gate_proj.weight[0, 0] = 3.0  # one matrix with singular values 3 and 4
        experts[0].gate_proj.weight[1, 1] = 4.0
        experts[1].gate_proj.weight[0, 0] = 1.0  # two matrices, one singular value each
        experts[1].up_proj.weight[0, 0] = 2.0
    frobenius, spectral = expert_update_norms(before, experts)
    assert frobenius.tolist() == pytest.approx([5.0, math.sqrt(5)])
    assert spectral.tolist() == pytest.approx([4.0, 2.0])  # the largest across the matrices


def test_expert_update_norms_of_several_mixtures_at_once_are_each_mixtures_own() -> None:
    generator = torch.Generator().manual_seed(0)
    mixtures = [[FeedForward(8, 6) for _ in range(3)] for _ in range(2)]  # two mixtures' experts
    befores = [
        [[torch.randn(p.shape, generator=generator) for p in e.parameters()] for e in m]
        for m in mixtures
    ]
    together = expert_update_norms(
        [b for m in befores for b in m], [e for m in mixtures for e in m]
    )
    alone = [expert_update_norms(b, m) for b, m in zip(befores, mixtures)]
    for stacked, own in zip(together, zip(*alone)):
        assert torch.allclose(stacked, torch.cat(own), rtol=1e-6)


def test_row_similarity_is_the_mean_cosine_between_distinct_rows() -> None:
    assert row_similarity(torch.ones(4, 6)) == pytest.approx(1.0)  # all the same way
    assert row_similarity(torch.eye(4)) == pytest.approx(0.0)  # orthogonal
    assert row_similarity(torch.tensor([[1.0, 0.0], [-2.0, 0.0]])) == pytest.approx(-1.0)
    e1, e2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert row_similarity(torch.stack([e1, e1, e2])) == pytest.approx(1 / 3)  # one pair of three


def test_the_angle_of_a_step_holds_even_when_the_step_is_tiny() -> None:
    one = torch.tensor([[1.0, 0.0]])
    assert angle([one], [torch.tensor([[1.0, 1.0]])]) == pytest.approx(math.pi / 4)
    assert angle([one], [2 * one]) == 0.0  # a rescaling turns nothing
    assert angle([one, one], [one, -one]) == pytest.approx(math.pi / 2)  # the list as one vector
    tiny = angle([one], [torch.tensor([[1.0, 1e-6]])])  # the cosine would round to one
    assert tiny == pytest.approx(1e-6, rel=1e-6)
    assert angle([torch.zeros(2)], [one[0]]) is None


def test_svd_entropy_is_the_log_rank_of_an_isotropic_matrix() -> None:
    assert svd_entropy(torch.eye(4)) == pytest.approx(math.log(4))
    assert svd_entropy(torch.ones(4, 6)) == pytest.approx(0.0, abs=1e-4)  # rank one
    assert svd_entropy(torch.zeros(3, 3)) == 0.0
    torch.manual_seed(0)
    assert 0 < svd_entropy(torch.randn(4, 16)) < math.log(4)


def test_kurtosis_is_excess_and_grows_with_the_tails() -> None:
    assert kurtosis(torch.tensor([-1.0, 1.0, -1.0, 1.0])) == pytest.approx(-2.0)  # two points
    assert kurtosis(torch.tensor([[-1.0, 0.0, 1.0]])) == pytest.approx(-1.5)  # any shape
    torch.manual_seed(0)
    normal = torch.randn(200_000)
    assert kurtosis(normal) == pytest.approx(0.0, abs=0.05)
    heavy = normal.clone()
    heavy[0] = 100.0  # one outlier
    heavy_tailed = kurtosis(heavy)
    assert heavy_tailed is not None and heavy_tailed > 10
    assert kurtosis(torch.ones(5)) is None and kurtosis(torch.zeros(0)) is None


def test_reading_follows_the_update_of_each_expert_and_router_row(tmp_path: Path) -> None:
    telemetry = Telemetry(model(), tmp_path / "routing.jsonl")
    mixture = telemetry.mixtures[0]
    before = Snapshot.of(mixture, copy(mixture.parameters()))
    with torch.no_grad():
        for index, expert in enumerate(mixture.experts):
            next(expert.parameters()).diagonal()[: index + 1] += 1.0  # e + 1 singular values of one
        mixture.router.weight[0] += 1.0  # only the first router row moves
        mixture.expert_load.copy_(torch.arange(LAPTOP.num_experts))  # more tokens, higher rank
    logits = torch.randn(5, LAPTOP.num_experts)  # [N, E]
    mixture.router_logits = logits
    mixture.router_input_units = torch.zeros(LAPTOP.d_model)  # five inputs that cancel out
    mixture.router_input_units[0] = 1.0  # all but one of them: a sum of norm one
    signs = torch.tensor([-1.0, 1.0]).repeat(LAPTOP.d_model // 2)  # [D]
    mixture.expert_outputs = signs.expand(120, -1).clone()  # every expert's rows: two points
    mixture.expert_outputs[2, 0] = 50.0  # in the second row of expert 2, an outlier
    telemetry.blocks[0].residual = torch.full((2, 3, LAPTOP.d_model), -7.0)  # [B, T, D]
    telemetry.inputs[0].residual = torch.zeros(2, 3, LAPTOP.d_model)  # [B, T, D]: no logit
    reading = telemetry.reading(0, before, expert_update_norms(before.experts, mixture.experts))
    # The update of an orthogonalized momentum of rank e + 1: the norms disagree, and the readings without a suffix take the Frobenius one.
    moved = torch.arange(1, LAPTOP.num_experts + 1).sqrt()  # [E]
    assert reading["expert_update_norm"] == pytest.approx(moved.tolist())
    assert reading["expert_update_spectral_norm"] == pytest.approx([1.0] * LAPTOP.num_experts)
    follows = torch.corrcoef(torch.stack([moved, torch.arange(LAPTOP.num_experts).float()]))[0, 1]
    assert reading["update_load_correlation"] == pytest.approx(follows.item())
    assert "update_load_correlation_spectral" in reading  # of sizes that differ by rounding alone
    assert reading["expert_update_dispersion"] == pytest.approx((moved.std() / moved.mean()).item())
    assert reading["expert_update_dispersion_spectral"] == pytest.approx(0.0, abs=1e-6)
    assert reading["router_row_dispersion"] == pytest.approx(math.sqrt(LAPTOP.num_experts))
    # Row 0 moved by one in every coordinate, the inputs' mean direction is the first axis.
    assert reading["router_update_common_share"] == pytest.approx(1 / math.sqrt(LAPTOP.d_model))
    assert reading["load"] == pytest.approx((torch.arange(LAPTOP.num_experts) / 120).tolist())
    assert reading["bias_span"] == 0.0
    assert reading["residual_max"] == 7.0
    assert reading["attention_logit_max"] == 0.0  # a zero input gives zero queries and keys
    assert reading["router_row_similarity"] == pytest.approx(row_similarity(mixture.router.weight))
    assert reading["router_logit_max"] == pytest.approx(logits.abs().max().item())
    assert reading["router_logit_norm"] == pytest.approx(logits.norm(dim=-1).mean().item())
    assert reading["router_input_similarity"] == pytest.approx((1 - 5) / (5 * 4))
    mean_scores = torch.sigmoid(logits).mean(0)  # [E]
    assert reading["mean_score_span"] == pytest.approx(
        (mean_scores.max() - mean_scores.min()).item()
    )
    raw = raw_fractions(logits, LAPTOP.num_active_experts)  # [E]
    assert reading["raw_usage_perplexity"] == pytest.approx(usage_perplexity(raw))
    assert reading["raw_max_load_ratio"] == pytest.approx(raw.max().item() * LAPTOP.num_experts)
    svd = reading["router_svd_entropy"]
    assert isinstance(svd, float) and 0 < svd <= math.log(LAPTOP.num_experts)
    assert reading["expert_output_max"] == 50.0
    kurtoses = reading["expert_kurtosis"]
    assert isinstance(kurtoses, list) and len(kurtoses) == LAPTOP.num_experts
    assert kurtoses[0] is None  # no tokens
    assert kurtoses[1] == pytest.approx(-2.0) and kurtoses[3:] == pytest.approx([-2.0] * 13)
    assert kurtoses[2] > 0 and reading["expert_kurtosis_max"] == kurtoses[2]


def test_watch_writes_one_line_per_step_with_every_mixture(tmp_path: Path) -> None:
    net = model()
    optimizers = build(net, "adamw", adamw_lr=1e-3, muon_lr=1e-3)
    telemetry = Telemetry(net, tmp_path / "routing.jsonl")
    for step in (1, 2):
        with telemetry.watch(step):
            train_step(net, optimizers, batch(), factor=1.0)
    lines = readings(tmp_path / "routing.jsonl")
    assert [line["step"] for line in lines] == [1, 2]
    for line, mixture in zip(lines, net.decoder.mixtures()):
        assert isinstance(line["mixtures"], list) and len(line["mixtures"]) == MIXTURES
    first, second = (line["mixtures"] for line in lines)
    assert isinstance(first, list) and isinstance(second, list)
    for reading, mixture in zip(second, net.decoder.mixtures()):
        assert reading["bias"] == pytest.approx(mixture.router_bias.tolist())
        assert reading["bias_span"] == pytest.approx(
            (mixture.router_bias.max() - mixture.router_bias.min()).item()
        )
        assert sum(reading["load"]) == pytest.approx(1.0)
        assert 1.0 <= reading["usage_perplexity"] <= LAPTOP.num_experts
        assert 1.0 <= reading["raw_usage_perplexity"] <= LAPTOP.num_experts
        assert (
            1.0 <= reading["raw_max_load_ratio"] <= LAPTOP.num_experts / LAPTOP.num_active_experts
        )
        assert -1.0 <= reading["router_input_similarity"] <= 1.0
        assert 0.0 <= reading["mean_score_span"] < 1.0
        assert 0.0 < reading["router_update_common_share"] <= 1.0
        assert reading["margin_low"] <= reading["margin_mean"]
        frobenius, spectral = reading["expert_update_norm"], reading["expert_update_spectral_norm"]
        assert isinstance(frobenius, list) and isinstance(spectral, list)
        assert len(frobenius) == len(spectral) == LAPTOP.num_experts
        assert min(spectral) > 0  # weight decay moves even idle experts
        assert all(s <= f for s, f in zip(spectral, frobenius))  # the spectral norm is the smaller
        assert reading["residual_max"] > 0 and reading["router_logit_max"] > 0
        assert reading["expert_output_max"] > 0 and math.isfinite(reading["attention_logit_max"])
        assert -1.0 <= reading["router_row_similarity"] <= 1.0
        kurtoses, load = reading["expert_kurtosis"], reading["load"]
        assert isinstance(kurtoses, list) and isinstance(load, list)
        assert [k is None for k in kurtoses] == [share == 0 for share in load]
        assert reading["expert_kurtosis_max"] == max(k for k in kurtoses if k is not None)
    assert all(reading["load_churn"] == 0.0 for reading in first)
    assert all(reading["load_churn"] >= 0.0 for reading in second)
    groups = lines[1]["groups"]
    assert isinstance(groups, dict) and sorted(groups) == sorted((*GROUPS, GAINS))  # all here
    for reading in groups.values():
        assert reading["weight_norm"] > 0 and 0 < reading["angular_step"] < 1
        assert 0 < reading["update_norm"] < reading["weight_norm"]
    router_norm = math.hypot(*(m.router.weight.norm().item() for m in net.decoder.mixtures()))
    assert groups["router"]["weight_norm"] == pytest.approx(router_norm)
    gains = [p.detach() for p in net.parameters() if p.ndim < 2]  # the norm gains, on their own too
    assert groups[GAINS]["weight_norm"] == pytest.approx(total_norm(gains).item())
    assert groups[GAINS]["weight_norm"] < groups["tables_and_norms"]["weight_norm"]
    assert groups[GAINS]["update_norm"] < groups["tables_and_norms"]["update_norm"]
    kept = [
        (m.router_logits, m.expert_outputs, m.router_input_units) for m in net.decoder.mixtures()
    ]
    residuals = [block.residual for block in net.decoder.mixture_blocks()]
    net.eval()
    net(batch()[0])  # an evaluation forward leaves the training step's tensors in place
    assert all(
        a is m.router_logits and b is m.expert_outputs and c is m.router_input_units
        for (a, b, c), m in zip(kept, net.decoder.mixtures())
    )
    assert all(r is b.residual for r, b in zip(residuals, net.decoder.mixture_blocks()))


def test_a_snapshot_without_a_write_leaves_no_line(tmp_path: Path) -> None:
    net = model()
    optimizers = build(net, "adamw", adamw_lr=1e-3, muon_lr=1e-3)
    telemetry = Telemetry(net, tmp_path / "routing.jsonl")
    telemetry.snapshot()  # what the trainer does with a step that diverged: no reading of it
    train_step(net, optimizers, batch(), factor=1.0)
    before = telemetry.snapshot()
    train_step(net, optimizers, batch(), factor=1.0)
    telemetry.write(2, before)
    lines = readings(tmp_path / "routing.jsonl")
    assert [line["step"] for line in lines] == [2]
    first = lines[0]["mixtures"]
    assert isinstance(first, list) and first[0]["load_churn"] == 0.0  # nothing to compare with


def test_resume_continues_the_file_and_the_churn(tmp_path: Path) -> None:
    net = model()
    optimizers = build(net, "adamw", adamw_lr=1e-3, muon_lr=1e-3)
    path = tmp_path / "routing.jsonl"
    telemetry = Telemetry(net, path)
    for step in (1, 2):
        with telemetry.watch(step):
            train_step(net, optimizers, batch(), factor=1.0)
    continued = Telemetry(net, path, resume=True)
    assert all(
        a is not None and b is not None and torch.equal(a, b)
        for a, b in zip(continued.previous, telemetry.previous)
    )
    with continued.watch(3):
        train_step(net, optimizers, batch(), factor=1.0)
    assert [line["step"] for line in readings(path)] == [1, 2, 3]
    empty = Telemetry(net, tmp_path / "other.jsonl", resume=True)  # nothing to continue from
    assert empty.previous == [None] * MIXTURES


def test_watch_writes_nothing_when_the_step_fails(tmp_path: Path) -> None:
    net = model()
    telemetry = Telemetry(net, tmp_path / "routing.jsonl")
    with pytest.raises(RuntimeError, match="boom"), telemetry.watch(1):
        net(batch()[0])
        raise RuntimeError("boom")
    assert (tmp_path / "routing.jsonl").read_text(encoding="utf-8") == ""


def test_watch_needs_a_forward_inside_it(tmp_path: Path) -> None:
    telemetry = Telemetry(model(), tmp_path / "routing.jsonl")
    with pytest.raises(RuntimeError, match="not run"), telemetry.watch(1):
        pass
