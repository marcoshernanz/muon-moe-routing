"""Tests of the model at its module constants; every test runs on a laptop in seconds."""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from tfg.data import VOCAB_SIZE
from tfg.model import (
    Decoder,
    FeedForward,
    LanguageModel,
    MixtureOfExperts,
    apply_rope,
    rope_tables,
    sequence_balance,
)
from tfg.tier import CONFIRMATION, KAGGLE_T4, LAPTOP, MECHANISM, TIERS, Tier

VOCAB = 256
BATCH, SEQ = 2, LAPTOP.context_len


def generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def model() -> LanguageModel:
    torch.manual_seed(0)
    return LanguageModel(VOCAB, LAPTOP)


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    seq = torch.randint(0, VOCAB, (BATCH, SEQ + 1), generator=generator(1))  # [B, T+1]
    return seq[:, :-1], seq[:, 1:]


def hand_count(tier: Tier, vocab: int) -> tuple[int, int]:
    """The total and active parameter counts derived on paper from a tier."""
    attention = 5 * tier.d_model * tier.d_model  # q, k, v, o, and the output gate
    norms = 2 * tier.d_model + 2 * tier.d_head  # two block norms and the q/k gains
    dense = attention + norms + 3 * tier.d_model * tier.d_ffn
    expert = 3 * tier.d_model * tier.d_expert
    mixture = (
        attention
        + norms
        + tier.num_experts * tier.d_model  # router
        + 3 * tier.d_model * tier.d_shared  # shared expert
    )
    mixtures = tier.num_blocks - tier.dense_blocks
    shared = (
        2 * vocab * tier.d_model + tier.d_model + tier.dense_blocks * dense + mixtures * mixture
    )
    total = shared + mixtures * tier.num_experts * expert
    active = shared + mixtures * tier.num_active_experts * expert
    return total, active


@pytest.mark.parametrize("tier", list(TIERS.values()), ids=list(TIERS))
def test_parameter_count_matches_the_hand_formula(tier: Tier) -> None:
    total, _ = hand_count(tier, VOCAB)
    assert sum(p.numel() for p in LanguageModel(VOCAB, tier).parameters()) == total


def test_the_mechanism_tier_has_the_plan_size() -> None:
    """The plan quotes 55M active parameters without the tables, 72M with them, 303M in total."""
    total, active = hand_count(MECHANISM, VOCAB_SIZE)
    embeddings = 2 * VOCAB_SIZE * MECHANISM.d_model
    assert 54e6 < active - embeddings < 56e6
    assert 71e6 < active < 73e6
    assert 300e6 < total < 306e6
    assert MECHANISM.tokens_per_step == 64 * 1024


def test_the_confirmation_tier_has_the_plan_size_and_the_mechanism_tiers_routing() -> None:
    """The plan quotes 108M active parameters without the tables, 133M with them, 496M in total; the routing problem is the mechanism tier's, only the width, depth, and expert width move."""
    total, active = hand_count(CONFIRMATION, VOCAB_SIZE)
    embeddings = 2 * VOCAB_SIZE * CONFIRMATION.d_model
    assert 107e6 < active - embeddings < 109e6
    assert 132e6 < active < 134e6
    assert 493e6 < total < 499e6
    fixed = ("num_experts", "num_active_experts", "dense_blocks", "context_len", "batch_size")
    assert all(getattr(CONFIRMATION, name) == getattr(MECHANISM, name) for name in fixed)
    assert (
        CONFIRMATION.d_model > MECHANISM.d_model and CONFIRMATION.num_blocks > MECHANISM.num_blocks
    )
    assert CONFIRMATION.d_expert < MECHANISM.d_expert
    assert CONFIRMATION.d_head == MECHANISM.d_head == 64


def test_the_kaggle_tier_trains_a_step_on_this_machine() -> None:
    torch.manual_seed(0)
    m = LanguageModel(VOCAB, KAGGLE_T4).train()
    tokens = torch.randint(0, VOCAB, (1, KAGGLE_T4.context_len + 1), generator=generator(12))
    loss = F.cross_entropy(m(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    assert abs(loss.item() - math.log(VOCAB)) < 0.5
    assert all(p.grad is not None for p in m.parameters())
    assert len(m.decoder.mixtures()) == KAGGLE_T4.num_blocks - KAGGLE_T4.dense_blocks


def test_the_slice_fallback_matches_the_grouped_experts() -> None:
    """GPUs older than the grouped kernel take this path; on CPU both run, so compare them."""
    mixture = MixtureOfExperts(LAPTOP)
    counts = torch.tensor([5, 0, 3, 8, 0, 1, 2, 4, 0, 0, 6, 7, 1, 2, 3, 6])  # [E], one per expert
    inputs = torch.randn(int(counts.sum()), LAPTOP.d_model, generator=generator(13))  # [M, D]
    torch.testing.assert_close(
        mixture.run_experts_by_slice(inputs, counts), mixture.run_experts(inputs, counts)
    )


def test_first_blocks_are_dense_and_the_rest_are_mixtures() -> None:
    blocks = model().decoder.blocks
    for i, block in enumerate(blocks):
        expected = FeedForward if i < LAPTOP.dense_blocks else MixtureOfExperts
        assert isinstance(block.ffn, expected)


def test_initial_loss_is_near_uniform() -> None:
    x, y = batch()
    logits = model()(x)  # [B, T, V]
    assert logits.shape == (BATCH, SEQ, VOCAB)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    assert abs(loss.item() - math.log(VOCAB)) < 0.5


def test_every_parameter_gets_a_finite_gradient() -> None:
    m = model()
    x, y = batch()
    loss = torch.nn.functional.cross_entropy(m(x).flatten(0, 1), y.flatten())
    loss.backward()
    for name, param in m.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


MIXTURE_BLOCKS = LAPTOP.num_blocks - LAPTOP.dense_blocks


@pytest.mark.parametrize("recompute", [1, MIXTURE_BLOCKS])
def test_recompute_gives_the_same_loss_gradients_and_load_bit_for_bit(recompute: int) -> None:
    inputs, targets = batch()
    torch.manual_seed(0)
    recomputed = LanguageModel(VOCAB, LAPTOP, recompute=recompute)
    kept = model()
    for net in (kept, recomputed):
        F.cross_entropy(net(inputs).flatten(0, 1), targets.flatten()).backward()
    for mine, yours in zip(recomputed.parameters(), kept.parameters()):
        assert mine.grad is not None and yours.grad is not None
        assert torch.equal(mine.grad, yours.grad)
    for mine, yours in zip(recomputed.decoder.mixtures(), kept.decoder.mixtures()):
        assert torch.equal(mine.expert_load, yours.expert_load)  # written twice, counted once
        assert mine.expert_outputs is not None and yours.expert_outputs is not None
        assert torch.equal(mine.expert_outputs, yours.expert_outputs)
    recomputed.eval()  # nothing to recompute without a backward pass
    with torch.no_grad():
        assert torch.equal(recomputed(inputs), kept.eval()(inputs))


def test_recompute_takes_the_first_mixture_blocks_and_no_more_than_there_are() -> None:
    assert not Decoder(LAPTOP).recomputed
    assert list(Decoder(LAPTOP, recompute=2).recomputed) == [1, 2]  # block 0 is dense
    for count in (-1, MIXTURE_BLOCKS + 1):
        with pytest.raises(ValueError, match="mixture blocks"):
            Decoder(LAPTOP, recompute=count)


def test_changing_a_later_token_leaves_earlier_logits_unchanged() -> None:
    m = model().eval()
    x, _ = batch()
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % VOCAB
    with torch.no_grad():
        a, b = m(x), m(x2)
    torch.testing.assert_close(a[:, :-1], b[:, :-1])
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_the_attention_logit_max_is_the_largest_causal_score() -> None:
    attn = model().decoder.mixture_blocks()[0].attn
    x = torch.randn(BATCH, SEQ, LAPTOP.d_model, generator=generator(3))  # [B, T, D]
    q, k = attn.queries_keys(x)  # [B, H, T, Dh] each
    scores = q @ k.transpose(-1, -2) / math.sqrt(LAPTOP.d_head)  # [B, H, T, T]
    causal = torch.ones(SEQ, SEQ, dtype=torch.bool).tril()  # [T, T]
    assert attn.logit_max(x) == pytest.approx(scores[:, :, causal].max().item())
    assert attn.logit_max(x) <= scores.max().item()  # the future pairs are left out
    assert attn.logit_max(x[:, : SEQ // 2]) <= attn.logit_max(x)  # a prefix: fewer pairs


def test_mixture_inputs_are_the_blocks_before_the_mixture_blocks() -> None:
    decoder = model().decoder
    blocks = list(decoder.blocks)
    assert decoder.mixture_inputs() == blocks[LAPTOP.dense_blocks - 1 : -1]
    assert decoder.mixture_blocks() == blocks[LAPTOP.dense_blocks :]
    every_block_a_mixture = LanguageModel(VOCAB, dataclasses.replace(LAPTOP, dense_blocks=0))
    with pytest.raises(ValueError, match="block 0 must be dense"):
        every_block_a_mixture.decoder.mixture_inputs()


def test_rope_preserves_norms_and_leaves_position_zero_alone() -> None:
    cos, sin = rope_tables(LAPTOP.context_len, LAPTOP.d_head)
    x = torch.randn(1, 1, SEQ, LAPTOP.d_head, generator=generator(2))  # [B, H, T, Dh]
    rotated = apply_rope(x, cos, sin)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])


def test_every_token_is_sent_to_exactly_k_experts() -> None:
    m = model().train()
    x, _ = batch()
    m(x)
    for mixture in m.decoder.mixtures():
        assert int(mixture.expert_load.sum()) == BATCH * SEQ * LAPTOP.num_active_experts


def test_the_load_counts_the_slots_routed_to_each_expert() -> None:
    torch.manual_seed(8)
    mixture = MixtureOfExperts(LAPTOP).train()
    mixture.router_bias.uniform_(-0.1, 0.1)  # so the bias takes part in the selection
    x = torch.randn(BATCH, SEQ, LAPTOP.d_model, generator=generator(9))  # [B, T, D]
    mixture(x)
    _, chosen, _ = mixture.route(x.reshape(-1, LAPTOP.d_model))  # [N, K]
    expected = torch.bincount(chosen.flatten(), minlength=LAPTOP.num_experts)  # [E]
    assert torch.equal(mixture.expert_load, expected)


def test_eval_mode_does_not_touch_the_recorded_load() -> None:
    m = model().eval()
    x, _ = batch()
    m(x)
    for mixture in m.decoder.mixtures():
        assert int(mixture.expert_load.sum()) == 0


def test_bias_steers_selection_but_not_the_mixing_weights() -> None:
    mixture = MixtureOfExperts(LAPTOP)
    tokens = torch.randn(BATCH * SEQ, LAPTOP.d_model, generator=generator(3))  # [N, D]
    scores = torch.sigmoid(mixture.router(tokens))  # [N, E]
    unbiased = scores.topk(LAPTOP.num_active_experts, dim=-1).indices  # [N, K]
    assert not (unbiased == 0).any(dim=-1).all()

    mixture.router_bias[0] = 10.0
    chosen = (
        (scores + mixture.router_bias).topk(LAPTOP.num_active_experts, dim=-1).indices
    )  # [N, K]
    assert (chosen == 0).any(dim=-1).all()
    weights = scores.gather(-1, chosen)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH * SEQ))


def test_update_biases_moves_against_the_load() -> None:
    mixture = MixtureOfExperts(LAPTOP)
    load = torch.full((LAPTOP.num_experts,), 10, dtype=torch.long)
    load[0], load[1] = 30, 0  # one over-used expert, one starved
    mixture.expert_load.copy_(load)
    mixture.update_biases()
    assert mixture.router_bias[0] < 0 < mixture.router_bias[1]
    assert torch.all(mixture.router_bias[2:] > 0)  # everyone under the mean moves up


def reference_forward(mixture: MixtureOfExperts, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
    """The mixture as a loop over experts, each gathering its own tokens: the dispatch the grouped one replaced, kept here as the reference it must match."""
    batch_size, seq_len, _ = x.size()
    tokens = x.reshape(-1, LAPTOP.d_model)  # [N, D]
    scores = torch.sigmoid(mixture.router(tokens.float()))  # [N, E]
    chosen = (
        (scores + mixture.router_bias).topk(LAPTOP.num_active_experts, dim=-1).indices
    )  # [N, K]
    weights = scores.gather(-1, chosen)  # [N, K]
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(tokens.dtype)  # [N, K]
    routed = torch.zeros_like(tokens)  # [N, D]
    for index, expert in enumerate(mixture.experts):
        token_index, slot = (chosen == index).nonzero(as_tuple=True)  # [n], [n]
        routed.index_add_(
            0, token_index, weights[token_index, slot, None] * expert(tokens[token_index])
        )
    out = routed + mixture.shared_expert(tokens)  # [N, D]
    return out.reshape(batch_size, seq_len, LAPTOP.d_model)  # [B, T, D]


def test_grouped_dispatch_matches_the_per_expert_loop_with_gradients() -> None:
    """On CUDA the grouped experts run in bf16 while the loop runs in float32, hence the slack."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atol, rtol = (2e-2, 2e-2) if device.type == "cuda" else (1e-5, 1e-4)
    torch.manual_seed(8)
    mixture = MixtureOfExperts(LAPTOP).to(device).train()
    mixture.router_bias.uniform_(-0.1, 0.1)  # so the bias takes part in the selection
    x = torch.randn(BATCH, SEQ, LAPTOP.d_model, generator=generator(9)).to(device)  # [B, T, D]
    grad = torch.randn(BATCH, SEQ, LAPTOP.d_model, generator=generator(10)).to(device)  # [B, T, D]

    out = mixture(x)  # [B, T, D]
    load = mixture.expert_load.clone()  # [E]
    out.backward(grad)
    grads = [p.grad.clone() for p in mixture.parameters() if p.grad is not None]
    assert len(grads) == 1 + 3 * LAPTOP.num_experts + 3  # router, experts, shared expert
    mixture.zero_grad()

    reference = reference_forward(mixture, x)  # [B, T, D]
    reference.backward(grad)
    torch.testing.assert_close(out, reference, atol=atol, rtol=rtol)
    for got, want in zip(grads, [p.grad for p in mixture.parameters() if p.grad is not None]):
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
    assert torch.equal(mixture.expert_load, load)  # the loads are counted the same way
    assert int(load.sum()) == BATCH * SEQ * LAPTOP.num_active_experts


def test_dispatch_handles_experts_that_receive_no_tokens() -> None:
    mixture = MixtureOfExperts(LAPTOP)
    mixture.router_bias[: LAPTOP.num_active_experts] = 10.0  # the first K experts take every token
    x = torch.randn(1, SEQ, LAPTOP.d_model, generator=generator(11))  # [B, T, D]
    out = mixture.train()(x)  # [B, T, D]
    assert torch.equal(
        mixture.expert_load[LAPTOP.num_active_experts :],
        torch.zeros(LAPTOP.num_experts - LAPTOP.num_active_experts, dtype=torch.long),
    )
    torch.testing.assert_close(out, reference_forward(mixture, x), atol=1e-5, rtol=1e-4)
    out.sum().backward()  # the idle experts get a zero gradient, not a missing one
    assert all(p.grad is not None for p in mixture.parameters())


def test_dispatch_equals_a_dense_computation() -> None:
    mixture = MixtureOfExperts(LAPTOP)
    x = torch.randn(1, SEQ, LAPTOP.d_model, generator=generator(4))  # [B, T, D]
    out = mixture(x).view(-1, LAPTOP.d_model)  # [N, D]

    tokens = x.view(-1, LAPTOP.d_model)  # [N, D]
    scores = torch.sigmoid(mixture.router(tokens))  # [N, E]
    chosen = (
        (scores + mixture.router_bias).topk(LAPTOP.num_active_experts, dim=-1).indices
    )  # [N, K]
    weights = scores.gather(-1, chosen)  # [N, K]
    weights = weights / weights.sum(dim=-1, keepdim=True)  # [N, K]
    dense = mixture.shared_expert(tokens)  # [N, D]
    for index, expert in enumerate(mixture.experts):
        weight = (weights * (chosen == index)).sum(dim=-1, keepdim=True)  # [N, 1]
        dense = dense + weight * expert(tokens)  # [N, D]
    torch.testing.assert_close(out, dense, atol=1e-5, rtol=1e-4)


def test_routing_under_autocast_matches_the_float32_routing() -> None:
    """The router scores in float32 even under bf16 autocast, so the selection does not change."""
    torch.manual_seed(6)
    mixture = MixtureOfExperts(LAPTOP).train()
    tokens = torch.randn(1, SEQ, LAPTOP.d_model, generator=generator(5))  # [1, T, D]
    mixture.router_bias.uniform_(-0.01, 0.01, generator=generator(7))  # bias steps of 0.001 exist

    mixture(tokens)
    exact_load = mixture.expert_load.clone()  # [E]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = mixture(tokens)  # [1, T, D]
    assert out.dtype == tokens.dtype  # the residual stream keeps the input dtype
    assert torch.equal(mixture.expert_load, exact_load)

    # Why it matters: bf16 cannot represent a 0.001 step on a score near 0.5 at all.
    half = torch.tensor(0.5, dtype=torch.bfloat16)
    assert torch.equal(half + 0.001, half)
    assert not torch.equal(half.float() + 0.001, half.float())


def test_the_compiled_model_matches_the_eager_one_in_a_single_graph() -> None:
    """One graph for the whole model, and the same outputs and gradients as the eager forward.

    On a CPU the compiler runs the experts in bf16, so the match is to bf16 precision.
    """
    torch.manual_seed(0)
    net = LanguageModel(VOCAB_SIZE, LAPTOP)
    tokens = torch.randint(0, VOCAB_SIZE, (2, LAPTOP.context_len))  # [B, T]

    eager = net(tokens)  # [B, T, V]
    eager.square().mean().backward()
    eager_grads = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    net.zero_grad()
    compiled = torch.compile(net, fullgraph=True)(tokens)  # a break would raise [B, T, V]
    compiled.square().mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    torch.testing.assert_close(compiled, eager, atol=2e-2, rtol=2e-2)
    assert len(grads) == len(eager_grads)
    for grad, eager_grad in zip(grads, eager_grads):
        torch.testing.assert_close(grad, eager_grad, atol=2e-2, rtol=2e-2)


def test_the_compiled_model_stays_a_single_graph_with_recompute() -> None:
    """Recomputed blocks store tensors for the telemetry, which would split the graph and leave them uncompiled: one graph still, with one block recomputed and the others kept, the stores and the load there, and the eager forward's outputs and gradients."""
    torch.manual_seed(0)
    net = LanguageModel(VOCAB_SIZE, LAPTOP, recompute=1)
    tokens = torch.randint(0, VOCAB_SIZE, (2, LAPTOP.context_len))  # [B, T]

    eager = net(tokens)  # [B, T, V]
    eager.square().mean().backward()
    eager_grads = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    eager_loads = [mixture.expert_load.clone() for mixture in net.decoder.mixtures()]
    net.zero_grad()
    for mixture in net.decoder.mixtures():
        mixture.expert_load.zero_()
        mixture.expert_outputs = None
    compiled = torch.compile(net, fullgraph=True)(tokens)  # a break would raise [B, T, V]
    compiled.square().mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    torch.testing.assert_close(compiled, eager, atol=2e-2, rtol=2e-2)
    assert len(grads) == len(eager_grads)
    for grad, eager_grad in zip(grads, eager_grads):
        torch.testing.assert_close(grad, eager_grad, atol=2e-2, rtol=2e-2)
    for mixture, eager_load in zip(net.decoder.mixtures(), eager_loads):
        assert mixture.expert_outputs is not None
        moved = (mixture.expert_load - eager_load).abs().sum() / eager_load.sum()
        assert eager_load.sum() > 0 and moved < 0.02  # counted once; rounding may move a near-tie


def test_route_returns_the_selection_and_the_normalized_weights() -> None:
    mixture = MixtureOfExperts(LAPTOP)
    mixture.router_bias.uniform_(-0.1, 0.1, generator=generator(0))
    tokens = torch.randn(BATCH * SEQ, LAPTOP.d_model, generator=generator(1))  # [N, D]
    logits, chosen, weights = mixture.route(tokens)
    assert logits.shape == (BATCH * SEQ, LAPTOP.num_experts)
    assert chosen.shape == weights.shape == (BATCH * SEQ, LAPTOP.num_active_experts)
    scores = torch.sigmoid(logits)  # [N, E]
    selection = (scores + mixture.router_bias).topk(LAPTOP.num_active_experts, dim=-1).indices
    assert torch.equal(chosen, selection)
    taken = scores.gather(-1, chosen)  # [N, K]
    assert torch.allclose(weights, taken / taken.sum(dim=-1, keepdim=True))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH * SEQ))


def test_the_sequence_balance_is_one_when_balanced_and_grows_with_imbalance() -> None:
    """DeepSeek-V3's equations on a small case: two sequences of four tokens, four experts, two per token, the top-k taken on the scores alone whatever the bias."""
    pairs = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )  # experts 0 and 1, or 2 and 3
    even = pairs.repeat(2, 1)  # every expert in the top two of two tokens of four [T, E]
    noise = 0.1 * torch.rand(2, 4, 4, generator=generator(0))  # breaks nothing: 4 against 0.1
    logits = 4 * torch.stack([even, even]) - 2 + noise  # [B, T, E]
    # Every share is one, so the loss is the mean normalized score summed over the experts: one.
    assert sequence_balance(logits, 2).item() == pytest.approx(1.0)
    skewed = 4 * torch.stack([even, pairs[:1].repeat(4, 1)]) - 2 + noise  # the second on 0 and 1
    scores = torch.sigmoid(skewed)
    probs = (scores / scores.sum(dim=-1, keepdim=True)).mean(dim=1)  # [B, E]
    shares = torch.tensor([[1.0] * 4, [2.0, 2.0, 0.0, 0.0]])  # [B, E]: slots over the even two
    by_hand = (shares * probs).sum(dim=-1).mean()
    assert sequence_balance(skewed, 2).item() == pytest.approx(by_hand.item())
    # The even sequence reads one, the skewed one about 1.75: its two favoured experts take all its slots and 88 percent of its normalized score.
    assert by_hand.item() > 1.3


def test_the_balance_loss_is_kept_in_training_alone_and_reaches_the_router_and_the_backbone() -> (
    None
):
    torch.manual_seed(0)
    net = LanguageModel(VOCAB, LAPTOP, balance_loss=True)
    x, _ = batch()
    with pytest.raises(RuntimeError, match="no balance loss"):  # before any training forward
        net.sequence_balance()
    net.eval()(x)
    assert all(mixture.balance is None for mixture in net.decoder.mixtures())
    net.train()(x)
    loss = net.sequence_balance()
    assert loss.requires_grad and loss.item() > 0.9 * MIXTURE_BLOCKS  # about one per mixture
    loss.backward()
    first = net.decoder.mixtures()[0]
    assert first.router.weight.grad is not None and first.router.weight.grad.abs().sum() > 0
    dense = net.decoder.blocks[0].ffn  # the backbone before the first mixture
    assert isinstance(dense, FeedForward)
    assert dense.up_proj.weight.grad is not None and dense.up_proj.weight.grad.abs().sum() > 0
    last = net.decoder.mixtures()[-1]  # its experts feed no router: the loss moves scores alone
    assert all(p.grad is None for p in last.experts.parameters())
    plain = model()
    plain(x)
    with pytest.raises(RuntimeError, match="no balance loss"):  # a model built without it
        plain.sequence_balance()


def test_the_compiled_model_keeps_the_balance_loss_in_a_single_graph_with_recompute() -> None:
    """The balance loss stored inside the recomputed first mixture, as the balance-loss block trains it: one graph, and the eager forward's gradients to bf16 precision, the loss's own part included."""

    def gradients(compiled: bool, weight: float) -> list[torch.Tensor]:
        torch.manual_seed(0)
        net = LanguageModel(VOCAB_SIZE, LAPTOP, recompute=1, balance_loss=True)
        tokens = torch.randint(0, VOCAB_SIZE, (2, LAPTOP.context_len), generator=generator(4))
        out = (torch.compile(net, fullgraph=True) if compiled else net)(tokens)  # [B, T, V]
        (out.square().mean() + weight * net.sequence_balance()).backward()
        return [p.grad.clone() for p in net.parameters() if p.grad is not None]

    eager, compiled = gradients(False, 1.0), gradients(True, 1.0)
    assert len(compiled) == len(eager)
    for grad, eager_grad in zip(compiled, eager):
        torch.testing.assert_close(grad, eager_grad, atol=2e-2, rtol=2e-2)
    without = gradients(True, 0.0)  # the loss's own part, compiled against eager
    eager_without = gradients(False, 0.0)
    parts = [(c - w, e - v) for c, w, e, v in zip(compiled, without, eager, eager_without)]
    assert any(e.abs().sum() > 0 for _, e in parts)
    for part, eager_part in parts:
        torch.testing.assert_close(part, eager_part, atol=1e-3, rtol=5e-2)
