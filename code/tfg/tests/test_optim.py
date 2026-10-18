"""Tests of the optimizer partitions: the grouping, the assignments, and the update rules."""

# pylint: disable=not-callable  # Pylint cannot infer torch's C-backed callables

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tfg.data import VOCAB_SIZE
from tfg.model import LanguageModel
from tfg.optim import (
    BETAS,
    GROUPS,
    KINDS,
    MATRICES,
    MOMENTUM,
    PARTITIONS,
    WEIGHT_DECAY,
    Momentum,
    Muon,
    adamw,
    adjusted_lr,
    assign,
    build,
    group_of,
    muon,
    parameter_groups,
    parameters,
)
from tfg.tier import LAPTOP

MIXTURES = LAPTOP.num_blocks - LAPTOP.dense_blocks
SHAPE = torch.Size([64, 128])


def model() -> LanguageModel:
    torch.manual_seed(0)
    return LanguageModel(VOCAB_SIZE, LAPTOP)


def ids(params: list[nn.Parameter]) -> set[int]:
    return {id(p) for p in params}


def one_step(optimizer: torch.optim.Optimizer, weight: nn.Parameter, grad: torch.Tensor) -> None:
    weight.grad = grad.clone()
    optimizer.step()
    optimizer.zero_grad()


def test_every_parameter_lands_in_exactly_one_group() -> None:
    net = model()
    groups = parameter_groups(net)
    assert {group: len(params) for group, params in groups.items()} == {
        "router": MIXTURES,
        "experts": MIXTURES * LAPTOP.num_experts * 3,
        "shared": MIXTURES * 3,
        "attention": LAPTOP.num_blocks * 5,
        "dense": LAPTOP.dense_blocks * 3,
        "tables_and_norms": 2 + LAPTOP.num_blocks * 4 + 1,
    }
    seen = [id(p) for params in groups.values() for p in params]
    assert len(seen) == len(set(seen)) == len(list(net.parameters()))
    assert all(p.ndim == 2 for group in MATRICES for p in groups[group])
    with pytest.raises(ValueError, match="no parameter group"):
        group_of("mystery.weight", nn.Parameter(torch.zeros(2, 2)))


def test_partitions_cover_every_group_and_keep_tables_on_adamw() -> None:
    for name, assignment in PARTITIONS.items():
        assert tuple(assignment) == GROUPS, name
        assert assignment["tables_and_norms"] == "adamw", name
        assert set(assignment.values()) <= set(KINDS), name
    assert all(PARTITIONS["adamw"][group] == "adamw" for group in GROUPS)
    assert all(PARTITIONS["muon"][group] == "muon" for group in MATRICES)
    assert PARTITIONS["muon_except_router"]["router"] == "adamw"
    assert PARTITIONS["muon_experts"]["attention"] == "adamw"
    assert all(PARTITIONS["momentum"][group] == "momentum" for group in MATRICES)
    assert PARTITIONS["muon_router"]["router"] == "muon"
    assert all(PARTITIONS["muon_router"][group] == "adamw" for group in GROUPS if group != "router")
    assert all(PARTITIONS["muon_dense"][group] == "adamw" for group in GROUPS if group != "dense")
    assert PARTITIONS["muon_dense"]["dense"] == "muon"
    feeding = PARTITIONS["muon_attention_dense"]
    assert feeding["attention"] == feeding["dense"] == "muon"
    assert all(feeding[group] == "adamw" for group in GROUPS if group not in ("attention", "dense"))
    with pytest.raises(ValueError, match="always stay"):
        assign("muon", ("tables_and_norms",))


def test_build_places_each_group_on_its_optimizer() -> None:
    net = model()
    groups = parameter_groups(net)
    optimizers = build(net, "muon_except_router", adamw_lr=1e-3, muon_lr=2e-3)
    assert [(k, type(o).__name__) for k, o in optimizers.items()] == [
        ("adamw", "AdamW"),
        ("muon", "Muon"),
    ]
    on_adamw = ids(parameters(optimizers["adamw"]))
    on_muon = ids(parameters(optimizers["muon"]))
    assert ids(groups["router"]) | ids(groups["tables_and_norms"]) == on_adamw
    assert on_muon == {id(p) for group in MATRICES if group != "router" for p in groups[group]}
    assert on_adamw | on_muon == ids(list(net.parameters()))  # a partition: all, once each
    assert on_adamw.isdisjoint(on_muon)
    assert all(g["peak_lr"] == g["lr"] for o in optimizers.values() for g in o.param_groups)
    assert optimizers["muon"].param_groups[0]["lr"] == 2e-3
    assert list(build(net, "adamw", adamw_lr=1e-3, muon_lr=1e-3)) == ["adamw"]
    kinds = build(net, "momentum", adamw_lr=1e-3, muon_lr=1e-3)
    assert [(k, type(o).__name__) for k, o in kinds.items()] == [
        ("adamw", "AdamW"),
        ("momentum", "Momentum"),
    ]


def test_the_router_only_arm_gives_the_routers_a_muon_of_their_own() -> None:
    net = model()
    groups = parameter_groups(net)
    optimizers = build(net, "muon_router", adamw_lr=1e-3, muon_lr=2e-3, router_lr_multiplier=2.0)
    assert [(k, type(o).__name__) for k, o in optimizers.items()] == [
        ("adamw", "AdamW"),
        ("muon", "Muon"),
    ]
    assert ids(parameters(optimizers["muon"])) == ids(groups["router"])
    assert ids(parameters(optimizers["adamw"])) == ids(list(net.parameters())) - ids(
        groups["router"]
    )
    (routers,) = optimizers["muon"].param_groups
    assert routers["lr"] == routers["peak_lr"] == pytest.approx(4e-3)
    assert routers["weight_decay"] == WEIGHT_DECAY


@pytest.mark.parametrize(
    ("partition", "holder", "rate"),
    [
        ("adamw", "adamw", 1e-3),
        ("muon_except_router", "adamw", 1e-3),
        ("muon", "muon", 2e-3),
        ("muon_router", "muon", 2e-3),
    ],
)
def test_the_routers_form_their_own_group_at_a_multiple_of_the_rate(
    partition: str, holder: str, rate: float
) -> None:
    net = model()
    routers = ids(parameter_groups(net)["router"])
    for multiplier in (1.0, 4.0):
        optimizers = build(
            net, partition, adamw_lr=1e-3, muon_lr=2e-3, router_lr_multiplier=multiplier
        )
        group = optimizers[holder].param_groups[-1]  # the routers are added last
        assert ids(group["params"]) == routers
        assert group["lr"] == group["peak_lr"] == pytest.approx(rate * multiplier)
        assert group["weight_decay"] == WEIGHT_DECAY
        others = [g for o in optimizers.values() for g in o.param_groups if g is not group]
        assert all(ids(g["params"]).isdisjoint(routers) for g in others)
        assert all(g["lr"] in (1e-3, 2e-3) for g in others)  # nothing else is scaled
    with pytest.raises(ValueError, match="must be positive"):
        build(net, partition, adamw_lr=1e-3, muon_lr=2e-3, router_lr_multiplier=0.0)


def test_the_second_moment_decay_reaches_every_adamw_group() -> None:
    net = model()
    for partition in ("adamw", "muon_except_router", "muon_router"):
        default = build(net, partition, adamw_lr=1e-3, muon_lr=2e-3)
        assert all(g["betas"] == BETAS for g in default["adamw"].param_groups)
        raised = build(net, partition, adamw_lr=1e-3, muon_lr=2e-3, adamw_beta2=0.98)
        assert all(g["betas"] == (BETAS[0], 0.98) for g in raised["adamw"].param_groups)
        assert all(
            "betas" not in g
            for o in raised.values()
            if o is not raised["adamw"]
            for g in o.param_groups
        )
    assert adamw(list(net.parameters()), 1e-3, beta2=0.98).defaults["betas"] == (BETAS[0], 0.98)


def test_adamw_decays_matrices_only() -> None:
    net = model()
    optimizer = adamw(list(net.parameters()), 1e-3)
    assert optimizer.defaults["fused"] is True  # one kernel for every tensor's update
    groups = optimizer.param_groups
    assert len(groups) == 2
    decayed, plain = groups[0], groups[1]
    assert decayed["weight_decay"] == WEIGHT_DECAY and plain["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decayed["params"])
    assert all(p.ndim < 2 for p in plain["params"])
    assert len(decayed["params"]) + len(plain["params"]) == len(list(net.parameters()))


def test_adamw_partition_matches_plain_adamw_step_for_step() -> None:
    grouped, plain = model(), model()
    optimizers = {
        grouped: build(grouped, "adamw", adamw_lr=1e-3, muon_lr=1e-3)["adamw"],
        plain: adamw(list(plain.parameters()), 1e-3),
    }
    tokens = torch.randint(0, VOCAB_SIZE, (2, 16), generator=torch.Generator().manual_seed(1))
    for _ in range(3):
        for net, optimizer in optimizers.items():
            logits = net(tokens[:, :-1])  # [B, T, V]
            F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten()).backward()
            optimizer.step()
            optimizer.zero_grad()
    assert all(torch.equal(a, b) for a, b in zip(grouped.parameters(), plain.parameters()))


def test_muon_update_is_near_orthogonal_and_the_control_matches_its_size() -> None:
    grad = torch.randn(*SHAPE, generator=torch.Generator().manual_seed(0))  # [M, N]
    rate = adjusted_lr(1.0, SHAPE)
    assert rate == pytest.approx(0.2 * math.sqrt(128))

    weight = nn.Parameter(torch.zeros(SHAPE))
    one_step(muon([weight], 1.0), weight, grad)
    orthogonalized = -weight.detach() / rate  # [M, N]
    singular_values = torch.linalg.svdvals(orthogonalized)  # [M]
    assert 0.5 < singular_values.min() and singular_values.max() < 1.5

    control = nn.Parameter(torch.zeros(SHAPE))
    one_step(Momentum([control], 1.0), control, grad)
    expected = rate * math.sqrt(64) * grad / grad.norm()  # [M, N]
    assert torch.allclose(-control.detach(), expected, atol=1e-6)
    assert 0.8 < weight.norm() / control.norm() < 1.2


def test_control_keeps_muon_momentum_exactly() -> None:
    generator = torch.Generator().manual_seed(2)
    grads = [torch.randn(*SHAPE, generator=generator) for _ in range(3)]  # [M, N] each
    control, plain = nn.Parameter(torch.zeros(SHAPE)), nn.Parameter(torch.zeros(SHAPE))
    control_opt = Momentum([control], 0.1)
    unorthogonalized = torch.optim.Muon(
        [plain],
        lr=0.1,
        weight_decay=WEIGHT_DECAY,
        momentum=0.95,
        ns_steps=0,
        adjust_lr_fn="match_rms_adamw",
    )
    for grad in grads:
        one_step(control_opt, control, grad)
        one_step(unorthogonalized, plain, grad)
    # Muon with no Newton-Schulz steps applies the Frobenius-normalized momentum, rounded to bf16; the control differs only by the sqrt(min(M, N)) size factor, so they agree to bf16 precision.
    gap = (control - math.sqrt(64) * plain).norm() / control.norm()
    assert gap < 1e-2


SHAPES = [(48, 32)] * 3 + [(32, 48)] * 2 + [(40, 40)] + [(48, 32)]  # tall, wide, square, repeated


def matrices(seed: int) -> list[nn.Parameter]:
    generator = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.randn(*shape, generator=generator)) for shape in SHAPES]


def step_all(optimizer: torch.optim.Optimizer, params: list[nn.Parameter], seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for param in params:
        param.grad = torch.randn(param.shape, generator=generator)
    optimizer.step()


def test_muon_is_torch_muon_bit_for_bit_and_resumes_from_its_state() -> None:
    """The matrices of one shape go through Newton-Schulz as a stack; the weights and the momentum buffers must still be torch's to the bit, over two parameter groups at different rates, and a state saved by one optimizer must continue in the other."""
    mine, theirs = matrices(0), matrices(0)
    stacked = muon(mine[:5], 0.02)
    stacked.add_param_group({"params": mine[5:], "lr": 0.05})
    reference = torch.optim.Muon(theirs[:5], **stacked.defaults)  # the same settings
    reference.add_param_group({"params": theirs[5:], "lr": 0.05})
    for seed in range(4):
        step_all(stacked, mine, seed)
        step_all(reference, theirs, seed)
    assert all(torch.equal(a, b) for a, b in zip(mine, theirs))
    state, reference_state = stacked.state_dict(), reference.state_dict()
    assert state["state"].keys() == reference_state["state"].keys()
    for index, buffers in state["state"].items():
        assert torch.equal(
            buffers["momentum_buffer"], reference_state["state"][index]["momentum_buffer"]
        )

    resumed = muon(theirs[:5], 0.02)  # torch's state, continued stacked
    resumed.add_param_group({"params": theirs[5:], "lr": 0.05})
    resumed.load_state_dict(reference_state)
    for seed in range(4, 6):
        step_all(stacked, mine, seed)
        step_all(resumed, theirs, seed)
    assert all(torch.equal(a, b) for a, b in zip(mine, theirs))

    plain = matrices(1)
    with pytest.raises(ValueError, match="Nesterov"):
        step_all(Muon(plain, lr=0.02, nesterov=False), plain, 0)


def test_the_control_takes_each_matrix_to_the_norm_of_an_orthogonal_one() -> None:
    """The control steps the matrices of one shape together; each must still move as the rule says for it alone: its Nesterov momentum, scaled to Frobenius norm sqrt(min(M, N)), at Muon's adjusted rate, after the decay."""
    params, lr = matrices(0), 0.02
    start = [param.detach().clone() for param in params]
    control = Momentum(params, lr)
    buffers = [torch.zeros_like(param) for param in params]
    for seed in range(3):
        before = [param.detach().clone() for param in params]
        step_all(control, params, seed)
        generator = torch.Generator().manual_seed(seed)
        for param, old, buffer in zip(params, before, buffers):
            grad = torch.randn(param.shape, generator=generator)
            buffer.lerp_(grad, 1 - MOMENTUM)
            update = grad.lerp(buffer, MOMENTUM)
            update *= math.sqrt(min(param.shape)) / update.norm()
            expected = old * (1 - lr * WEIGHT_DECAY) - adjusted_lr(lr, param.shape) * update
            torch.testing.assert_close(param.detach(), expected, atol=1e-6, rtol=1e-6)
    assert not any(torch.equal(param, old) for param, old in zip(params, start))
