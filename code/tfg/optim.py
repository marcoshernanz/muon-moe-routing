"""Optimizer partitions: which optimizer trains which group of parameters.

Every parameter of the model belongs to exactly one named group. A partition assigns each group one optimizer kind: AdamW, Muon, or the control that keeps Muon's momentum and update size but skips the orthogonalization. Token tables and norm gains always stay on AdamW, as in every production recipe. The partition is the independent variable of the thesis.

Shape letters used in the comments:
    M, N   rows and columns of one weight matrix
    G      matrices of one shape, stepped together
"""

import math
from collections.abc import Callable, Iterable
from typing import overload

import torch
from torch import nn
from torch.optim import Optimizer

GROUPS = ("router", "experts", "shared", "attention", "dense", "tables_and_norms")
MATRICES = GROUPS[:-1]  # every group Muon may train
KINDS = ("adamw", "muon", "momentum")

# Shared.
WEIGHT_DECAY = 0.1  # decoupled, on matrices only

# AdamW.
BETAS = (0.9, 0.95)

# Muon and its control.
MOMENTUM = 0.95  # Nesterov
NS_STEPS = 5
LR_ADJUST = "match_rms_adamw"  # scale each matrix's rate by 0.2 * sqrt(max(M, N))


def assign(kind: str, groups: Iterable[str]) -> dict[str, str]:
    """Every group on AdamW except ``groups``, which go on ``kind``."""
    chosen = set(groups)
    if "tables_and_norms" in chosen:
        raise ValueError("tables and norms always stay on AdamW")
    return {group: kind if group in chosen else "adamw" for group in GROUPS}


PARTITIONS = {
    "adamw": assign("muon", ()),
    "muon": assign("muon", MATRICES),  # the DeepSeek-V4 rule
    "muon_except_router": assign("muon", ("experts", "shared", "attention", "dense")),  # Qwen
    "muon_experts": assign("muon", ("experts",)),
    "muon_attention": assign("muon", ("attention",)),
    "momentum": assign("momentum", MATRICES),  # step-size control: Muon without orthogonalization
    "muon_router": assign("muon", ("router",)),  # the router-geometry test
    "muon_dense": assign("muon", ("dense",)),  # the dense block 0, which feeds the first mixture
    "muon_attention_dense": assign("muon", ("attention", "dense")),  # all that feeds mixture 1
}


def group_of(name: str, param: nn.Parameter) -> str:
    """The group of one parameter, from its dimensionality and its path in the model."""
    if param.ndim < 2 or name.startswith(("embed_tokens.", "lm_head.")):
        return "tables_and_norms"
    markers = (
        ("router", ".router."),
        ("experts", ".experts."),
        ("shared", ".shared_expert."),
        ("attention", ".attn."),
        ("dense", ".ffn."),  # after the mixture groups, whose paths also pass through .ffn.
    )
    for group, marker in markers:
        if marker in name:
            return group
    raise ValueError(f"no parameter group for {name}")


def parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Every parameter of the model in exactly one named group."""
    groups: dict[str, list[nn.Parameter]] = {group: [] for group in GROUPS}
    for name, param in model.named_parameters():
        groups[group_of(name, param)].append(param)
    return groups


def adamw(params: list[nn.Parameter], lr: float, *, beta2: float = BETAS[1]) -> torch.optim.AdamW:
    """AdamW with decoupled weight decay on the matrices only; gains and biases are not decayed. ``beta2``, the second-moment decay, is overridden by the tuning block's check run alone."""
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": WEIGHT_DECAY},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(BETAS[0], beta2), fused=True)


def by_shape(params: list[nn.Parameter]) -> list[list[nn.Parameter]]:
    """The parameters that have a gradient, grouped by shape in the order they come: the matrices of one group and shape are orthogonalized together."""
    groups: dict[torch.Size, list[nn.Parameter]] = {}
    for param in params:
        if param.grad is not None:
            groups.setdefault(param.shape, []).append(param)
    return list(groups.values())


def nesterov(
    state: dict[torch.Tensor, dict[str, torch.Tensor]],
    params: list[nn.Parameter],
    *,
    key: str,
    momentum: float,
) -> torch.Tensor:  # [G, M, N]
    """Move each momentum buffer of ``params``, kept in ``state`` under ``key``, toward its gradient and return the Nesterov updates of all the matrices, stacked: the momentum lines of ``torch.optim.Muon``, one matrix at a time."""
    updates = []
    for param in params:
        grad = param.grad
        assert grad is not None  # by_shape keeps only the parameters that have one
        if key not in state[param]:
            state[param][key] = torch.zeros_like(grad)  # [M, N]
        buffer = state[param][key]  # [M, N]
        buffer.lerp_(grad, 1 - momentum)
        updates.append(grad.lerp(buffer, momentum))  # [M, N]
    return torch.stack(updates)  # [G, M, N]


def orthogonalize(
    update: torch.Tensor, coefficients: tuple[float, float, float], *, steps: int, eps: float
) -> torch.Tensor:  # [G, M, N]
    """The Newton-Schulz iteration of ``torch.optim.Muon`` on a stack of matrices of one shape, operation for operation, with one batched matrix product in place of one product per matrix."""
    a, b, c = coefficients
    x = update.bfloat16()  # [G, M, N]
    tall = update.size(1) > update.size(2)
    if tall:
        x = x.transpose(1, 2)  # [G, N, M]
    x.div_(x.flatten(1).norm(dim=1).clamp(min=eps)[:, None, None])  # each matrix to norm one
    for _ in range(steps):
        gram = x @ x.transpose(1, 2)
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, gram_update, x, beta=a)
    return x.transpose(1, 2) if tall else x  # [G, M, N]


def adjusted_lr(lr: float, shape: torch.Size) -> float:
    """Muon's ``match_rms_adamw`` rule, so that updates have the entry size AdamW's have; the product is taken in torch's order, which makes the rate torch's to the bit."""
    return lr * (0.2 * math.sqrt(max(shape)))


def descend(params: list[nn.Parameter], update: torch.Tensor, *, lr: float, decay: float) -> None:
    """The decoupled weight decay and the step along ``update`` at the adjusted rate: the last lines of ``torch.optim.Muon``, one matrix at a time."""
    rate = adjusted_lr(lr, params[0].shape)
    for param, step in zip(params, update):  # [M, N] each
        param.mul_(1 - lr * decay)
        param.add_(step, alpha=-rate)


class Muon(torch.optim.Muon):
    """``torch.optim.Muon`` with the Newton-Schulz iteration run over the matrices of one shape at once.

    Torch's Muon orthogonalizes one matrix at a time: five rounds of three small matrix products each, for every one of the few hundred expert matrices, which leaves most of a large GPU idle and nearly doubled the whole training step on the H100. Here the matrices of one group and shape are stacked and orthogonalized together with batched products. Everything else is torch's step line for line, and the momentum stays in the optimizer's state as one buffer per parameter under torch's name, so the checkpoints are torch's and, on the CPU, so are the numbers bit for bit (checked in the tests). On a GPU the batched bf16 products round differently from torch's one-matrix ones, which there reduce in bf16: the numbers are torch's algorithm, rounded closer to fp32 than torch rounds it. Only the settings the thesis uses are handled: Nesterov momentum and the ``match_rms_adamw`` rate.
    """

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            if not group["nesterov"] or group["adjust_lr_fn"] != LR_ADJUST:
                raise ValueError("only Nesterov momentum and the match_rms_adamw rate are handled")
            for params in by_shape(group["params"]):
                update = nesterov(
                    self.state, params, key="momentum_buffer", momentum=group["momentum"]
                )  # [G, M, N]
                update = orthogonalize(
                    update, group["ns_coefficients"], steps=group["ns_steps"], eps=group["eps"]
                )  # [G, M, N]
                descend(params, update, lr=group["lr"], decay=group["weight_decay"])
        return loss


def muon(params: list[nn.Parameter], lr: float) -> Muon:
    return Muon(
        params,
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        nesterov=True,
        ns_steps=NS_STEPS,
        adjust_lr_fn=LR_ADJUST,
    )


class Momentum(Optimizer):
    """Muon's step without the orthogonalization.

    The same Nesterov momentum, scaled to the Frobenius norm an orthogonal matrix of the same shape has, sqrt(min(M, N)), then the same rate adjustment and decoupled weight decay. What it lacks against Muon is only the direction change, so it is the step-size control. It shares Muon's lines, and like Muon it takes the norms of the matrices of one shape at once.
    """

    def __init__(self, params: list[nn.Parameter], lr: float):
        super().__init__(params, {"lr": lr})

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for params in by_shape(group["params"]):
                update = nesterov(self.state, params, key="momentum", momentum=MOMENTUM)
                norms = update.flatten(1).norm(dim=1).clamp(min=1e-7)[:, None, None]  # [G, 1, 1]
                update *= math.sqrt(min(params[0].shape)) / norms  # [G, M, N]
                descend(params, update, lr=group["lr"], decay=WEIGHT_DECAY)
        return loss


def optimizer_of(
    kind: str, params: list[nn.Parameter], lr: float, *, adamw_beta2: float = BETAS[1]
) -> Optimizer:
    """One optimizer of ``kind`` over ``params`` at the peak rate ``lr``."""
    if kind == "adamw":
        return adamw(params, lr, beta2=adamw_beta2)
    if kind == "muon":
        return muon(params, lr)
    return Momentum(params, lr)


def parameters(optimizer: Optimizer) -> list[nn.Parameter]:
    """Every parameter an optimizer trains."""
    return [param for group in optimizer.param_groups for param in group["params"]]


def build(
    model: nn.Module,
    partition: str,
    *,
    adamw_lr: float,
    muon_lr: float,
    router_lr_multiplier: float = 1.0,
    adamw_beta2: float = BETAS[1],
) -> dict[str, Optimizer]:
    """The optimizers of one partition by kind: AdamW, plus Muon or the control when a group uses them.

    The routers form their own parameter group on whichever optimizer the partition gives them, at that optimizer's rate times ``router_lr_multiplier``; when nothing else is on their kind, as in the router-only arm, that group is an optimizer of its own. Every parameter group remembers its ``peak_lr`` so the training loop can scale all rates by one warmup factor. ``adamw_beta2`` reaches every AdamW group.
    """
    if router_lr_multiplier <= 0:
        raise ValueError(f"router_lr_multiplier {router_lr_multiplier} must be positive")
    by_kind: dict[str, list[nn.Parameter]] = {kind: [] for kind in KINDS}
    groups = parameter_groups(model)
    for group, params in groups.items():
        if group != "router":
            by_kind[PARTITIONS[partition][group]] += params
    rates = {"adamw": adamw_lr, "muon": muon_lr, "momentum": muon_lr}
    optimizers = {
        kind: optimizer_of(kind, params, rates[kind], adamw_beta2=adamw_beta2)
        for kind, params in by_kind.items()
        if params
    }
    router_kind = PARTITIONS[partition]["router"]
    router_lr = rates[router_kind] * router_lr_multiplier
    if router_kind in optimizers:
        optimizers[router_kind].add_param_group(
            {
                "params": groups["router"],
                "lr": router_lr,
                "weight_decay": WEIGHT_DECAY,  # a matrix: decayed like the rest, on every kind
            }
        )
    else:
        optimizers[router_kind] = optimizer_of(
            router_kind, groups["router"], router_lr, adamw_beta2=adamw_beta2
        )
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            group["peak_lr"] = group["lr"]
    return optimizers
