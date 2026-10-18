"""Routing telemetry: what the router, the balancer, and the optimizer do to every mixture.

A ``Telemetry`` watches one training step at a time. Before the step it copies every parameter of the model and the router biases; every mixture keeps the router logits of its training forward, the sum of its router inputs as unit vectors, its expert outputs, and the residual stream leaving its block, which is also the next block's input, where that block's attention logits are recomputed; after the step it writes one JSON line with, per mixture, the balancer state (bias span, load distribution, load churn), the router's behavior on the batch (score entropy, selection margin, logit norm, the similarity of its inputs across tokens, the spread of its mean score across experts, and the load it would give on its scores alone, without the bias), the share of the router's update that acts on the mean direction of its inputs, the entropy of its weight's singular values and the similarity of its rows, the two mechanism readings: how the size of each expert's update follows the tokens the expert saw and how unevenly it is spread over the experts, the size being the Frobenius norm and, beside it under ``_spectral``, the spectral norm (the one Muon fixes), and how unevenly the router's rows moved, and the stability readings: the largest residual activation, attention logit, router logit, and expert output, and the kurtosis of each expert's outputs, which rises as they grow outliers; and, per parameter group of the partition and once more for the norm gains alone, the norm of its weights, the norm of the step's change to them, and the angle the step turned them through, the step size the optimizer gives the group relative to where it stands. Steps that are not watched cost nothing.

Shape letters used in the comments:
    N   tokens in a batch
    M   routing slots in a batch, N * K
    E   routed experts
    K   experts active per token
    D   model width
"""

import json
import math
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from tfg.model import LanguageModel, MixtureOfExperts
from tfg.optim import parameter_groups

MARGIN_QUANTILE = 0.1  # the low tail of the selection margin, where routing flips first
POWER_ITERATIONS = 100  # rounds of the power method behind each update's spectral norm
POWER_SEED = 0  # of the vector the power method starts from: fixed, so the reading is deterministic
GAINS = "gains"  # the norm gains as one more group of the readings; Moonlight's stability reading


def load_fractions(load: torch.Tensor) -> torch.Tensor:  # [E]
    """The share of the batch's routing slots each expert took."""
    return load.float() / load.sum().clamp(min=1)  # [E]


def usage_perplexity(fractions: torch.Tensor) -> float:  # [E]
    """exp of the load entropy, the number of experts in even use: E when balanced, down to k when the same k experts take every token (a token's k slots go to k different experts)."""
    entropy = -(fractions * fractions.clamp(min=1e-12).log()).sum()  # []
    return entropy.exp().item()


def churn(previous: torch.Tensor | None, current: torch.Tensor) -> float:  # [E], [E]
    """Total variation between two load distributions: the share of slots that changed expert."""
    if previous is None:
        return 0.0
    return 0.5 * (current - previous).abs().sum().item()


def input_similarity(units: torch.Tensor, count: int) -> float:  # [D]
    """The mean cosine between the inputs of two distinct tokens, from the sum of the ``count`` inputs as unit vectors: its square adds every pair's cosine twice and every token's with itself once. Zero when the inputs point every way, one when they all point the same way; a direction they share raises it, as it raises the logit of the experts whose router rows lean that way for every token at once."""
    total = units.double().dot(units.double())  # []
    return ((total - count) / (count * (count - 1))).item()


def common_share(update: torch.Tensor, units: torch.Tensor) -> float | None:  # [E, D], [D]
    """The share of a router update, in Frobenius norm, that acts on the mean direction of the router's inputs: the norm of the change it makes to every expert's logit of that direction over the norm of the whole update. One when the update moves every row along that direction alone, near one over the square root of D for a direction it does not favor, and at most one over the square root of the rank for an orthogonalized update, whose singular values are all equal. None when the update or the direction is zero."""
    direction = units.double().norm()  # []
    whole = update.double().norm()  # []
    if direction == 0 or whole == 0:
        return None
    return ((update.double() @ (units.double() / direction)).norm() / whole).item()


def raw_fractions(logits: torch.Tensor, num_active: int) -> torch.Tensor:  # [N, E]
    """The share of the batch's routing slots each expert would take on the router's scores alone, without the balancer's bias: what the router asks for before the bias corrects it."""
    chosen = logits.topk(num_active, dim=-1).indices  # the sigmoid keeps the order [N, K]
    return load_fractions(torch.bincount(chosen.flatten(), minlength=logits.size(1)))  # [E]


def router_entropy(logits: torch.Tensor) -> float:  # [N, E]
    """Entropy of the sigmoid scores normalized over experts, averaged over tokens (nats)."""
    scores = torch.sigmoid(logits)  # [N, E]
    probs = scores / scores.sum(dim=-1, keepdim=True)  # [N, E]
    return -(probs * probs.log()).sum(dim=-1).mean().item()


def selection_margins(
    logits: torch.Tensor, bias: torch.Tensor, num_active: int
) -> torch.Tensor:  # [N, E], [E]
    """Per token, the selection score of the last chosen expert minus the first rejected one."""
    selection = torch.sigmoid(logits) + bias  # [N, E]
    top = selection.topk(num_active + 1, dim=-1).values  # [N, K+1]
    return top[:, num_active - 1] - top[:, num_active]  # [N]


def svd_entropy(matrix: torch.Tensor) -> float:  # [E, D]
    """Entropy, in nats, of the singular values normalized to sum to one: the log of the rank when they are all equal, zero when one of them carries everything."""
    # pylint: disable-next=not-callable
    values = torch.linalg.svdvals(matrix.float())  # [min(E, D)]
    probs = values / values.sum().clamp(min=1e-12)  # [min(E, D)]
    return -(probs * probs.clamp(min=1e-12).log()).sum().item()


def row_similarity(matrix: torch.Tensor) -> float:  # [E, D]
    """Mean pairwise cosine between the rows: one when they all point the same way, zero when they are orthogonal; the router monitor of Huang et al."""
    rows = F.normalize(matrix.float(), dim=1)  # [E, D]
    cosines = rows @ rows.T  # [E, E]
    count = rows.size(0)
    return ((cosines.sum() - cosines.diagonal().sum()) / (count * (count - 1))).item()


def total_norm(matrices: list[torch.Tensor]) -> torch.Tensor:  # []
    """Frobenius norm of a list of matrices, as if they were one."""
    return torch.stack([m.norm() for m in matrices]).norm()


def update_norm(before: list[torch.Tensor], after: list[torch.Tensor]) -> torch.Tensor:  # []
    """Frobenius norm of the change across a list of matrices, as if they were one."""
    return total_norm([a - b for a, b in zip(after, before)])


def spectral_norms(matrices: torch.Tensor) -> torch.Tensor:  # [E, M, N]
    """The largest singular value of every matrix, by ``POWER_ITERATIONS`` rounds of the power method (the estimate behind ``torch.nn.utils.parametrizations.spectral_norm``) from one fixed random vector, drawn from its own generator so the reading is deterministic and touches nothing training draws from. A full SVD of every expert's every matrix costs seconds per logged step; this costs two matrix-vector products a round and lands within about one percent of the exact value where the largest singular values crowd together (Gaussian noise, a Newton-Schulz output), on it where one dominates, always from below."""
    num, _, cols = matrices.shape
    start = torch.randn(cols, 1, generator=torch.Generator().manual_seed(POWER_SEED))  # [N, 1]
    v = F.normalize(start.to(matrices).expand(num, -1, -1), dim=1)  # [E, N, 1]
    transposed = matrices.transpose(1, 2)  # [E, N, M]
    for _ in range(POWER_ITERATIONS):
        u = F.normalize(matrices @ v, dim=1)  # [E, M, 1]
        v = F.normalize(transposed @ u, dim=1)  # [E, N, 1]
    return (matrices @ v).flatten(1).norm(dim=1)  # the stretch of a unit vector [E]


def expert_update_norms(
    before: list[list[torch.Tensor]], experts: Iterable[nn.Module]
) -> tuple[torch.Tensor, torch.Tensor]:  # [E], [E]
    """Per expert, the Frobenius and the spectral norm of the step's change to its matrices, taken together as one block-diagonal matrix: the Frobenius norm adds their squares, the spectral norm is the largest of theirs. ``before`` holds every expert's matrices before the step, in ``parameters()`` order. The experts can come from several mixtures: each matrix of an expert is stacked across all of them and read in one batch, so the power method runs three times per step rather than three times per mixture, since its thousands of small kernels, not their arithmetic, are what the reading costs."""
    after = [[p.detach() for p in expert.parameters()] for expert in experts]  # E lists
    frobenius, spectral = [], []
    for i in range(len(before[0])):  # one matrix of an expert at a time, one stack held at once
        update = torch.stack([a[i] - b[i] for a, b in zip(after, before)])  # [E, M, N]
        frobenius.append(update.flatten(1).norm(dim=1))  # [E]
        spectral.append(spectral_norms(update))  # [E]
    return torch.stack(frobenius).norm(dim=0), torch.stack(spectral).amax(dim=0)


def angle(before: list[torch.Tensor], after: list[torch.Tensor]) -> float | None:
    """The angle, in radians, between two lists of matrices flattened into one vector each: the rotation a step gives a group of weights, its relative step size when small. Computed from the update rather than from the cosine, which rounds to one for the steps of training; None when ``before`` is zero."""
    sums = torch.zeros(3, dtype=torch.double, device=before[0].device)  # w.w, w.u, u.u
    for b, a in zip(before, after):
        w, u = b.flatten().double(), (a - b).flatten().double()  # the weights and their update
        sums += torch.stack([w.dot(w), w.dot(u), u.dot(u)])
    ww, wu, uu = sums.tolist()
    if ww == 0:
        return None
    along = wu / math.sqrt(ww)  # the update's component along the weights
    across = math.sqrt(max(uu - along * along, 0.0))  # and the rest of it
    return math.atan2(across, math.sqrt(ww) + along)


def correlation(x: torch.Tensor, y: torch.Tensor) -> float | None:  # [E], [E]
    """Pearson correlation, or None when either side is constant."""
    x, y = x.float(), y.float()
    if x.std() == 0 or y.std() == 0:
        return None
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


def dispersion(values: torch.Tensor) -> float | None:  # [E]
    """Coefficient of variation, std over mean, or None when the mean is zero."""
    mean = values.mean()
    return None if mean == 0 else (values.std() / mean).item()


def kurtosis(values: torch.Tensor) -> float | None:
    """Excess kurtosis of all the values: zero for a normal distribution, higher the heavier the tails; None when there are no values or they do not vary."""
    if values.numel() == 0:
        return None
    values = values.double()
    centered = values - values.mean()
    variance = centered.pow(2).mean()  # []
    if variance == 0:
        return None
    return (centered.pow(4).mean() / variance.pow(2) - 3).item()


def copy(params: Iterable[nn.Parameter]) -> dict[torch.Tensor, torch.Tensor]:
    """A detached copy of every parameter, keyed by the parameter it was taken from."""
    return {p: p.detach().clone() for p in params}


@dataclass(frozen=True)
class Snapshot:
    """A mixture's trainable routing state before a step."""

    bias: torch.Tensor  # [E]
    router: torch.Tensor  # [E, D]
    experts: list[list[torch.Tensor]]  # E lists of one expert's matrices

    @classmethod
    def of(cls, mixture: MixtureOfExperts, copies: dict[torch.Tensor, torch.Tensor]) -> "Snapshot":
        """From the ``copies`` of the model's parameters taken before the step."""
        return cls(
            bias=mixture.router_bias.clone(),
            router=copies[mixture.router.weight],
            experts=[[copies[p] for p in e.parameters()] for e in mixture.experts],
        )


@dataclass(frozen=True)
class Before:
    """The model before a step: a copy of every parameter, and every mixture's routing state."""

    copies: dict[torch.Tensor, torch.Tensor]
    mixtures: list[Snapshot]


class Telemetry:
    """Write one JSON line of routing readings per watched training step.

    With ``resume`` the file is continued instead of truncated.
    """

    def __init__(self, model: LanguageModel, path: Path, *, resume: bool = False):
        self.mixtures = model.decoder.mixtures()
        self.blocks = model.decoder.mixture_blocks()  # the block of each mixture
        self.inputs = model.decoder.mixture_inputs()  # and the block feeding it
        self.groups = {name: ps for name, ps in parameter_groups(model).items() if ps}
        gains = [p for p in self.groups["tables_and_norms"] if p.ndim < 2]
        self.groups[GAINS] = gains  # read on their own too: beside the token tables they vanish
        self.path = path
        self.previous: list[torch.Tensor | None] = [None] * len(self.mixtures)  # [E] each
        if not resume:
            self.path.write_text("", encoding="utf-8")
            return
        lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []
        if lines:  # continue the churn from the last reading rather than restarting it at zero
            device = next(model.parameters()).device  # the loads it is compared with live there
            last = json.loads(lines[-1])["mixtures"]
            self.previous = [torch.tensor(r["load"], device=device) for r in last]  # [E] each

    def snapshot(self) -> Before:
        """Copy the parameters and each mixture's routing state before a step, and clear what the mixtures kept of their last forward, so that one the step does not run is noticed."""
        copies = copy(dict.fromkeys(p for ps in self.groups.values() for p in ps))  # each once
        for mixture, block in zip(self.mixtures, self.blocks):
            mixture.router_logits = mixture.expert_outputs = block.residual = None
            mixture.router_input_units = None
        for block in self.inputs:
            block.residual = None
        return Before(copies, [Snapshot.of(mixture, copies) for mixture in self.mixtures])

    def write(self, step: int, before: Before) -> None:
        """Write the line of readings of the step just taken from the snapshot taken ``before`` it."""
        experts = [expert for mixture in self.mixtures for expert in mixture.experts]
        frobenius, spectral = expert_update_norms(
            [matrices for snap in before.mixtures for matrices in snap.experts], experts
        )  # one [E] after another, a mixture's experts each
        sizes = [len(mixture.experts) for mixture in self.mixtures]
        norms = zip(frobenius.split(sizes), spectral.split(sizes))
        readings = [
            self.reading(index, snap, mixture_norms)
            for index, (snap, mixture_norms) in enumerate(zip(before.mixtures, norms))
        ]
        groups = {
            name: self.group_reading(name, [before.copies[p] for p in params])
            for name, params in self.groups.items()
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"step": step, "mixtures": readings, "groups": groups}) + "\n")

    @contextmanager
    def watch(self, step: int) -> Generator[None]:
        """Take the snapshot before the step inside, and write the line of readings after it."""
        before = self.snapshot()
        yield
        self.write(step, before)

    @torch.no_grad()
    def reading(
        self, index: int, before: Snapshot, norms: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, object]:
        """Every telemetry value of one mixture for the step just taken, given the Frobenius and spectral norms of its experts' updates, which ``write`` takes for every mixture at once."""
        mixture, block = self.mixtures[index], self.blocks[index]
        logits, outputs = mixture.router_logits, mixture.expert_outputs  # [N, E], [M, D]
        units = mixture.router_input_units  # [D]
        residual, inputs = block.residual, self.inputs[index].residual  # [B, T, D] each
        if logits is None or outputs is None or units is None or residual is None or inputs is None:
            raise RuntimeError(f"mixture {index} was not run under the watch")
        fractions = load_fractions(mixture.expert_load)  # [E]
        raw = raw_fractions(logits, mixture.num_active)  # [E]
        mean_scores = torch.sigmoid(logits).mean(0)  # [E]
        margins = selection_margins(logits, before.bias, mixture.num_active)  # [N]
        frobenius, spectral = norms  # [E], [E]
        router_update = mixture.router.weight.detach() - before.router  # [E, D]
        row_norms = router_update.norm(dim=1)  # [E]
        bounds = [0, *mixture.expert_load.cumsum(0).tolist()]  # each expert's rows of outputs
        kurtoses = [kurtosis(outputs[start:end]) for start, end in pairwise(bounds)]
        reading: dict[str, object] = {
            "bias_span": (mixture.router_bias.max() - mixture.router_bias.min()).item(),
            "bias": mixture.router_bias.tolist(),
            "load": fractions.tolist(),
            "usage_perplexity": usage_perplexity(fractions),
            "max_load_ratio": (fractions.max() * fractions.numel()).item(),
            "load_churn": churn(self.previous[index], fractions),
            "router_entropy": router_entropy(logits),
            "router_logit_norm": logits.norm(dim=-1).mean().item(),
            "router_input_similarity": input_similarity(units, logits.size(0)),
            "mean_score_span": (mean_scores.max() - mean_scores.min()).item(),
            "raw_usage_perplexity": usage_perplexity(raw),
            "raw_max_load_ratio": (raw.max() * raw.numel()).item(),
            "margin_mean": margins.mean().item(),
            "margin_low": margins.quantile(MARGIN_QUANTILE).item(),
            "expert_update_norm": frobenius.tolist(),
            "expert_update_spectral_norm": spectral.tolist(),
            "update_load_correlation": correlation(frobenius, mixture.expert_load),
            "update_load_correlation_spectral": correlation(spectral, mixture.expert_load),
            "expert_update_dispersion": dispersion(frobenius),
            "expert_update_dispersion_spectral": dispersion(spectral),
            "router_row_dispersion": dispersion(row_norms),
            "router_update_common_share": common_share(router_update, units),
            "router_svd_entropy": svd_entropy(mixture.router.weight.detach()),
            "router_row_similarity": row_similarity(mixture.router.weight.detach()),
            "residual_max": residual.abs().max().item(),
            "attention_logit_max": block.attn.logit_max(block.attn_norm(inputs)),
            "router_logit_max": logits.abs().max().item(),
            "expert_output_max": outputs.abs().max().item(),
            "expert_kurtosis": kurtoses,
            "expert_kurtosis_max": max((k for k in kurtoses if k is not None), default=None),
        }
        self.previous[index] = fractions
        return reading

    @torch.no_grad()
    def group_reading(self, name: str, before: list[torch.Tensor]) -> dict[str, object]:
        """The norm of a parameter group's weights after the step, the norm of the step's change to them, and the angle it turned them."""
        after = [p.detach() for p in self.groups[name]]
        return {
            "weight_norm": total_norm(after).item(),
            "update_norm": update_norm(before, after).item(),
            "angular_step": angle(before, after),
        }
