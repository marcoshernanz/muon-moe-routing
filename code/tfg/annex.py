"""The annex readings: what else the panel says about a run's routing, computed once at the end.

Six readings per mixture, from the panel of ``tfg.panel`` sent through a run's last checkpoint, reported in an appendix; no hypothesis depends on them. Router saturation is OLMoE's: per token, the overlap between the K experts chosen at the end and the K chosen at the early checkpoint the trainer keeps from one percent into the run, over K, averaged over the panel. Expert co-activation is OLMoE's too: of the tokens sent to expert i, the share also sent to expert j, an E by E matrix, with its largest and its mean off-diagonal entry as the summaries. Routing purity is the Routing Testbed's agreement with a reference router that sends each domain to experts of its own: per expert, the share of its tokens that come from its most frequent domain, averaged over the experts that received any; one quarter when the four domains are routed alike. The hidden-state cosine is the caveat of the Myth paper: the mean cosine between the normalized hidden states the router sees, one token from each of two domains (between) or two tokens of one (within), beside the cosine between the domains' expert-usage vectors, which the paper predicts follows it. Expert similarity runs every expert on one sample of tokens and takes the mean cosine between two experts' outputs on the same token, an E by E matrix again. Gradient alignment is the coupling of Ahrac, Hochwald, and Geva, who show that a router row and its expert receive gradients along the same input directions: from one backward pass over the first batch of every domain, the cosine between an expert's input-weight gradient and the part of it that acts along the gradient of the expert's router row; one when the whole gradient lies along that direction, one over the square root of the width when the direction is nothing special to the expert.

Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    N   tokens in a batch, B * T
    D   model width
    E   routed experts in a mixture
    K   experts active per token
    L   mixtures in the model
    G   domains in the panel
    S   sampled tokens
    R   rows of an expert's input weights, gate and up together
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import tyro

from tfg import checkpoint
from tfg.data import VOCAB_SIZE
from tfg.model import LanguageModel, MixtureOfExperts
from tfg.panel import DOMAINS, PANEL_DIR, batches, routing
from tfg.tier import TIERS, Tier
from tfg.train import EARLY_CHECKPOINT, autocast, cross_entropy, device

FILE = "annex.json"
SAMPLE_TOKENS = 2048  # per domain, spread over its first batch: what every expert is run on


def onehot(chosen: torch.Tensor, num_experts: int) -> torch.Tensor:  # [N, K]
    """Which experts each token chose, as a matrix of ones and zeros."""
    # pylint: disable-next=not-callable
    return F.one_hot(chosen, num_experts).sum(1)  # [N, E]


def saturation(early: torch.Tensor, final: torch.Tensor) -> float:  # [N, E] each
    """OLMoE's router saturation: per token, the overlap between the experts chosen early and the ones chosen at the end, over the K chosen, averaged over the tokens."""
    overlap = (early * final).sum(1).double()  # [N]
    return (overlap / final.sum(1)).mean().item()


def coactivation(pairs: torch.Tensor) -> npt.NDArray[np.float64]:  # [E, E]
    """OLMoE's expert co-activation, from the count of tokens sent to each pair of experts, with the count sent to each expert on the diagonal: of the tokens sent to expert i, the share also sent to expert j; the diagonal is left at zero."""
    counts = pairs.double()
    matrix = counts / counts.diagonal().clamp(min=1)[:, None]  # [E, E]
    matrix.fill_diagonal_(0.0)
    return matrix.cpu().numpy()


def off_diagonal(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:  # [E, E]
    """The entries off the diagonal, as one vector."""
    return matrix[~np.eye(len(matrix), dtype=bool)]  # [E * (E - 1)]


def purity(counts: npt.NDArray[np.float64]) -> float:  # [G, E]
    """The Routing Testbed's routing purity: per expert, the share of its tokens that come from its most frequent domain, averaged over the experts that received any."""
    received = counts.sum(0)  # [E]
    experts = received > 0  # [E]
    return float((counts.max(0)[experts] / received[experts]).mean())


def cosines(units: torch.Tensor, counts: torch.Tensor) -> npt.NDArray[np.float64]:  # [G, D], [G]
    """The mean cosine between the unit hidden states of two tokens, one from each of two domains, from the sum of each domain's unit states and its token count; on the diagonal, between two distinct tokens of the one domain."""
    dots = units @ units.T  # the sum over token pairs of their cosine [G, G]
    pairs = counts[:, None] * counts[None, :]  # [G, G]
    own = torch.diag(counts)  # every token's cosine with itself, on the diagonal [G, G]
    return ((dots - own) / (pairs - own)).cpu().numpy()


def row_cosines(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:  # [G, E]
    """The cosine between every two rows."""
    rows = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)  # [G, E]
    return rows @ rows.T  # [G, G]


@torch.no_grad()
def similarity(
    mixture: MixtureOfExperts, tokens: torch.Tensor
) -> npt.NDArray[np.float64]:  # [S, D]
    """The mean cosine between the outputs of two experts on the same token, over the tokens, for every pair: an E by E matrix with ones on the diagonal."""
    with autocast(tokens.device):
        outputs = torch.stack([expert(tokens) for expert in mixture.experts])  # [E, S, D]
    units = F.normalize(outputs.double(), dim=-1).flatten(1)  # [E, S * D]
    return ((units @ units.T) / tokens.size(0)).cpu().numpy()  # [E, E]


def alignment(router: torch.Tensor, expert: torch.Tensor) -> float | None:  # [D], [R, D]
    """The cosine between an expert's input-weight gradient and the part of it that acts along its router row's gradient: the norm of the gradient's action on the unit router direction over the gradient's whole norm; None when either gradient is zero."""
    if not router.any() or not expert.any():
        return None
    direction = F.normalize(router.double(), dim=0)  # [D]
    return ((expert.double() @ direction).norm() / expert.double().norm()).item()


def sample(hidden: torch.Tensor) -> torch.Tensor:  # [N, D]
    """Up to ``SAMPLE_TOKENS`` of the rows, evenly spread, as their own tensor."""
    return hidden[:: math.ceil(len(hidden) / SAMPLE_TOKENS)].clone()  # [S, D]


def load(run_dir: Path, shape: Tier, dev: torch.device, *, file: str) -> tuple[LanguageModel, int]:
    """A saved model of the run, in evaluation mode, and the step it was saved at."""
    model = LanguageModel(VOCAB_SIZE, shape).to(dev)
    step = checkpoint.load_model(run_dir, model, file=file)
    return model.eval(), step


@dataclass
class Tally:
    """What one pass of the panel through the two checkpoints accumulates."""

    counts: torch.Tensor  # tokens of each domain sent to each expert [L, G, E]
    pairs: (
        torch.Tensor
    )  # tokens sent to each pair of experts, and to each on the diagonal [L, E, E]
    overlap: torch.Tensor  # over K, of the early and the final choices, summed over tokens [L]
    units: torch.Tensor  # the sum of the unit hidden states of each domain [L, G, D]
    samples: list[list[torch.Tensor]]  # per mixture, a sample of each domain's first batch [S, D]
    firsts: dict[str, torch.Tensor]  # the first batch of each domain [B, T]
    tokens: dict[str, int]  # per domain

    @classmethod
    def of(
        cls, model: LanguageModel, early: LanguageModel, *, panel_dir: Path, batch_size: int
    ) -> "Tally":
        """Send the panel through both models."""
        shape, dev = model.tier, next(model.parameters()).device
        num_mixtures, num_experts = len(model.decoder.mixtures()), shape.num_experts
        tally = cls(
            counts=torch.zeros(
                num_mixtures, len(DOMAINS), num_experts, dtype=torch.long, device=dev
            ),
            pairs=torch.zeros(
                num_mixtures, num_experts, num_experts, dtype=torch.double, device=dev
            ),
            overlap=torch.zeros(num_mixtures, dtype=torch.double, device=dev),
            units=torch.zeros(
                num_mixtures, len(DOMAINS), shape.d_model, dtype=torch.double, device=dev
            ),
            samples=[[] for _ in range(num_mixtures)],
            firsts={},
            tokens=dict.fromkeys(DOMAINS, 0),
        )
        for domain, inputs in batches(panel_dir, shape, batch_size=batch_size, dev=dev):  # [B, T]
            first = domain not in tally.firsts
            index = DOMAINS.index(domain)
            seen, before = routing(model, inputs), routing(early, inputs)
            for mixture, ((hidden, chosen), (_, chosen_early)) in enumerate(zip(seen, before)):
                active = onehot(chosen, num_experts).double()  # [N, E]
                tally.counts[mixture, index] += active.sum(0).long()
                tally.pairs[mixture] += active.T @ active
                tally.overlap[mixture] += (active * onehot(chosen_early, num_experts)).sum()
                tally.units[mixture, index] += F.normalize(hidden.double(), dim=1).sum(0)
                if first:
                    tally.samples[mixture].append(sample(hidden))
            if first:
                tally.firsts[domain] = inputs
            tally.tokens[domain] += inputs.numel()
        tally.overlap /= shape.num_active_experts
        return tally

    def reading(self, mixture: MixtureOfExperts, index: int) -> dict[str, object]:
        """Every annex reading of one mixture, after the gradient pass."""
        usage = self.counts[index].double().cpu().numpy()  # [G, E]
        counts = torch.tensor(
            [self.tokens[domain] for domain in DOMAINS],
            dtype=torch.double,
            device=self.counts.device,
        )  # [G]
        pairs, hidden, routed = (
            coactivation(self.pairs[index]),
            cosines(self.units[index], counts),
            row_cosines(usage),
        )  # [E, E], [G, G], [G, G]
        similar = similarity(mixture, torch.cat(self.samples[index]))  # [E, E]
        router = mixture.router.weight.grad  # [E, D]
        assert router is not None
        aligned = [
            alignment(router[e], input_grad(expert)) for e, expert in enumerate(mixture.experts)
        ]
        defined = [value for value in aligned if value is not None]
        return {
            "saturation": (self.overlap[index] / sum(self.tokens.values())).item(),
            "coactivation": pairs.tolist(),
            "coactivation_max": float(off_diagonal(pairs).max()),
            "coactivation_mean": float(off_diagonal(pairs).mean()),
            "routing_purity": purity(usage),
            "hidden_cosine": hidden.tolist(),
            "hidden_cosine_between": float(off_diagonal(hidden).mean()),
            "hidden_cosine_within": float(hidden.diagonal().mean()),
            "routing_cosine": routed.tolist(),
            "routing_cosine_between": float(off_diagonal(routed).mean()),
            "expert_similarity": similar.tolist(),
            "expert_similarity_mean": float(off_diagonal(similar).mean()),
            "gradient_alignment": aligned,
            "gradient_alignment_mean": sum(defined) / len(defined) if defined else None,
        }


def input_grad(expert: torch.nn.Module) -> torch.Tensor:  # [R, D]
    """The gradient of an expert's input weights, gate and up stacked; zero where the expert took no part."""
    weights = [expert.get_parameter(f"{name}.weight") for name in ("gate_proj", "up_proj")]
    return torch.cat([torch.zeros_like(w) if w.grad is None else w.grad for w in weights])


def evaluate(
    run_dir: Path, *, panel_dir: Path = PANEL_DIR, batch_size: int | None = None
) -> dict[str, object]:
    """Send the panel through the run's last checkpoint and its early one; write and return ``annex.json``."""
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    shape = TIERS[settings["tier"]]
    dev = device()
    if not (run_dir / checkpoint.EARLY_FILE).exists():
        raise FileNotFoundError(
            f"{run_dir} has no {checkpoint.EARLY_FILE}: the trainer writes it "
            f"{EARLY_CHECKPOINT:.0%} into a run, and the saturation reading needs it"
        )
    model, step = load(run_dir, shape, dev, file=checkpoint.FILE)
    early, early_step = load(run_dir, shape, dev, file=checkpoint.EARLY_FILE)
    batch_size = batch_size or shape.batch_size
    tally = Tally.of(model, early, panel_dir=panel_dir, batch_size=batch_size)
    model.zero_grad(set_to_none=True)
    for inputs in tally.firsts.values():  # one backward pass, over the first batch of each domain
        with autocast(dev):
            loss = cross_entropy(model(inputs[:, :-1]), inputs[:, 1:])  # []
        loss.backward()
    mixtures = [tally.reading(m, i) for i, m in enumerate(model.decoder.mixtures())]
    result: dict[str, object] = {
        "step": step,
        "early_step": early_step,
        "tokens": tally.tokens,
        "batch_size": batch_size,  # of each domain's first batch, which the sample and the gradients come from
        "sample": sum(len(s) for s in tally.samples[0]),
        "chance": {  # the readings' values when the routing is uniform at random
            "coactivation": (shape.num_active_experts - 1) / (shape.num_experts - 1),
            "routing_purity": 1 / len(DOMAINS),
            "gradient_alignment": 1 / math.sqrt(shape.d_model),
        },
        "mixtures": mixtures,
    }
    (run_dir / FILE).write_text(json.dumps(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "step": step,
                "saturation": [m["saturation"] for m in mixtures],
                "routing_purity": [m["routing_purity"] for m in mixtures],
            }
        )
    )
    return result


def main(args: list[str] | None = None) -> dict[str, object]:
    """The command line: every setting of ``evaluate`` as a flag."""
    return tyro.cli(evaluate, args=args)


if __name__ == "__main__":
    main()
