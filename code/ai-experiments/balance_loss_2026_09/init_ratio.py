"""How hard the balance loss pulls at initialization, before the block: the thesis model at the mechanism tier, built as the block builds it, on one batch of 8 real sequences of the training split's first shard, on the laptop's CPU in float32. Per mixture, the sequence-wise balance loss (one is even) and the Frobenius norm of its router's gradient from the loss at the block's weight, 1e-4, over that of the cross-entropy's; and the same ratio over every parameter of the model. Three seeds, 3 to 5, the block's. Usage, from code/: PYTHONPATH=. uv run python ai-experiments/balance_loss_2026_09/init_ratio.py; writes outputs/init_ratio.txt."""

from pathlib import Path

import numpy as np
import torch
from tfg.data import DATA_DIR, VOCAB_SIZE, load_shard
from tfg.model import LanguageModel
from tfg.tier import TIERS
from tfg.train import cross_entropy

WEIGHT = 1e-4  # the block's, DeepSeek-V3's and V4's
SEQUENCES = 8  # of the tier's 64, for the laptop's memory
OUT = Path(__file__).resolve().parent / "outputs" / "init_ratio.txt"

tier = TIERS["mechanism"]
tokens = load_shard(DATA_DIR / "train_00000.npy")
length = tier.context_len + 1
batch = torch.from_numpy(
    np.stack([tokens[i * length : (i + 1) * length] for i in range(SEQUENCES)]).astype(np.int64)
)
inputs, targets = batch[:, :-1], batch[:, 1:]
lines = [
    f"mechanism tier at initialization, {SEQUENCES} sequences of {tier.context_len} tokens, weight {WEIGHT:g}"
]


def gradients(model: LanguageModel, loss: torch.Tensor) -> dict[str, torch.Tensor]:
    model.zero_grad()
    loss.backward(retain_graph=True)
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


for seed in (3, 4, 5):
    torch.manual_seed(seed)
    model = LanguageModel(VOCAB_SIZE, tier, balance_loss=True)
    model.train()
    ce = cross_entropy(model(inputs), targets)
    balance = model.sequence_balance()
    per_mixture = [mixture.balance.item() for mixture in model.decoder.mixtures()]
    ce_grads = gradients(model, ce)
    loss_grads = gradients(model, WEIGHT * balance)
    routers = [name for name in loss_grads if name.endswith("router.weight")]
    ratios = [(loss_grads[n].norm() / ce_grads[n].norm()).item() for n in routers]
    whole = (
        torch.stack([g.norm() for g in loss_grads.values()]).norm()
        / torch.stack([g.norm() for g in ce_grads.values()]).norm()
    ).item()
    lines.append(
        f"seed {seed}: cross-entropy {ce.item():.4f}, balance loss per mixture "
        + " ".join(f"{b:.4f}" for b in per_mixture)
        + f" (sum {balance.item():.4f})"
    )
    lines.append(
        "  router gradient, loss over cross-entropy, per mixture: "
        + " ".join(f"{r:.2e}" for r in ratios)
        + f"; every parameter: {whole:.2e}"
    )
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
