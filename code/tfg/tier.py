"""The shapes the thesis trains at: one model shape and batch per tier.

The laptop tier is what the tests use and runs in seconds on a CPU. The Kaggle T4 tier fits the free 16 GB GPU and checks the code on real hardware while the H100 is pending. The mechanism tier is the shape of the plan's Gate 1, stress grid, and RQ3 block. The confirmation tier is the plan's larger shape: it keeps the mechanism tier's routing problem (the expert count, the active experts, the context, and the batch) and grows in width and depth, with narrower experts. Everything that holds at every tier, like the initialization scale or the balancer's update rate, stays a constant in ``tfg.model``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    name: str
    d_model: int
    num_blocks: int
    dense_blocks: int  # the first blocks, dense feed-forward instead of a mixture
    num_heads: int
    context_len: int
    num_experts: int
    num_active_experts: int
    d_expert: int  # hidden width of one routed expert
    d_shared: int  # hidden width of the shared expert
    batch_size: int  # sequences per step

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads

    @property
    def d_ffn(self) -> int:
        """Hidden width of a dense block's feed-forward network."""
        return 4 * self.d_model

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.context_len


LAPTOP = Tier(
    name="laptop",
    d_model=128,
    num_blocks=4,
    dense_blocks=1,
    num_heads=4,
    context_len=64,
    num_experts=16,
    num_active_experts=4,
    d_expert=64,
    d_shared=64,
    batch_size=8,
)

KAGGLE_T4 = Tier(
    name="kaggle_t4",
    d_model=384,
    num_blocks=6,
    dense_blocks=1,
    num_heads=6,
    context_len=512,
    num_experts=16,
    num_active_experts=4,
    d_expert=512,
    d_shared=512,
    batch_size=16,
)

MECHANISM = Tier(
    name="mechanism",
    d_model=512,
    num_blocks=8,
    dense_blocks=1,
    num_heads=8,
    context_len=1024,
    num_experts=32,
    num_active_experts=4,
    d_expert=768,
    d_shared=768,
    batch_size=64,
)

CONFIRMATION = Tier(
    name="confirmation",
    d_model=768,
    num_blocks=12,
    dense_blocks=1,
    num_heads=12,
    context_len=1024,
    num_experts=32,
    num_active_experts=4,
    d_expert=512,
    d_shared=512,
    batch_size=64,
)

TIERS = {tier.name: tier for tier in (LAPTOP, KAGGLE_T4, MECHANISM, CONFIRMATION)}
