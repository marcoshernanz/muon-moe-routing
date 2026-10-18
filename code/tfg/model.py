"""A pre-norm decoder with a dense first block and mixture-of-experts blocks after it.

The shape of a model is a ``Tier`` from ``tfg.tier``; the constants here are the choices that hold at every tier.

Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    N   tokens in a batch, B * T
    M   routing slots in a batch, N * K
    D   model width
    H   attention heads
    Dh  width of one head, D / H
    F   hidden width of a feed-forward network
    E   routed experts in a mixture
    K   experts active per token
    V   vocabulary size
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tfg.tier import Tier

ROPE_BASE = 10000.0
BIAS_UPDATE_RATE = 1e-3
NORM_EPS = 1e-5
INIT_STD = 0.02
GROUPED_MM_CAPABILITY = (8, 0)  # the grouped matmul kernel needs an Ampere or newer GPU


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:  # [B, T, D]
    """Split the embedding axis into separate attention heads."""
    batch_size, seq_len, d_model = x.size()
    x = x.reshape(batch_size, seq_len, num_heads, d_model // num_heads)  # [B, T, H, Dh]
    return x.transpose(1, 2)  # [B, H, T, Dh]


def combine_heads(x: torch.Tensor) -> torch.Tensor:  # [B, H, T, Dh]
    """Merge the attention heads back into one embedding axis."""
    batch_size, num_heads, seq_len, d_head = x.size()
    x = x.transpose(1, 2)  # [B, T, H, Dh]
    return x.reshape(batch_size, seq_len, num_heads * d_head)  # [B, T, D]


def rotate_half(x: torch.Tensor) -> torch.Tensor:  # [B, H, T, Dh]
    """Pair each feature with the one half a head apart and rotate the pair."""
    x1, x2 = x.chunk(2, dim=-1)  # [B, H, T, Dh/2] each
    return torch.cat([-x2, x1], dim=-1)  # [B, H, T, Dh]


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:  # [B, H, T, Dh]
    """Rotate queries or keys by a position-dependent angle."""
    seq_len = x.size(2)
    return x * cos[:seq_len] + rotate_half(x) * sin[:seq_len]  # [B, H, T, Dh]


def rope_tables(context_len: int, d_head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the cosine and sine tables for the split-half rotation."""
    inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, d_head, 2) / d_head))  # [Dh/2]
    positions = torch.arange(context_len)  # [T]
    angles = positions[:, None] * inv_freq[None, :]  # [T, Dh/2]
    angles = torch.cat([angles, angles], dim=-1)  # [T, Dh]
    return angles.cos(), angles.sin()  # [T, Dh] each


class SelfAttention(nn.Module):
    """Causal multi-head attention with rotary positions, QK-norm, and a sigmoid output gate."""

    def __init__(self, tier: Tier):
        super().__init__()
        self.num_heads = tier.num_heads
        self.q_proj = nn.Linear(tier.d_model, tier.d_model, bias=False)
        self.k_proj = nn.Linear(tier.d_model, tier.d_model, bias=False)
        self.v_proj = nn.Linear(tier.d_model, tier.d_model, bias=False)
        self.g_proj = nn.Linear(tier.d_model, tier.d_model, bias=False)
        self.o_proj = nn.Linear(tier.d_model, tier.d_model, bias=False)

        self.q_norm = nn.RMSNorm(tier.d_head, eps=NORM_EPS)
        self.k_norm = nn.RMSNorm(tier.d_head, eps=NORM_EPS)

        cos, sin = rope_tables(tier.context_len, tier.d_head)  # [T, Dh] each
        self.rope_cos = nn.Buffer(cos, persistent=False)  # [T, Dh]
        self.rope_sin = nn.Buffer(sin, persistent=False)  # [T, Dh]

    def queries_keys(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # [B, T, D]
        """The queries and the keys of ``x``: split into heads, normalized, and rotated."""
        q = self.q_norm(split_heads(self.q_proj(x), self.num_heads))  # [B, H, T, Dh]
        k = self.k_norm(split_heads(self.k_proj(x), self.num_heads))  # [B, H, T, Dh]
        q = apply_rope(q, self.rope_cos, self.rope_sin)  # [B, H, T, Dh]
        k = apply_rope(k, self.rope_cos, self.rope_sin)  # [B, H, T, Dh]
        return q, k

    @torch.no_grad()
    def logit_max(self, x: torch.Tensor) -> float:  # [B, T, D]
        """The largest attention logit of ``x``: the largest scaled query-key product over every head and causal pair, the number the softmax sees and the precursor of an attention blow-up. The fused kernel of ``forward`` never materializes the scores, so this recomputes them, one sequence at a time to keep the memory at a head's worth of squares; the telemetry asks for it on watched steps only."""
        q, k = self.queries_keys(x)  # [B, H, T, Dh] each
        future = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        maxima = []  # one per sequence
        for queries, keys in zip(q, k):  # [H, T, Dh] each
            scores = queries @ keys.transpose(-1, -2) / math.sqrt(q.size(-1))  # [H, T, T]
            maxima.append(scores.masked_fill_(future, -math.inf).amax())
        return torch.stack(maxima).amax().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        q, k = self.queries_keys(x)  # [B, H, T, Dh] each
        v = split_heads(self.v_proj(x), self.num_heads)  # [B, H, T, Dh]

        # pylint: disable-next=not-callable
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, T, Dh]
        gate = torch.sigmoid(self.g_proj(x))  # [B, T, D]
        return self.o_proj(gate * combine_heads(attended))  # [B, T, D]


class FeedForward(nn.Module):
    """Gated MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., D]
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)  # [..., F]
        return self.down_proj(hidden)  # [..., D]


def sequence_balance(logits: torch.Tensor, num_active: int) -> torch.Tensor:  # [B, T, E]
    """DeepSeek-V3's sequence-wise balance loss before its weight (arXiv 2412.19437, equations 17 to 20): in each sequence, every expert's share of the top-k slots over its even share, times its mean score normalized over the experts, summed over the experts; then the mean over the sequences. The top-k is the scores' own, without the bias, as the paper writes it, so the loss sees the imbalance the scores would make and the bias cancels. It is one when every expert takes its even share of every sequence. The shares come from a selection, which has no gradient, so the loss moves the scores alone: it lowers the scores of the experts a sequence's scores favour."""
    num_experts = logits.size(-1)
    scores = torch.sigmoid(logits)  # [B, T, E]
    probs = scores / scores.sum(dim=-1, keepdim=True)  # [B, T, E]
    top = scores.topk(num_active, dim=-1).indices  # [B, T, K]
    # One on each expert of the top-k.
    slots = torch.zeros_like(scores).scatter_(-1, top, 1.0)  # [B, T, E]
    shares = slots.mean(dim=1) * num_experts / num_active  # [B, E]
    return (shares * probs.mean(dim=1)).sum(dim=-1).mean()  # []


class MixtureOfExperts(nn.Module):
    """Route each token to K of E narrow experts, plus one shared expert every token uses.

    Experts are selected by the top-k of ``sigmoid(logits) + bias`` but weighted by the raw sigmoid scores, renormalized, so the bias steers load without touching the values a token receives. The bias is not a parameter: ``update_biases`` moves it against the load of the last training batch (DeepSeek's loss-free balancing) instead of a loss. With ``balance_loss`` on, a training forward also keeps its ``sequence_balance``, for the trainer to add to the cross-entropy at a small weight: the balance loss the frontier recipes keep beside the bias.

    Dispatch sorts the N * K routing slots by expert, so every expert's tokens are one contiguous slice, and runs all experts at once with a grouped matmul per projection: no per-expert loop, no host synchronization. GPUs older than the kernel fall back to one matmul per slice.
    """

    def __init__(self, tier: Tier, *, balance_loss: bool = False):
        super().__init__()
        self.num_active = tier.num_active_experts
        self.balance_loss = balance_loss
        self.router = nn.Linear(tier.d_model, tier.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(tier.d_model, tier.d_expert) for _ in range(tier.num_experts)]
        )
        self.shared_expert = FeedForward(tier.d_model, tier.d_shared)
        self.router_bias = nn.Buffer(torch.zeros(tier.num_experts))  # [E]

        load = torch.zeros(tier.num_experts, dtype=torch.long)  # [E]
        self.expert_load = nn.Buffer(load, persistent=False)  # [E]
        self.router_logits: torch.Tensor | None = None  # of the last training forward [N, E]
        self.expert_outputs: torch.Tensor | None = None  # of the same, sorted by expert [M, D]
        self.router_input_units: torch.Tensor | None = (
            None  # its inputs, as unit vectors, summed [D]
        )
        self.balance: torch.Tensor | None = None  # its sequence_balance, with its graph []

    @torch.no_grad()
    def update_biases(self, rate: float = BIAS_UPDATE_RATE) -> None:
        """Raise the bias of every under-used expert by ``rate`` and lower it by as much for every over-used one."""
        fair_share = self.expert_load.sum() / len(self.experts)  # []
        self.router_bias += rate * torch.sign(fair_share - self.expert_load)  # [E]

    def run_experts(
        self, inputs: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:  # [M, D], [E]
        """Every expert on its own slice of ``inputs``: the first ``counts[0]`` rows go to expert 0, the next ``counts[1]`` to expert 1, and so on."""
        if (
            inputs.is_cuda
            and torch.cuda.get_device_capability(inputs.device) < GROUPED_MM_CAPABILITY
        ):
            return self.run_experts_by_slice(inputs, counts)
        # The CUDA kernel is bf16-only, and so is the compiler's stand-in for it on any device.
        eager_cpu = inputs.device.type == "cpu" and not torch.compiler.is_compiling()
        dtype = inputs.dtype if eager_cpu else torch.bfloat16
        gate, up, down = (
            torch.stack([getattr(expert, name).weight for expert in self.experts]).to(dtype)
            for name in ("gate_proj", "up_proj", "down_proj")
        )  # [E, F, D], [E, F, D], [E, D, F]
        offsets = counts.cumsum(0).to(torch.int32)  # end row of each expert's slice [E]
        inputs = inputs.to(dtype)  # [M, D]
        gated = F.grouped_mm(inputs, gate.transpose(1, 2), offs=offsets)  # [M, F]
        hidden = F.silu(gated) * F.grouped_mm(inputs, up.transpose(1, 2), offs=offsets)  # [M, F]
        return F.grouped_mm(hidden, down.transpose(1, 2), offs=offsets)  # [M, D]

    def run_experts_by_slice(
        self, inputs: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:  # [M, D], [E]
        """The same as ``run_experts`` with one matmul per expert; one host sync for the slices."""
        bounds = [0, *counts.cumsum(0).tolist()]  # [E+1]
        outputs = [
            expert(inputs[start:end])
            for expert, start, end in zip(self.experts, bounds[:-1], bounds[1:])
        ]  # E of [counts[e], D]
        return torch.cat(outputs)  # [M, D]

    def route(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [N, D]
        """The router's logits, the experts chosen per token, and their mixing weights."""
        with torch.autocast(tokens.device.type, enabled=False):
            logits = self.router(tokens.float())  # [N, E]
            scores = torch.sigmoid(logits)  # [N, E]
            chosen = (scores + self.router_bias).topk(self.num_active, dim=-1).indices  # [N, K]
            weights = scores.gather(-1, chosen)  # [N, K]
            weights = weights / weights.sum(dim=-1, keepdim=True)  # [N, K]
        return logits, chosen, weights.to(tokens.dtype)  # [N, E], [N, K], [N, K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        batch_size, seq_len, d_model = x.size()
        tokens = x.reshape(-1, d_model)  # [N, D]
        logits, chosen, weights = self.route(tokens)  # [N, E], [N, K], [N, K]

        slots = chosen.flatten()  # the N * K routing slots [M]
        # A comparison and a sum rather than bincount, whose output shape depends on the data and splits the compiled graph, or index_add, whose deterministic CUDA path adds the ones one at a time.
        experts = torch.arange(len(self.experts), device=slots.device)  # [E]
        load = (slots[:, None] == experts).sum(0)  # [E]
        # index_select rather than indexing: its backward, index_add, is deterministic on CPU.
        order = slots.argsort(stable=True)  # the slots sorted by expert [M]
        token = order // self.num_active  # the token each sorted slot belongs to [M]
        outputs = self.run_experts(tokens.index_select(0, token), load)  # [M, D]
        if self.training:  # for the balancer and the telemetry: hooks die under the compiler
            self.expert_load.copy_(load)  # [E]
            self.router_logits = logits.detach()  # [N, E]
            self.expert_outputs = outputs.detach()  # [M, D]
            # A sum rather than the inputs themselves, which would hold a batch's worth of every mixture's inputs through the optimizer step.
            self.router_input_units = F.normalize(tokens.detach().float(), dim=-1).sum(0)  # [D]
        if self.training and self.balance_loss:  # not detached: the trainer backpropagates it
            self.balance = sequence_balance(
                logits.reshape(batch_size, seq_len, -1), self.num_active
            )  # []
        outputs = outputs * weights.flatten().index_select(0, order)[:, None]  # [M, D]
        routed = torch.zeros_like(tokens).index_add_(0, token, outputs)  # [N, D]

        out = routed + self.shared_expert(tokens)  # [N, D]
        return out.reshape(batch_size, seq_len, d_model)  # [B, T, D]


class DecoderBlock(nn.Module):
    """Pre-norm attention, then a pre-norm feed-forward, dense or mixture."""

    def __init__(self, tier: Tier, *, is_dense: bool, balance_loss: bool = False):
        super().__init__()
        self.attn_norm = nn.RMSNorm(tier.d_model, eps=NORM_EPS)
        self.attn = SelfAttention(tier)
        self.ffn_norm = nn.RMSNorm(tier.d_model, eps=NORM_EPS)
        self.ffn = (
            FeedForward(tier.d_model, tier.d_ffn)
            if is_dense
            else MixtureOfExperts(tier, balance_loss=balance_loss)
        )
        self.residual: torch.Tensor | None = None  # the output of the last training forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        x = x + self.attn(self.attn_norm(x))  # [B, T, D]
        x = x + self.ffn(self.ffn_norm(x))  # [B, T, D]
        if self.training:
            self.residual = x.detach()  # for the telemetry [B, T, D]
        return x


class Decoder(nn.Module):
    """The block stack, dense first, then a final norm.

    With ``recompute`` set to K, a training forward keeps only the input of the first K mixture blocks and the backward pass runs each of them again to get the rest (activation checkpointing): the same numbers, bit for bit in an eager run, for one more forward pass of those blocks per step and the memory of their activations. The mixture blocks hold the most activations, and the first of them are the ones that lower the peak: the memory peaks as the backward pass starts, with every block's activations held, and a recomputed block brings its activations back when the backward pass reaches it, which for the last blocks is at that peak and for the first ones is after the rest are freed. Under ``torch.compile`` the compiler fuses the recomputed graph its own way, so the numbers match the run without it only to rounding. The blocks store tensors for the telemetry as they run, which the compiler accepts inside a recomputed region only once it is told not to store them again in the backward pass; without that it splits the graph there and the blocks run uncompiled. The balance loss a mixture keeps is stored the same way, with its graph, and its gradient comes through the recomputation like the cross-entropy's.
    """

    def __init__(self, tier: Tier, *, recompute: int = 0, balance_loss: bool = False):
        super().__init__()
        if not 0 <= recompute <= tier.num_blocks - tier.dense_blocks:
            raise ValueError(
                f"recompute {recompute} is not between 0 and the "
                f"{tier.num_blocks - tier.dense_blocks} mixture blocks"
            )
        self.recomputed = range(tier.dense_blocks, tier.dense_blocks + recompute)  # block indices
        if recompute:  # the blocks' stores for the telemetry belong to the first forward alone
            config = torch._dynamo.config  # pyright: ignore[reportPrivateUsage]  # no public name
            config.skip_fwd_side_effects_in_bwd_under_checkpoint = True  # ty: ignore[invalid-assignment]  # torch leaves the flag unannotated
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(tier, is_dense=i < tier.dense_blocks, balance_loss=balance_loss)
                for i in range(tier.num_blocks)
            ]
        )
        self.out_norm = nn.RMSNorm(tier.d_model, eps=NORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        for i, block in enumerate(self.blocks):
            if i in self.recomputed and self.training:
                x = checkpoint(block, x, use_reentrant=False)  # [B, T, D]
            else:
                x = block(x)  # [B, T, D]
        return self.out_norm(x)  # [B, T, D]

    def mixtures(self) -> list[MixtureOfExperts]:
        """The mixture-of-experts modules in block order, for the balancer and the telemetry."""
        return [block.ffn for block in self.blocks if isinstance(block.ffn, MixtureOfExperts)]

    def mixture_blocks(self) -> list[DecoderBlock]:
        """The blocks of ``mixtures``, in the same order, for the telemetry."""
        blocks = [block for block in self.blocks if isinstance(block, DecoderBlock)]  # all of them
        return [block for block in blocks if isinstance(block.ffn, MixtureOfExperts)]

    def mixture_inputs(self) -> list[DecoderBlock]:
        """The block before each of ``mixture_blocks``, in the same order: its output is the mixture block's input, which the telemetry reads the attention logits off."""
        blocks = [block for block in self.blocks if isinstance(block, DecoderBlock)]  # all of them
        mixtures = [i for i, block in enumerate(blocks) if isinstance(block.ffn, MixtureOfExperts)]
        if mixtures and mixtures[0] == 0:
            raise ValueError("block 0 must be dense: a mixture there has no block before it")
        return [blocks[i - 1] for i in mixtures]


class LanguageModel(nn.Module):
    """Token embeddings, the decoder, and an untied output head; ``recompute`` is the decoder's, and ``balance_loss`` has every mixture keep its sequence-wise balance loss."""

    def __init__(
        self, vocab_size: int, tier: Tier, *, recompute: int = 0, balance_loss: bool = False
    ):
        super().__init__()
        self.tier = tier
        self.embed_tokens = nn.Embedding(vocab_size, tier.d_model)
        self.decoder = Decoder(tier, recompute=recompute, balance_loss=balance_loss)
        self.lm_head = nn.Linear(tier.d_model, vocab_size, bias=False)
        self.apply(init_weights)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [B, T]
        hidden = self.decoder(self.embed_tokens(tokens))  # [B, T, D]
        return self.lm_head(hidden)  # [B, T, V]

    def sequence_balance(self) -> torch.Tensor:
        """The mixtures' sequence-wise balance losses of the last training forward, summed: DeepSeek-V3 defines the loss per MoE layer, so each mixture adds its own at the full weight; the model must have been built with ``balance_loss``."""
        mixtures = self.decoder.mixtures()
        losses = [mixture.balance for mixture in mixtures if mixture.balance is not None]
        if len(losses) < len(mixtures):
            raise RuntimeError("no balance loss kept: build the model with balance_loss and train")
        return torch.stack(losses).sum()  # []

    @torch.no_grad()
    def update_biases(self, rate: float = BIAS_UPDATE_RATE) -> None:
        """Apply the loss-free balancing step at ``rate`` to every mixture, after each optimizer step."""
        for mixture in self.decoder.mixtures():
            mixture.update_biases(rate)


def init_weights(module: nn.Module) -> None:
    """Draw every weight from one narrow normal; norm gains keep their ones."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=INIT_STD)
