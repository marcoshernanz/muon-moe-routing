"""The specialization panel: four domain slices of one token count, and where a trained model's routers send them.

The web slice is a held-out FineWeb-Edu validation shard; math, code, and encyclopedic text stream from the Hub and are tokenized with the thesis tokenizer, documents in file order, each followed by the newline the shards were made with. The evaluation runs every slice through a run's checkpoint and counts, per mixture, how many of each domain's tokens reach each expert: the share per expert, which sums to K against a uniform K / E, and the divergence between the domains' usage distributions, which is the endpoint. The tokenizer compresses code poorly and one whitespace id carries a third of its tokens, so the code slice is counted twice, on all tokens and on non-whitespace tokens, and the divergence is reported both ways.

Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    N   tokens in a batch, B * T
    D   model width
    E   routed experts in a mixture
    K   experts active per token
    L   mixtures in the model
    P   tokens in a slice
    V   vocabulary size
    W   windows in a slice
"""

import gc
import json
import time
from collections.abc import Callable, Generator, Iterator
from itertools import batched
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tiktoken
import torch
import tyro
from datasets import load_dataset  # pyright: ignore[reportMissingTypeStubs]
from huggingface_hub.hf_file_system import HfFileSystemFile, HfFileSystemStreamFile
from scipy import stats
from torch import nn

from tfg import checkpoint
from tfg.data import DATA_DIR, TOKEN_DTYPE, VOCAB_SIZE, download_file, load_shard, tokenizer
from tfg.model import LanguageModel, MixtureOfExperts
from tfg.tier import TIERS, Tier
from tfg.train import autocast, device

PANEL_DIR = DATA_DIR.parent / "panel"
TOKENS = 2_000_000  # per domain
SEPARATOR = "\n"  # follows every document, as when the shards were made
WEB_SHARD = (
    "validation_00009.npy"  # a full held-out shard; the loss reads the first, the last is short
)
SOURCES = {  # the raw-text domains: Hub dataset, config, text column
    "math": ("open-web-math/open-web-math", None, "text"),
    "code": ("codeparrot/codeparrot-clean-valid", None, "content"),
    "wiki": ("wikimedia/wikipedia", "20231101.en", "text"),
}
DOMAINS = ["web", *SOURCES]
COUNTED = [*DOMAINS, "code_nonspace"]  # the code slice again, without its whitespace tokens
DOCUMENTS_PER_BATCH = 64  # tokenized together
SETTLE_SECONDS = 120.0  # at most, for a closed stream's reader to let go of its Hub file
SETTLE_POLL = 0.1  # seconds between looks
HUB_FILES = (HfFileSystemFile, HfFileSystemStreamFile)  # what a stream reads through
FILE = "specialization.json"


def documents(domain: str) -> Generator[str]:
    """The texts of a raw-text domain, streamed from the Hub in file order."""
    name, config, column = SOURCES[domain]
    for row in load_dataset(name, config, split="train", streaming=True):
        yield row[column]


def tokenize(
    texts: Iterator[str], bpe: tiktoken.Encoding, tokens: int
) -> npt.NDArray[np.uint16]:  # [P]
    """The first ``tokens`` ids of the texts, each followed by the separator."""
    pieces: list[npt.NDArray[np.uint16]] = []
    total = 0
    for batch in batched(texts, DOCUMENTS_PER_BATCH):
        for ids in bpe.encode_batch([text + SEPARATOR for text in batch]):
            pieces.append(np.asarray(ids, dtype=TOKEN_DTYPE))
            total += len(pieces[-1])
        if total >= tokens:
            return np.concatenate(pieces)[:tokens]  # [P]
    raise ValueError(f"the texts hold {total} tokens, fewer than the {tokens} asked for")


def settle() -> None:
    """Wait for the readers of closed streams to let go of their Hub files. A parquet stream reads its next batch on a background thread, and a read still in flight when the interpreter exits calls back into an interpreter that is finalizing and never returns."""
    deadline = time.monotonic() + SETTLE_SECONDS
    gc.collect()
    while any(type(item) in HUB_FILES for item in gc.get_objects()):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"a Hub stream reader is still busy {SETTLE_SECONDS:g}s after its stream closed"
            )
        time.sleep(SETTLE_POLL)
        gc.collect()


def domain_slice(domain: str, bpe: tiktoken.Encoding, tokens: int) -> npt.NDArray[np.uint16]:
    """The first ``tokens`` ids of a raw-text domain, its stream closed and its reader gone before they are returned."""  # [P]
    texts = documents(domain)
    try:
        return tokenize(texts, bpe, tokens)
    finally:
        texts.close()
        settle()


def web_slice(data_dir: Path, tokens: int) -> npt.NDArray[np.uint16]:  # [P]
    """The first ``tokens`` ids of the held-out shard."""
    shard = load_shard(download_file(WEB_SHARD, data_dir))
    if len(shard) < tokens:
        raise ValueError(
            f"{WEB_SHARD} holds {len(shard)} tokens, fewer than the {tokens} asked for"
        )
    return np.array(shard[:tokens])  # [P]


def whitespace_mask(bpe: tiktoken.Encoding) -> npt.NDArray[np.bool_]:  # [V]
    """Which ids decode to whitespace alone."""
    return np.array([bpe.decode([index]).isspace() for index in range(bpe.n_vocab)])


def build(panel_dir: Path = PANEL_DIR, *, tokens: int = TOKENS, data_dir: Path = DATA_DIR) -> Path:
    """Write one ``.npy`` of ``tokens`` ids per domain and the whitespace mask under ``panel_dir``, replacing what is there."""
    panel_dir.mkdir(parents=True, exist_ok=True)
    bpe = tokenizer(data_dir)
    np.save(panel_dir / "whitespace.npy", whitespace_mask(bpe))
    for domain in DOMAINS:
        ids = (
            web_slice(data_dir, tokens) if domain == "web" else domain_slice(domain, bpe, tokens)
        )  # [P]
        np.save(panel_dir / f"{domain}.npy", ids)
        print(f"{domain}: {len(ids)} tokens")
    return panel_dir


def batches(
    panel_dir: Path, shape: Tier, *, batch_size: int, dev: torch.device
) -> Generator[tuple[str, torch.Tensor]]:
    """The panel, domain by domain: every slice cut into windows of the tier's context and served in batches of ``batch_size`` on ``dev``, each with its domain's name; the slices must be of one length."""
    slices = {domain: np.load(panel_dir / f"{domain}.npy") for domain in DOMAINS}  # [P] each
    if len({len(ids) for ids in slices.values()}) != 1:
        raise ValueError(f"the slices under {panel_dir} differ in length; rebuild the panel")
    for domain, ids in slices.items():
        windows = ids[: len(ids) // shape.context_len * shape.context_len]  # [W * T]
        windows = windows.reshape(-1, shape.context_len)  # [W, T]
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size].astype(np.int64)  # [B, T]
            yield domain, torch.from_numpy(batch).to(dev)  # [B, T]


def routing(
    model: LanguageModel, inputs: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:  # [B, T]
    """Per mixture, in block order, the hidden states its router sees and the experts it chooses for the tokens of a batch, through a hook on each mixture's input."""
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []  # L of ([N, D], [N, K])

    def record(mixture: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        assert isinstance(mixture, MixtureOfExperts)
        tokens = args[0].reshape(-1, args[0].size(-1))  # [N, D]
        seen.append((tokens, mixture.route(tokens)[1]))  # [N, D], [N, K]

    handles = [m.register_forward_pre_hook(record) for m in model.decoder.mixtures()]
    try:
        with torch.no_grad(), autocast(inputs.device):
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return seen


def choices(model: LanguageModel, inputs: torch.Tensor) -> list[torch.Tensor]:  # [B, T]
    """The experts every mixture chooses for the tokens of a batch, in block order."""
    return [chosen for _, chosen in routing(model, inputs)]  # L of [N, K]


def divergence(distributions: npt.NDArray[np.float64]) -> float:  # [domains, E]
    """Jensen-Shannon divergence of several distributions, in nats: the entropy of their mean minus their mean entropy; zero when they agree, log of their number when they share no expert. Rows are normalized, so counts do as well as distributions."""
    distributions = distributions / distributions.sum(axis=1, keepdims=True)
    mixed = stats.entropy(distributions.mean(axis=0))
    each = stats.entropy(distributions, axis=1)
    return float(mixed - np.mean(each))


def evaluate(
    run_dir: Path, *, panel_dir: Path = PANEL_DIR, batch_size: int | None = None
) -> dict[str, object]:
    """Run every slice through the run's checkpoint; write and return ``specialization.json``."""
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    shape = TIERS[settings["tier"]]
    batch_size = batch_size or shape.batch_size
    dev = device()
    model = LanguageModel(VOCAB_SIZE, shape).to(dev)
    step = checkpoint.load_model(run_dir, model)
    model.eval()
    whitespace = torch.from_numpy(np.load(panel_dir / "whitespace.npy")).to(dev)  # [V]
    num_mixtures, num_experts = len(model.decoder.mixtures()), shape.num_experts
    counts = {
        name: torch.zeros(num_mixtures, num_experts, dtype=torch.long, device=dev)
        for name in COUNTED
    }  # [L, E] each
    tokens = dict.fromkeys(COUNTED, 0)  # counted per name
    for domain, inputs in batches(panel_dir, shape, batch_size=batch_size, dev=dev):  # [B, T]
        keep = ~whitespace[inputs.flatten()]  # [N]
        for index, chosen in enumerate(choices(model, inputs)):
            counts[domain][index] += torch.bincount(chosen.flatten(), minlength=num_experts)
            if domain == "code":
                kept = chosen[keep].flatten()  # [n * K]
                counts["code_nonspace"][index] += torch.bincount(kept, minlength=num_experts)
        tokens[domain] += inputs.numel()
        if domain == "code":
            tokens["code_nonspace"] += int(keep.sum())
    usage = {
        name: (matrix.double() / tokens[name]).cpu().numpy() for name, matrix in counts.items()
    }
    mixtures = [
        {
            "share": {name: shares[index].tolist() for name, shares in usage.items()},
            "specialization": divergence(np.stack([usage[name][index] for name in DOMAINS])),
            "specialization_nonspace": divergence(
                np.stack(
                    [usage[name if name != "code" else "code_nonspace"][index] for name in DOMAINS]
                )
            ),
        }
        for index in range(num_mixtures)
    ]
    result: dict[str, object] = {
        "step": step,
        "tokens": tokens,
        "uniform": shape.num_active_experts / num_experts,
        "mixtures": mixtures,
    }
    (run_dir / FILE).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({"step": step, "specialization": [m["specialization"] for m in mixtures]}))
    return result


def main(args: list[str] | None = None) -> object:
    """The command line: ``build`` and ``evaluate``, each setting as a flag."""
    commands: dict[str, Callable[..., object]] = {"build": build, "evaluate": evaluate}
    return tyro.extras.subcommand_cli_from_dict(commands, args=args)


if __name__ == "__main__":
    main()
