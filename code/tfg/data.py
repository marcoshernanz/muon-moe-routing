"""Token batches from the pre-tokenized FineWeb-Edu shards, and the tokenizer that made them.

A shard is a one-dimensional ``.npy`` array of ``uint16`` token ids. A split is cut into consecutive windows of ``seq_len + 1`` tokens that never cross a shard boundary; the tail of a shard shorter than one window is dropped. In seeded order every epoch is a fresh permutation of the windows derived from ``(seed, epoch)``, so the batch at any step is a pure function of the seed and the step, and resuming a run needs no loader state.

Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    S   shards in a split
    W   windows in a split
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tiktoken
import torch
from huggingface_hub import hf_hub_download

VOCAB_SIZE = 16_384
TOKEN_DTYPE = np.uint16
HF_REPO = "marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full"
TOKENIZER_FILE = "fineweb_edu_sample10bt_bpe_16384.json"  # the split pattern and the merges
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fineweb-edu"


def download(
    data_dir: Path = DATA_DIR,
    *,
    train_shards: int | None = None,
    validation_shards: int | None = None,
) -> Path:
    """Fetch the shards from the Hub into ``data_dir``, all of them unless a count is given.

    Files are fetched by name from the counts in ``metadata.json``, and files already present are skipped, so calling it again is cheap.
    """
    metadata = json.loads(download_file("metadata.json", data_dir).read_text(encoding="utf-8"))
    for split, count in (("train", train_shards), ("validation", validation_shards)):
        total: int = metadata["shards"][split]
        for index in range(total if count is None else min(count, total)):
            download_file(f"{split}_{index:05d}.npy", data_dir)
    return data_dir


def download_file(name: str, data_dir: Path) -> Path:
    """Fetch one file of the Hub dataset into ``data_dir``; shards never change, so a present file is taken as complete without a network check."""
    path = data_dir / name
    if not path.exists():
        hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=data_dir)
    return path


def build_tokenizer(split_pattern: str, merge_pairs: list[list[int]]) -> tiktoken.Encoding:
    """A byte-level BPE from its merge list: ids 0 to 255 are the bytes, the rest one per merge in order. Text is split by ``split_pattern`` into chunks merged independently."""
    pieces = [bytes([byte]) for byte in range(256)]  # the bytes of each id [V]
    for left, right in merge_pairs:
        pieces.append(pieces[left] + pieces[right])
    ranks = {piece: index for index, piece in enumerate(pieces)}
    if len(ranks) != len(pieces):
        raise ValueError("two merges yield the same bytes, which the library cannot tell apart")
    return tiktoken.Encoding("tfg", pat_str=split_pattern, mergeable_ranks=ranks, special_tokens={})


def tokenizer(data_dir: Path = DATA_DIR) -> tiktoken.Encoding:
    """The tokenizer the shards were made with, fetched from the Hub beside them."""
    artifact = json.loads(download_file(TOKENIZER_FILE, data_dir).read_text(encoding="utf-8"))
    return build_tokenizer(artifact["split_pattern"], artifact["merge_pairs"])


def shard_paths(data_dir: Path, split: str) -> list[Path]:
    """The shard files of one split, in name order."""
    paths = sorted(data_dir.glob(f"{split}_*.npy"))
    if not paths:
        raise FileNotFoundError(
            f"no {split}_*.npy shards in {data_dir}; run tfg.data.download() first"
        )
    return paths


def load_shard(path: Path) -> npt.NDArray[np.uint16]:
    """Memory-map one shard, reading nothing from disk until a window is taken."""
    tokens = np.load(path, mmap_mode="r")
    if tokens.ndim != 1 or tokens.dtype != TOKEN_DTYPE:
        raise ValueError(
            f"{path}: expected a 1-D {TOKEN_DTYPE.__name__} array, got {tokens.dtype}{tokens.shape}"
        )
    return tokens


class Windows:
    """Every window of ``window_len`` tokens in a split, addressable by index."""

    def __init__(self, paths: list[Path], window_len: int):
        self.window_len = window_len
        self.shards = [load_shard(path) for path in paths]  # S memory maps
        counts = [len(shard) // window_len for shard in self.shards]  # [S]
        self.first = np.cumsum([0, *counts])  # index of each shard's first window [S+1]

    def __len__(self) -> int:
        return int(self.first[-1])  # W

    def __getitem__(self, index: int) -> npt.NDArray[np.uint16]:  # [T+1]
        if not 0 <= index < len(self):
            raise IndexError(f"window {index} of {len(self)}")
        shard = int(np.searchsorted(self.first, index, side="right")) - 1
        start = (index - int(self.first[shard])) * self.window_len
        return self.shards[shard][start : start + self.window_len]  # [T+1]


class Batches:
    """Batches of ``(inputs, targets)`` from a window store, each ``[B, T]`` of int64.

    With a seed, ``batches[step]`` follows a per-epoch permutation of the windows and wraps into the next epoch when the step runs past one; without a seed the windows come in file order. Iterating yields exactly one epoch.
    """

    def __init__(self, windows: Windows, batch_size: int, *, seed: int | None = None):
        if len(windows) < batch_size:
            raise ValueError(f"{len(windows)} windows cannot fill a batch of {batch_size}")
        self.windows = windows
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = -1
        self._order = np.empty(0, dtype=np.int64)  # [W]

    def __len__(self) -> int:
        return len(self.windows) // self.batch_size  # batches per epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return (self[step] for step in range(len(self)))

    def __getitem__(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:  # [B, T] each
        if step < 0:
            raise IndexError(f"step {step}")
        epoch, offset = divmod(step, len(self))
        if epoch != self._epoch:
            self._epoch, self._order = epoch, self.order(epoch)
        ids = self._order[offset * self.batch_size : (offset + 1) * self.batch_size]  # [B]
        windows = np.stack([self.windows[i] for i in ids.tolist()], dtype=np.int64)  # [B, T+1]
        tokens = torch.from_numpy(windows)  # [B, T+1]
        return tokens[:, :-1], tokens[:, 1:]  # [B, T] each

    def order(self, epoch: int) -> npt.NDArray[np.int64]:  # [W]
        """The window indices of one epoch: a seeded permutation, or file order without a seed."""
        if self.seed is None:
            return np.arange(len(self.windows))  # [W]
        return np.random.default_rng([self.seed, epoch]).permutation(len(self.windows))  # [W]


def load_batches(
    split: str,
    batch_size: int,
    seq_len: int,
    *,
    seed: int | None = None,
    data_dir: Path = DATA_DIR,
) -> Batches:
    """Batches of ``seq_len`` tokens and their next-token targets from one split."""
    return Batches(Windows(shard_paths(data_dir, split), seq_len + 1), batch_size, seed=seed)


if __name__ == "__main__":
    print(download())
