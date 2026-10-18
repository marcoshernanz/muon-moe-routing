"""Tests for the token batch loader, on synthetic shards and on the real ones when present."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tfg.data import (
    DATA_DIR,
    TOKENIZER_FILE,
    VOCAB_SIZE,
    Batches,
    Windows,
    build_tokenizer,
    load_batches,
    load_shard,
    shard_paths,
    tokenizer,
)
from tfg.model import LanguageModel
from tfg.tier import LAPTOP

SEQ_LEN = 4
WINDOW = SEQ_LEN + 1
PATTERN = r"\s+\S+|\S+|\s+"  # the thesis tokenizer's chunking


def write_shards(tmp_path: Path) -> Path:
    np.save(tmp_path / "train_00000.npy", np.arange(200, dtype=np.uint16))  # 40 windows
    np.save(tmp_path / "train_00001.npy", np.arange(1000, 1004, dtype=np.uint16))  # too short
    np.save(tmp_path / "train_00002.npy", np.arange(2000, 2012, dtype=np.uint16))  # 2, tail of 2
    np.save(tmp_path / "validation_00000.npy", np.arange(3000, 3010, dtype=np.uint16))
    return tmp_path


def first_tokens(batches: Batches, steps: range) -> list[int]:
    return [int(token) for step in steps for token in batches[step][0][:, 0]]


def test_shard_paths_come_per_split_in_name_order(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    train = shard_paths(data_dir, "train")
    assert [path.name for path in train] == [
        "train_00000.npy",
        "train_00001.npy",
        "train_00002.npy",
    ]
    assert len(shard_paths(data_dir, "validation")) == 1
    with pytest.raises(FileNotFoundError, match="download"):
        shard_paths(data_dir, "test")


def test_load_shard_rejects_other_dtypes_and_shapes(tmp_path: Path) -> None:
    np.save(tmp_path / "wide.npy", np.zeros(10, dtype=np.int32))
    np.save(tmp_path / "square.npy", np.zeros((2, 5), dtype=np.uint16))
    with pytest.raises(ValueError, match="int32"):
        load_shard(tmp_path / "wide.npy")
    with pytest.raises(ValueError, match=r"\(2, 5\)"):
        load_shard(tmp_path / "square.npy")


def test_windows_never_cross_shards_and_drop_tails(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    windows = Windows(shard_paths(data_dir, "train"), WINDOW)
    assert len(windows) == 42
    assert windows[0].tolist() == [0, 1, 2, 3, 4]
    assert windows[39].tolist() == [195, 196, 197, 198, 199]
    assert windows[40].tolist() == [2000, 2001, 2002, 2003, 2004]
    assert windows[41].tolist() == [2005, 2006, 2007, 2008, 2009]
    with pytest.raises(IndexError):
        _ = windows[42]
    with pytest.raises(IndexError):
        _ = windows[-1]


def test_sequential_batches_are_inputs_and_their_next_tokens(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    batches = Batches(Windows(shard_paths(data_dir, "train"), WINDOW), 2)
    inputs, targets = batches[0]
    assert inputs.dtype == targets.dtype == torch.int64
    assert inputs.tolist() == [[0, 1, 2, 3], [5, 6, 7, 8]]
    assert targets.tolist() == [[1, 2, 3, 4], [6, 7, 8, 9]]
    assert len(batches) == 21
    epoch = list(batches)
    assert len(epoch) == 21
    assert epoch[-1][0].tolist() == [[2000, 2001, 2002, 2003], [2005, 2006, 2007, 2008]]


def test_seeded_order_uses_every_window_once_per_epoch_then_reshuffles(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    windows = Windows(shard_paths(data_dir, "train"), WINDOW)
    batches = Batches(windows, 6, seed=0)
    assert len(batches) == 7
    starts = {int(windows[i][0]) for i in range(len(windows))}
    epoch_0 = first_tokens(batches, range(7))
    epoch_1 = first_tokens(batches, range(7, 14))
    assert set(epoch_0) == starts and len(epoch_0) == 42
    assert set(epoch_1) == starts and len(epoch_1) == 42
    assert epoch_0 != epoch_1
    assert epoch_0 != first_tokens(Batches(windows, 6), range(7))


def test_batches_depend_only_on_seed_and_step(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    windows = Windows(shard_paths(data_dir, "train"), WINDOW)
    sequential = Batches(windows, 6, seed=0)
    for step in range(57):
        _ = sequential[step]
    fresh = Batches(windows, 6, seed=0)
    assert torch.equal(fresh[57][0], sequential[57][0])
    assert torch.equal(fresh[3][1], Batches(windows, 6, seed=0)[3][1])
    assert not np.array_equal(fresh.order(0), Batches(windows, 6, seed=1).order(0))


def test_batches_reject_impossible_sizes_and_negative_steps(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    windows = Windows(shard_paths(data_dir, "validation"), WINDOW)
    with pytest.raises(ValueError, match="cannot fill"):
        Batches(windows, 3)
    with pytest.raises(IndexError):
        _ = Batches(windows, 2)[-1]


def test_load_batches_reads_one_split(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    batches = load_batches("validation", 2, SEQ_LEN, data_dir=data_dir)
    assert len(batches) == 1
    inputs, targets = batches[0]
    assert inputs.tolist() == [[3000, 3001, 3002, 3003], [3005, 3006, 3007, 3008]]
    assert targets.tolist() == [[3001, 3002, 3003, 3004], [3006, 3007, 3008, 3009]]


@pytest.mark.skipif(not DATA_DIR.exists(), reason="run tfg.data.download() to fetch the shards")
def test_real_shards_feed_the_model() -> None:
    for path in shard_paths(DATA_DIR, "validation"):
        assert int(load_shard(path).max()) < VOCAB_SIZE
    batches = load_batches("validation", 8, LAPTOP.context_len, data_dir=DATA_DIR)
    inputs, targets = batches[0]
    assert inputs.shape == targets.shape == (8, LAPTOP.context_len)
    torch.manual_seed(0)
    logits = LanguageModel(VOCAB_SIZE, LAPTOP)(inputs)  # [B, T, V]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    assert abs(loss.item() - math.log(VOCAB_SIZE)) < 0.5


def test_build_tokenizer_merges_by_rank_within_chunks_and_round_trips() -> None:
    bpe = build_tokenizer(PATTERN, [[97, 98], [256, 98], [32, 97]])  # "ab", "abb", " a"
    assert bpe.n_vocab == 259
    assert bpe.encode("ab") == [256] and bpe.encode("abb") == [257]
    assert bpe.encode("abab") == [256, 256]
    assert bpe.encode(" ab") == [32, 256]  # the earlier merge goes first
    assert bpe.encode(" a b") == [258, 32, 98]  # the chunks " a" and " b"
    text = "héllo\n\n  wörld\t🎉 ab"
    ids = bpe.encode(text)
    assert bpe.decode(ids) == text and max(ids) < 259
    with pytest.raises(ValueError, match="same bytes"):
        build_tokenizer(PATTERN, [[97, 98], [97, 98]])


@pytest.mark.skipif(
    not (DATA_DIR / TOKENIZER_FILE).exists(), reason="run tfg.data.tokenizer() to fetch it"
)
def test_the_real_tokenizer_has_the_vocabulary_of_the_shards() -> None:
    bpe = tokenizer()
    assert bpe.n_vocab == VOCAB_SIZE
    text = "The Muon optimizer orthogonalizes each update.\n"
    ids = bpe.encode(text)
    assert bpe.decode(ids) == text and len(ids) < len(text) and max(ids) < VOCAB_SIZE
