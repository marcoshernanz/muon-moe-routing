"""Tests of the specialization panel: slices from a fake stream, the whitespace mask, the divergence, and the evaluation of a real short training run on a synthetic panel."""

import json
import math
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import tiktoken
import torch
from huggingface_hub.hf_file_system import HfFileSystemFile

import tfg.panel
from tfg.data import TOKENIZER_FILE, VOCAB_SIZE, build_tokenizer
from tfg.model import LanguageModel
from tfg.panel import (
    DOMAINS,
    FILE,
    WEB_SHARD,
    batches,
    build,
    choices,
    divergence,
    domain_slice,
    evaluate,
    main,
    settle,
    tokenize,
    web_slice,
    whitespace_mask,
)
from tfg.tests.synthetic import train_briefly, write_panel, write_shards
from tfg.tier import LAPTOP

PATTERN = r"\s+\S+|\S+|\s+"
MERGES = [[97, 98], [32, 32]]  # "ab" is 256, two spaces are 257


def tiny() -> tiktoken.Encoding:
    return build_tokenizer(PATTERN, MERGES)


def test_tokenize_takes_the_first_tokens_with_a_separator_after_each_text() -> None:
    bpe = tiny()
    ids = tokenize(iter(["ab", "a  b", "ab"]), bpe, 6)
    assert ids.tolist() == [256, 10, 97, 257, 98, 10] and ids.dtype == np.uint16
    assert tokenize(iter(["ab", "ab"]), bpe, 3).tolist() == [256, 10, 256]
    with pytest.raises(ValueError, match="fewer"):
        tokenize(iter(["ab"]), bpe, 6)


def test_domain_slice_closes_the_stream_it_leaves_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    def stream(domain: str) -> Generator[str]:  # a generator, like the streamed documents
        try:
            yield from [f"{domain} ab"] * 5
        finally:
            closed.append(domain)

    monkeypatch.setattr(tfg.panel, "documents", stream)
    ids = domain_slice("math", tiny(), 8)
    assert ids.tolist() == [109, 97, 116, 104, 32, 256, 10, 109]
    assert closed == ["math"]  # closed on return, not left to the garbage collector


def test_settle_waits_for_a_hub_file_to_go_and_gives_up_at_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tfg.panel, "SETTLE_SECONDS", 0.3)
    monkeypatch.setattr(tfg.panel, "SETTLE_POLL", 0.05)
    settle()  # nothing to wait for
    lingering = object.__new__(HfFileSystemFile)  # a reader still holding a file
    with pytest.raises(TimeoutError, match="still busy"):
        settle()
    del lingering
    settle()


def test_the_whitespace_mask_marks_the_ids_that_decode_to_whitespace_alone() -> None:
    mask = whitespace_mask(tiny())
    assert mask.shape == (258,) and mask.dtype == np.bool_
    assert mask[[9, 10, 32, 257]].all()
    assert not mask[[97, 256, 0xC2]].any()  # a lone continuation byte is not text
    assert mask.sum() == 11  # five controls, the separators 28 to 31, the space, and the pair


def test_the_web_slice_is_the_head_of_the_held_out_shard(tmp_path: Path) -> None:
    np.save(tmp_path / WEB_SHARD, np.arange(100, dtype=np.uint16))
    assert web_slice(tmp_path, 50).tolist() == list(range(50))
    with pytest.raises(ValueError, match="fewer"):
        web_slice(tmp_path, 200)


def test_divergence_is_zero_when_the_domains_agree_and_log_d_when_they_share_nothing() -> None:
    assert divergence(np.tile([0.5, 0.25, 0.25, 0.0], (4, 1))) == pytest.approx(0.0)
    assert divergence(np.eye(4)) == pytest.approx(math.log(4))
    halves = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]) / 2
    assert divergence(halves) == pytest.approx(math.log(2))
    counts = np.array([[8, 2], [2, 8]])  # rows are normalized
    assert divergence(counts) == pytest.approx(divergence(counts / 10))


def test_build_writes_matched_slices_and_the_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    artifact = {"version": 1, "split_pattern": PATTERN, "merge_pairs": MERGES}
    (data_dir / TOKENIZER_FILE).write_text(json.dumps(artifact), encoding="utf-8")
    np.save(data_dir / WEB_SHARD, np.arange(100, dtype=np.uint16))
    monkeypatch.setattr(tfg.panel, "documents", lambda domain: (t for t in [f"{domain} ab"] * 5))
    panel_dir = build(tmp_path / "panel", tokens=8, data_dir=data_dir)
    for domain in DOMAINS:
        ids = np.load(panel_dir / f"{domain}.npy")
        assert ids.shape == (8,) and ids.dtype == np.uint16
    assert np.load(panel_dir / "web.npy").tolist() == list(range(8))
    assert np.load(panel_dir / "math.npy").tolist() == [109, 97, 116, 104, 32, 256, 10, 109]
    mask = np.load(panel_dir / "whitespace.npy")
    assert mask.shape == (258,) and mask[32] and not mask[97]


def test_batches_serve_every_domain_in_windows_of_the_context(tmp_path: Path) -> None:
    panel_dir = write_panel(tmp_path, tokens=200)  # three windows of 64 per domain, 8 left over
    served = list(batches(panel_dir, LAPTOP, batch_size=2, dev=torch.device("cpu")))
    assert [domain for domain, _ in served] == [d for d in DOMAINS for _ in range(2)]
    assert [tuple(inputs.shape) for _, inputs in served] == [(2, 64), (1, 64)] * len(DOMAINS)
    assert all(inputs.dtype == torch.int64 for _, inputs in served)
    assert served[0][1].flatten().tolist() == np.load(panel_dir / "web.npy")[:128].tolist()
    np.save(panel_dir / "wiki.npy", np.load(panel_dir / "wiki.npy")[:-1])
    with pytest.raises(ValueError, match="differ in length"):
        next(batches(panel_dir, LAPTOP, batch_size=2, dev=torch.device("cpu")))


def test_choices_reports_every_mixture_and_leaves_no_hook_behind() -> None:
    torch.manual_seed(0)
    model = LanguageModel(VOCAB_SIZE, LAPTOP).train()
    inputs = torch.randint(VOCAB_SIZE, (2, LAPTOP.context_len))  # [B, T]
    chosen = choices(model, inputs)
    assert len(chosen) == LAPTOP.num_blocks - LAPTOP.dense_blocks
    for tensor, mixture in zip(chosen, model.decoder.mixtures()):
        assert tensor.shape == (2 * LAPTOP.context_len, LAPTOP.num_active_experts)
        logits = mixture.router_logits  # the training forward kept them
        assert logits is not None
        selection = (torch.sigmoid(logits) + mixture.router_bias).topk(LAPTOP.num_active_experts)
        assert torch.equal(tensor, selection.indices)
    model(inputs)  # a plain forward afterwards records nothing more: the hooks are gone
    assert len(chosen) == LAPTOP.num_blocks - LAPTOP.dense_blocks


def test_evaluate_counts_where_a_trained_run_sends_each_domain(tmp_path: Path) -> None:
    panel_dir = write_panel(tmp_path)
    run_dir = train_briefly(tmp_path / "run", write_shards(tmp_path))
    result = evaluate(run_dir, panel_dir=panel_dir)
    assert result == json.loads((run_dir / FILE).read_text(encoding="utf-8"))
    assert result["step"] == 10
    assert result["uniform"] == LAPTOP.num_active_experts / LAPTOP.num_experts
    tokens = result["tokens"]
    assert isinstance(tokens, dict)
    code = np.load(panel_dir / "code.npy")
    mask = np.load(panel_dir / "whitespace.npy")
    assert tokens["web"] == tokens["code"] == 1024
    assert tokens["code_nonspace"] == int((~mask[code]).sum()) < 700
    mixtures = result["mixtures"]
    assert isinstance(mixtures, list) and len(mixtures) == LAPTOP.num_blocks - LAPTOP.dense_blocks
    for mixture in mixtures:
        for share in mixture["share"].values():
            assert len(share) == LAPTOP.num_experts and min(share) >= 0
            assert sum(share) == pytest.approx(LAPTOP.num_active_experts)
        assert 0 <= mixture["specialization"] <= math.log(len(DOMAINS))
        assert 0 <= mixture["specialization_nonspace"] <= math.log(len(DOMAINS))
    assert any(m["share"]["code"] != m["share"]["code_nonspace"] for m in mixtures)
    again = main(
        ["evaluate", "--run-dir", str(run_dir), "--panel-dir", str(panel_dir), "--batch-size", "8"]
    )
    assert again == result
    np.save(panel_dir / "wiki.npy", np.load(panel_dir / "wiki.npy")[:-1])
    with pytest.raises(ValueError, match="differ in length"):
        evaluate(run_dir, panel_dir=panel_dir)
