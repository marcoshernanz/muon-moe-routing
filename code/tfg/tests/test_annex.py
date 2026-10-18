"""Tests of the annex readings: each on hand-made inputs, then all of them end to end on a real short training run against the synthetic panel."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import tfg.annex
from tfg import checkpoint
from tfg.annex import (
    FILE,
    alignment,
    coactivation,
    cosines,
    evaluate,
    main,
    off_diagonal,
    onehot,
    purity,
    row_cosines,
    sample,
    saturation,
    similarity,
)
from tfg.model import MixtureOfExperts
from tfg.panel import DOMAINS
from tfg.tests.synthetic import train_briefly, write_panel, write_shards
from tfg.tier import LAPTOP


def test_saturation_is_the_overlap_of_the_chosen_sets_over_k() -> None:
    early = onehot(torch.tensor([[0, 1], [2, 3], [0, 3]]), 4)  # [N, E]
    final = onehot(torch.tensor([[1, 0], [0, 1], [3, 2]]), 4)  # [N, E]
    assert early.tolist() == [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]]
    assert saturation(early, final) == pytest.approx((1 + 0 + 0.5) / 3)  # sets: no order within
    assert saturation(final, final) == 1.0


def test_coactivation_is_conditional_on_the_row_expert() -> None:
    pairs = torch.tensor(
        [[10, 4, 0], [4, 8, 2], [0, 2, 5]]
    )  # tokens sent to both, each on the diagonal
    matrix = coactivation(pairs)
    assert matrix.tolist() == [[0, 0.4, 0], [0.5, 0, 0.25], [0, 0.4, 0]]
    assert off_diagonal(matrix).tolist() == [0.4, 0, 0.5, 0.25, 0, 0.4]
    assert coactivation(torch.zeros(2, 2)).tolist() == [[0, 0], [0, 0]]  # experts without tokens


def test_purity_is_the_share_of_each_experts_tokens_from_its_main_domain() -> None:
    counts = np.array([[10, 0, 3], [0, 10, 3], [0, 0, 0], [0, 0, 6]], dtype=float)  # [G, E]
    assert purity(counts) == pytest.approx((1 + 1 + 0.5) / 3)
    assert purity(np.ones((4, 3))) == pytest.approx(0.25)  # every domain routed alike: chance
    counts[:, 2] = 0  # an expert that received nothing is left out
    assert purity(counts) == 1.0


def test_cosines_average_over_token_pairs_and_leave_a_tokens_self_out() -> None:
    # Domain a holds (1, 0) and (0, 1); domain b holds (1, 0) twice: their unit sums and counts.
    units = torch.tensor([[1.0, 1.0], [2.0, 0.0]], dtype=torch.double)  # [G, D]
    counts = torch.tensor([2.0, 2.0], dtype=torch.double)  # [G]
    assert np.allclose(cosines(units, counts), [[0.0, 0.5], [0.5, 1.0]])
    usage = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [G, E]
    half = math.sqrt(0.5)
    assert np.allclose(row_cosines(usage), [[1, 0, half], [0, 1, half], [half, half, 1]])


def test_similarity_is_one_between_twin_experts() -> None:
    torch.manual_seed(0)
    mixture = MixtureOfExperts(LAPTOP)
    mixture.experts[1].load_state_dict(mixture.experts[0].state_dict())
    tokens = torch.randn(50, LAPTOP.d_model)  # [S, D]
    matrix = similarity(mixture, tokens)
    assert matrix.shape == (LAPTOP.num_experts, LAPTOP.num_experts)
    assert np.allclose(matrix, matrix.T, atol=1e-6) and np.allclose(matrix.diagonal(), 1.0)
    assert matrix[0, 1] == pytest.approx(1.0) and matrix[0, 2] < matrix[0, 1]


def test_alignment_is_one_along_the_router_direction_and_zero_across() -> None:
    router = torch.tensor([1.0, 0.0, 0.0])  # [D]
    along = torch.outer(torch.tensor([2.0, -1.0]), router)  # [R, D]
    assert alignment(router, along) == pytest.approx(1.0)
    across = torch.outer(torch.tensor([2.0, -1.0]), torch.tensor([0.0, 1.0, 0.0]))  # [R, D]
    assert alignment(router, across) == 0.0
    assert alignment(router, torch.eye(3)) == pytest.approx(1 / math.sqrt(3))  # nothing special
    assert alignment(router, torch.zeros(2, 3)) is None
    assert alignment(torch.zeros(3), along) is None


def test_sample_spreads_at_most_the_sample_size_over_the_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tfg.annex, "SAMPLE_TOKENS", 4)
    rows = torch.arange(10.0)[:, None]  # [N, 1]
    taken = sample(rows)
    assert taken.flatten().tolist() == [0.0, 3.0, 6.0, 9.0]
    assert sample(rows[:3]).flatten().tolist() == [0.0, 1.0, 2.0]
    taken[0] = 99.0  # its own tensor: the rows are not held onto
    assert rows[0].item() == 0.0


def test_evaluate_reads_the_panel_through_the_end_and_the_early_checkpoint(
    tmp_path: Path,
) -> None:
    panel_dir = write_panel(tmp_path)
    run_dir = train_briefly(tmp_path / "run", write_shards(tmp_path))
    result = evaluate(run_dir, panel_dir=panel_dir)
    assert result == json.loads((run_dir / FILE).read_text(encoding="utf-8"))
    assert result["step"] == 10 and result["early_step"] == 1  # one percent of ten steps: one
    assert result["tokens"] == dict.fromkeys(DOMAINS, 1024)
    assert (
        result["sample"] == len(DOMAINS) * LAPTOP.batch_size * LAPTOP.context_len
    )  # first batches
    assert result["chance"] == {
        "coactivation": 3 / 15,
        "routing_purity": 0.25,
        "gradient_alignment": pytest.approx(1 / math.sqrt(LAPTOP.d_model)),
    }
    mixtures = result["mixtures"]
    assert isinstance(mixtures, list) and len(mixtures) == LAPTOP.num_blocks - LAPTOP.dense_blocks
    experts, domains = LAPTOP.num_experts, len(DOMAINS)
    for m in mixtures:
        assert 0 <= m["saturation"] <= 1
        pairs = np.array(m["coactivation"])  # [E, E]
        assert pairs.shape == (experts, experts) and not pairs.diagonal().any()
        assert (pairs >= 0).all() and (pairs <= 1).all()
        assert 0 <= m["coactivation_mean"] <= m["coactivation_max"] <= 1
        assert 0.25 <= m["routing_purity"] <= 1
        hidden = np.array(m["hidden_cosine"])  # [G, G]
        assert hidden.shape == (domains, domains) and np.allclose(hidden, hidden.T)
        assert (np.abs(hidden) <= 1 + 1e-9).all()
        assert m["hidden_cosine_between"] == pytest.approx(off_diagonal(hidden).mean())
        assert m["hidden_cosine_within"] == pytest.approx(hidden.diagonal().mean())
        routed = np.array(m["routing_cosine"])  # [G, G]
        assert np.allclose(routed.diagonal(), 1) and (routed >= 0).all()
        assert m["routing_cosine_between"] == pytest.approx(off_diagonal(routed).mean())
        similar = np.array(m["expert_similarity"])  # [E, E]
        assert similar.shape == (experts, experts) and np.allclose(similar.diagonal(), 1)
        assert np.allclose(similar, similar.T) and -1 <= m["expert_similarity_mean"] <= 1
        aligned = m["gradient_alignment"]
        assert len(aligned) == experts and all(a is None or 0 <= a <= 1 for a in aligned)
        assert 0 <= m["gradient_alignment_mean"] <= 1
    assert any(m["saturation"] < 1 for m in mixtures)  # the routing moved after the first step
    again = main(["--run-dir", str(run_dir), "--panel-dir", str(panel_dir), "--batch-size", "4"])
    assert again["step"] == 10 and again["sample"] == domains * 4 * LAPTOP.context_len
    smaller = again["mixtures"]
    assert isinstance(smaller, list)
    for one, other in zip(mixtures, smaller):  # counted over the whole panel: the same
        assert one["saturation"] == other["saturation"]
        assert one["coactivation"] == other["coactivation"]
        assert one["routing_purity"] == other["routing_purity"]
        assert one["hidden_cosine_between"] == pytest.approx(other["hidden_cosine_between"])
    shutil.copy(run_dir / checkpoint.FILE, run_dir / checkpoint.EARLY_FILE)  # the end itself
    same = evaluate(run_dir, panel_dir=panel_dir)
    saturated = same["mixtures"]
    assert isinstance(saturated, list)
    assert same["early_step"] == 10 and all(m["saturation"] == 1.0 for m in saturated)
    (run_dir / checkpoint.EARLY_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="early.pt"):
        evaluate(run_dir, panel_dir=panel_dir)
