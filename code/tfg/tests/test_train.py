"""Tests of the training loop on a synthetic, predictable token stream; seconds on a laptop."""

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from tfg import checkpoint
from tfg.data import VOCAB_SIZE, load_batches
from tfg.model import BIAS_UPDATE_RATE, LanguageModel
from tfg.optim import GROUPS, build
from tfg.telemetry import GAINS
from tfg.tests.synthetic import write_shards
from tfg.tier import LAPTOP
from tfg.train import (
    ADAMW_BETA2,
    FLOOR,
    Record,
    arm,
    autocast,
    cross_entropy,
    diverged,
    evaluate,
    lr_factor,
    main,
    train,
    train_step,
)


def quick_run(
    tmp_path: Path,
    *,
    steps: int,
    seed: int = 0,
    partition: str = "adamw",
    run_dir: Path | None = None,
    resume: bool = False,
    schedule: str = "cosine",
    compiled: bool = False,
    balanced: bool = True,
    router_lr_multiplier: float = 1.0,
    lr_multiplier: float = 1.0,
    muon_lr_multiplier: float = 1.0,
    adamw_clip: float = 1.0,
    muon_clip: float = 1.0,
    balancer_multiplier: float = 1.0,
    adamw_beta2: float = ADAMW_BETA2,
    balance_loss: float = 0.0,
    recompute: int = 0,
    checkpoint_every: int = 10,
) -> list[Record]:
    return train(
        tier="laptop",
        steps=steps,
        schedule=schedule,
        compiled=compiled,
        recompute=recompute,
        balanced=balanced,
        router_lr_multiplier=router_lr_multiplier,
        lr_multiplier=lr_multiplier,
        muon_lr_multiplier=muon_lr_multiplier,
        adamw_clip=adamw_clip,
        muon_clip=muon_clip,
        balancer_multiplier=balancer_multiplier,
        adamw_beta2=adamw_beta2,
        balance_loss=balance_loss,
        batch_size=8,
        partition=partition,
        adamw_lr=3e-3,
        muon_lr=3e-3,
        warmup_steps=5,
        log_every=10,
        eval_every=20,
        eval_batches=2,
        checkpoint_every=checkpoint_every,
        seed=seed,
        run_dir=run_dir or tmp_path / f"{partition}-{seed}",
        data_dir=write_shards(tmp_path),
        resume=resume,
    )


def records_of(path: Path) -> list[dict[str, object]]:
    """The records of a metrics file, without the timing that no two runs share."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "tokens_per_s"} for line in lines]


def test_lr_factor_warms_up_then_holds_or_decays_to_the_floor() -> None:
    def constant(step: int) -> float:
        return lr_factor(step, steps=100, warmup_steps=10, schedule="constant")

    def cosine(step: int) -> float:
        return lr_factor(step, steps=100, warmup_steps=10, schedule="cosine")

    assert constant(0) == pytest.approx(0.1)
    assert constant(9) == constant(99) == pytest.approx(1.0)
    assert cosine(0) == pytest.approx(0.1) and cosine(9) == pytest.approx(1.0)
    assert cosine(54) == pytest.approx(0.55)  # halfway down, the middle of one and the floor
    assert cosine(99) == pytest.approx(FLOOR)
    factors = [cosine(step) for step in range(100)]
    assert factors[:10] == sorted(factors[:10])
    assert factors[9:] == sorted(factors[9:], reverse=True)


def test_the_logged_factors_follow_the_schedule(tmp_path: Path) -> None:
    records = quick_run(tmp_path, steps=20)
    expected = [lr_factor(s - 1, steps=20, warmup_steps=5, schedule="cosine") for s in (10, 20)]
    assert [r["lr_factor"] for r in records] == expected
    assert records[-1]["lr_factor"] == pytest.approx(FLOOR)


def test_bad_schedules_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schedule 'linear'"):
        train(tier="laptop", schedule="linear", run_dir=tmp_path, data_dir=write_shards(tmp_path))
    with pytest.raises(ValueError, match="warmup_steps 5 below steps 5"):
        train(tier="laptop", steps=5, warmup_steps=5, run_dir=tmp_path, data_dir=tmp_path)


def test_evaluate_leaves_the_model_training_and_its_loads_alone(tmp_path: Path) -> None:
    batches = load_batches("validation", 2, LAPTOP.context_len, data_dir=write_shards(tmp_path))
    model = LanguageModel(VOCAB_SIZE, LAPTOP)
    loss = evaluate(model, batches, 2)
    assert abs(loss - math.log(VOCAB_SIZE)) < 0.5
    assert model.training
    assert all(int(m.expert_load.sum()) == 0 for m in model.decoder.mixtures())


def test_train_step_moves_the_balancer_biases_and_scales_every_rate(tmp_path: Path) -> None:
    batches = load_batches("train", 8, LAPTOP.context_len, data_dir=write_shards(tmp_path))
    torch.manual_seed(0)
    model = LanguageModel(VOCAB_SIZE, LAPTOP)
    optimizers = build(model, "muon", adamw_lr=1e-3, muon_lr=2e-3)
    inputs, targets = batches[0]
    with autocast(inputs.device):  # the norm over every parameter, as train_step sees it
        cross_entropy(model(inputs), targets).backward()
    total = nn.utils.clip_grad_norm_(model.parameters(), math.inf).item()
    model.zero_grad(set_to_none=True)
    loss, norms = train_step(model, optimizers, batches[0], factor=0.5)
    assert math.isfinite(loss) and list(norms) == ["adamw", "muon"]
    assert all(norm > 0 for norm in norms.values())
    assert math.hypot(*norms.values()) == pytest.approx(total, rel=1e-4)  # each once
    lrs = [g["lr"] for o in optimizers.values() for g in o.param_groups]
    assert lrs == [5e-4, 5e-4, 1e-3, 1e-3]
    for mixture in model.decoder.mixtures():
        assert int(mixture.expert_load.sum()) == 8 * LAPTOP.context_len * LAPTOP.num_active_experts
        assert mixture.router_bias.abs().max().item() == pytest.approx(BIAS_UPDATE_RATE)

    torch.manual_seed(0)  # the same model, its balancer four times as fast
    faster = LanguageModel(VOCAB_SIZE, LAPTOP)
    train_step(
        faster,
        build(faster, "muon", adamw_lr=1e-3, muon_lr=2e-3),
        batches[0],
        factor=0.5,
        balancer_multiplier=4.0,
    )
    for mixture, slow in zip(faster.decoder.mixtures(), model.decoder.mixtures()):
        assert torch.equal(mixture.router_bias, 4 * slow.router_bias)  # signs alike, step 4x


def test_train_learns_the_stream_and_logs_it(tmp_path: Path) -> None:
    partition = "adamw"
    records = quick_run(tmp_path, steps=60, partition=partition)
    assert [r["step"] for r in records] == [10, 20, 30, 40, 50, 60]
    assert records[-1]["loss"] < 0.5 * records[0]["loss"] < math.log(VOCAB_SIZE)
    assert [("val_loss" in r) for r in records] == [False, True, False, True, False, True]
    assert "val_loss" in records[-1] and "val_loss" in records[1]
    assert records[-1]["val_loss"] < records[1]["val_loss"]
    assert all(r["tokens_per_s"] > 0 and math.isfinite(r["grad_norm"]) for r in records)
    for r in records:  # every step since the previous log step, this one last
        assert list(r["grad_norms"]) == ["adamw"] and len(r["grad_norms"]["adamw"]) == 10
        assert r["grad_norms"]["adamw"][-1] == pytest.approx(r["grad_norm"])
        assert all(math.isfinite(norm) and norm > 0 for norm in r["grad_norms"]["adamw"])
        assert len(r["losses"]) == 10 and r["losses"][-1] == r["loss"]
        assert all(math.isfinite(loss) for loss in r["losses"])

    run_dir = tmp_path / f"{partition}-0"
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["steps"] == 60 and settings["partition"] == partition
    routing = (run_dir / "routing.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in routing] == [r["step"] for r in records]
    assert all(
        len(json.loads(line)["mixtures"]) == LAPTOP.num_blocks - LAPTOP.dense_blocks
        for line in routing
    )


def test_an_unbalanced_run_leaves_the_biases_at_zero_and_still_learns(tmp_path: Path) -> None:
    batches = load_batches("train", 8, LAPTOP.context_len, data_dir=write_shards(tmp_path))
    torch.manual_seed(0)
    model = LanguageModel(VOCAB_SIZE, LAPTOP)
    train_step(
        model,
        build(model, "adamw", adamw_lr=1e-3, muon_lr=1e-3),
        batches[0],
        factor=1.0,
        balanced=False,
    )
    assert all(not mixture.router_bias.any() for mixture in model.decoder.mixtures())

    run_dir = tmp_path / "unbalanced"
    records = quick_run(tmp_path, steps=30, run_dir=run_dir, balanced=False)
    assert records[-1]["loss"] < 0.5 * records[0]["loss"]
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["balanced"] is False and settings["router_lr_multiplier"] == 1.0
    for line in (run_dir / "routing.jsonl").read_text(encoding="utf-8").splitlines():
        for reading in json.loads(line)["mixtures"]:
            assert reading["bias_span"] == 0.0 and not any(reading["bias"])
            assert 1.0 <= reading["usage_perplexity"] <= LAPTOP.num_experts
    with pytest.raises(ValueError, match="settings differ"):  # the balancer is part of the arm
        quick_run(tmp_path, steps=30, run_dir=run_dir, resume=True)


def test_the_router_rate_multiplier_is_recorded_and_names_the_arm(tmp_path: Path) -> None:
    run_dir = tmp_path / "router"
    records = quick_run(tmp_path, steps=30, run_dir=run_dir, router_lr_multiplier=2.0)
    assert records[-1]["loss"] < 0.5 * records[0]["loss"]
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["router_lr_multiplier"] == 2.0 and settings["balanced"] is True
    with pytest.raises(ValueError, match="settings differ.*router_lr_multiplier"):  # named
        quick_run(tmp_path, steps=30, run_dir=run_dir, resume=True)
    with pytest.raises(ValueError, match="must be positive"):
        quick_run(tmp_path, steps=20, run_dir=tmp_path / "bad", router_lr_multiplier=-1.0)

    assert arm("adamw", balanced=True, router_lr_multiplier=1.0) == "adamw"
    assert arm("adamw", balanced=False, router_lr_multiplier=1.0) == "adamw-unbalanced"
    assert arm("adamw", balanced=True, router_lr_multiplier=0.5) == "adamw-router0.5x"
    assert arm("muon", balanced=False, router_lr_multiplier=4.0) == "muon-unbalanced-router4x"


def test_the_second_moment_decay_is_recorded_and_names_the_arm(tmp_path: Path) -> None:
    run_dir = tmp_path / "beta2"
    records = quick_run(tmp_path, steps=30, run_dir=run_dir, adamw_beta2=0.98)
    assert records[-1]["loss"] < 0.5 * records[0]["loss"]
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["adamw_beta2"] == 0.98
    state = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    assert all(g["betas"] == (0.9, 0.98) for g in state["optimizers"]["adamw"]["param_groups"])
    default = quick_run(tmp_path, steps=30, run_dir=tmp_path / "default")
    assert [r["loss"] for r in records] != [r["loss"] for r in default]  # the decay took effect
    with pytest.raises(ValueError, match="settings differ"):  # the decay is part of the arm
        quick_run(tmp_path, steps=30, run_dir=run_dir, resume=True)
    with pytest.raises(ValueError, match="must lie in"):
        quick_run(tmp_path, steps=20, run_dir=tmp_path / "bad", adamw_beta2=1.0)

    assert arm("adamw", balanced=True, router_lr_multiplier=1.0, adamw_beta2=0.98) == (
        "adamw-beta20.98"
    )
    assert arm("adamw", balanced=True, router_lr_multiplier=1.0, adamw_beta2=0.95) == "adamw"
    assert arm(
        "muon", balanced=False, router_lr_multiplier=1.0, lr_multiplier=2.0, adamw_beta2=0.98
    ) == ("muon-unbalanced-lr2x-beta20.98")


def test_the_balance_loss_is_recorded_logged_and_names_the_arm(tmp_path: Path) -> None:
    run_dir = tmp_path / "loss"
    records = quick_run(tmp_path, steps=30, partition="muon", run_dir=run_dir, balance_loss=0.01)
    assert records[-1]["loss"] < 0.5 * records[0]["loss"]
    mixtures = LAPTOP.num_blocks - LAPTOP.dense_blocks
    for record in records:  # about one per mixture, before its weight
        assert 0.9 * mixtures < record["sequence_balance"] < 2 * mixtures
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["balance_loss"] == 0.01
    default = quick_run(tmp_path, steps=30, partition="muon", run_dir=tmp_path / "default")
    assert all("sequence_balance" not in record for record in default)
    assert [r["loss"] for r in records] != [r["loss"] for r in default]  # the loss took effect
    with pytest.raises(ValueError, match="settings differ.*balance_loss"):  # part of the arm
        quick_run(tmp_path, steps=30, partition="muon", run_dir=run_dir, resume=True)
    with pytest.raises(ValueError, match="zero or positive"):
        quick_run(tmp_path, steps=20, run_dir=tmp_path / "bad", balance_loss=-1.0)

    assert arm("muon", balanced=True, router_lr_multiplier=1.0, balance_loss=1e-4) == (
        "muon-balanceloss0.0001"
    )
    assert arm("muon", balanced=True, router_lr_multiplier=1.0, balance_loss=0.0) == "muon"


def test_train_step_adds_the_balance_loss_to_the_gradient_and_returns_the_cross_entropy(
    tmp_path: Path,
) -> None:
    data = load_batches("train", 8, LAPTOP.context_len, seed=0, data_dir=write_shards(tmp_path))
    steps: dict[float, tuple[float, LanguageModel]] = {}
    for weight in (0.0, 1.0):
        torch.manual_seed(0)
        net = LanguageModel(VOCAB_SIZE, LAPTOP, balance_loss=True)
        optimizers = build(net, "adamw", adamw_lr=1e-3, muon_lr=1e-3)
        loss, _ = train_step(net, optimizers, data[0], factor=1.0, balance_loss=weight)
        steps[weight] = loss, net
    (plain, plain_net), (weighted, weighted_net) = steps[0.0], steps[1.0]
    assert weighted == plain  # the same forward: the cross-entropy alone comes back
    router = [
        (a.router.weight, b.router.weight)
        for a, b in zip(plain_net.decoder.mixtures(), weighted_net.decoder.mixtures())
    ]
    assert not all(torch.equal(a, b) for a, b in router)  # the loss moved the routers


def test_the_balance_loss_with_recompute_gives_the_same_run_bit_for_bit(tmp_path: Path) -> None:
    kept_dir, recomputed_dir = tmp_path / "kept", tmp_path / "recomputed"
    quick_run(tmp_path, steps=20, partition="muon", run_dir=kept_dir, balance_loss=0.01)
    quick_run(
        tmp_path, steps=20, partition="muon", run_dir=recomputed_dir, balance_loss=0.01, recompute=1
    )
    assert records_of(recomputed_dir / "metrics.jsonl") == records_of(kept_dir / "metrics.jsonl")


def test_the_rate_multiplier_scales_both_rates_and_each_optimizer_clips_on_its_own(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "stress"
    records = quick_run(
        tmp_path,
        steps=20,
        partition="muon",
        run_dir=run_dir,
        schedule="constant",
        lr_multiplier=2.0,
        adamw_clip=0.5,
        muon_clip=4.0,
        balancer_multiplier=2.0,
    )
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["lr_multiplier"] == 2.0 and settings["schedule"] == "constant"
    assert settings["adamw_clip"] == 0.5 and settings["muon_clip"] == 4.0
    assert settings["balancer_multiplier"] == 2.0
    assert settings["adamw_lr"] == settings["muon_lr"] == 3e-3  # the tuned rates, unscaled
    for r in records:
        assert list(r["grad_norms"]) == ["adamw", "muon"]
        assert r["grad_norm"] == pytest.approx(
            math.hypot(r["grad_norms"]["adamw"][-1], r["grad_norms"]["muon"][-1])
        )
    state = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    peaks = {
        kind: state["optimizers"][kind]["param_groups"][0]["peak_lr"] for kind in ("adamw", "muon")
    }
    assert peaks == {"adamw": pytest.approx(6e-3), "muon": pytest.approx(6e-3)}  # scaled
    with pytest.raises(ValueError, match="must be positive"):
        quick_run(tmp_path, steps=20, run_dir=tmp_path / "bad", lr_multiplier=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        quick_run(tmp_path, steps=20, run_dir=tmp_path / "bad", adamw_clip=-1.0)
    with pytest.raises(ValueError, match="must be positive"):
        quick_run(tmp_path, steps=20, run_dir=tmp_path / "bad", balancer_multiplier=0.0)
    with pytest.raises(ValueError, match="must be positive"):  # a clip set from a run with no norms
        quick_run(tmp_path, steps=20, run_dir=tmp_path / "bad", muon_clip=math.nan)

    assert arm("muon", balanced=True, router_lr_multiplier=1.0, lr_multiplier=4.0) == "muon-lr4x"
    assert (
        arm("adamw", balanced=False, router_lr_multiplier=2.0, lr_multiplier=0.5)
        == "adamw-unbalanced-router2x-lr0.5x"
    )
    control = arm(
        "muon", balanced=True, router_lr_multiplier=1.0, lr_multiplier=4.0, balancer_multiplier=4.0
    )
    assert control == "muon-lr4x-balancer4x"


def test_the_muon_rate_multiplier_scales_muon_alone_and_names_the_arm(tmp_path: Path) -> None:
    run_dir = tmp_path / "equal"
    quick_run(tmp_path, steps=20, partition="muon", run_dir=run_dir, muon_lr_multiplier=0.5)
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["muon_lr_multiplier"] == 0.5
    assert settings["adamw_lr"] == settings["muon_lr"] == 3e-3  # the tuned rates, unscaled
    state = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    peaks = {
        kind: state["optimizers"][kind]["param_groups"][0]["peak_lr"] for kind in ("adamw", "muon")
    }
    assert peaks == {"adamw": pytest.approx(3e-3), "muon": pytest.approx(1.5e-3)}
    with pytest.raises(ValueError, match="settings differ.*muon_lr_multiplier"):  # part of the arm
        quick_run(tmp_path, steps=20, partition="muon", run_dir=run_dir, resume=True)
    with pytest.raises(ValueError, match="must be positive"):
        quick_run(tmp_path, steps=20, run_dir=tmp_path / "bad", muon_lr_multiplier=0.0)

    assert arm("muon", balanced=True, router_lr_multiplier=1.0, muon_lr_multiplier=0.5) == (
        "muon-muonlr0.5x"
    )
    assert arm("muon", balanced=True, router_lr_multiplier=1.0, muon_lr_multiplier=1.0) == "muon"


def test_a_run_recorded_before_a_setting_existed_reads_it_at_its_value_then(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "old"
    quick_run(tmp_path, steps=20, partition="muon", run_dir=run_dir)
    path = run_dir / "settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    del settings["muon_lr_multiplier"]  # as the first chain's runs recorded them
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    assert not quick_run(tmp_path, steps=20, partition="muon", run_dir=run_dir, resume=True)
    assert "muon_lr_multiplier" not in json.loads(path.read_text(encoding="utf-8"))  # left as is
    with pytest.raises(ValueError, match="settings differ.*muon_lr_multiplier"):
        quick_run(
            tmp_path, steps=20, partition="muon", run_dir=run_dir, resume=True, muon_lr_multiplier=2
        )


def test_the_model_alone_is_saved_one_percent_into_the_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "twenty"
    quick_run(tmp_path, steps=20, run_dir=run_dir, schedule="constant")  # one percent: step 1
    early = torch.load(run_dir / checkpoint.EARLY_FILE, weights_only=True, map_location="cpu")
    assert early["step"] == 1 and set(early) == {"step", "model"}
    one_dir = tmp_path / "one"  # the same first step, under the same warmup
    quick_run(tmp_path, steps=1, run_dir=one_dir, schedule="constant")
    after_one = torch.load(one_dir / checkpoint.FILE, weights_only=True, map_location="cpu")
    after_one = after_one["model"]
    assert set(after_one) == set(early["model"])
    assert all(torch.equal(early["model"][name], tensor) for name, tensor in after_one.items())
    assert torch.load(one_dir / checkpoint.EARLY_FILE, weights_only=True)["step"] == 1
    model = LanguageModel(VOCAB_SIZE, LAPTOP)
    assert checkpoint.load_model(run_dir, model, file=checkpoint.EARLY_FILE) == 1
    assert torch.equal(model.lm_head.weight, early["model"]["lm_head.weight"])
    assert checkpoint.load_model(run_dir, model) == 20


@pytest.mark.parametrize("partition", ["muon", "momentum", "muon_router"])
def test_the_other_partitions_learn_too(tmp_path: Path, partition: str) -> None:
    records = quick_run(tmp_path, steps=30, partition=partition)
    assert [r["step"] for r in records] == [10, 20, 30]
    assert records[-1]["loss"] < 0.5 * records[0]["loss"]
    assert "val_loss" in records[-1] and records[-1]["val_loss"] < records[0]["loss"]


def test_same_seed_reproduces_exactly_and_another_seed_differs(tmp_path: Path) -> None:
    first = [r["loss"] for r in quick_run(tmp_path, steps=20)]
    again = [r["loss"] for r in quick_run(tmp_path, steps=20)]
    other = [r["loss"] for r in quick_run(tmp_path, steps=20, seed=1)]
    assert first == again
    assert first != other


@pytest.mark.parametrize("partition", ["adamw", "momentum"])
def test_a_resumed_run_matches_an_uninterrupted_one_exactly(
    tmp_path: Path, partition: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    straight_dir, resumed_dir = tmp_path / "straight", tmp_path / "resumed"
    quick_run(tmp_path, steps=40, partition=partition, run_dir=straight_dir)
    save = checkpoint.save

    def save_then_die(
        run_dir: Path, model: nn.Module, optimizers: dict[str, Optimizer], *, step: int
    ) -> Path:
        """The run is killed right after it writes its step-20 checkpoint."""
        path = save(run_dir, model, optimizers, step=step)
        if step == 20:
            raise KeyboardInterrupt
        return path

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "save", save_then_die)
        with pytest.raises(KeyboardInterrupt):
            quick_run(tmp_path, steps=40, partition=partition, run_dir=resumed_dir)
    assert [r["step"] for r in records_of(resumed_dir / "metrics.jsonl")] == [10, 20]
    second = quick_run(tmp_path, steps=40, partition=partition, run_dir=resumed_dir, resume=True)
    assert [r["step"] for r in second] == [30, 40]

    assert records_of(resumed_dir / "metrics.jsonl") == records_of(straight_dir / "metrics.jsonl")
    routing = [
        (d / "routing.jsonl").read_text(encoding="utf-8") for d in (resumed_dir, straight_dir)
    ]
    assert routing[0] == routing[1]
    saved = [
        torch.load(d / "checkpoint.pt", weights_only=True) for d in (resumed_dir, straight_dir)
    ]
    assert saved[0]["step"] == saved[1]["step"] == 40
    assert all(
        torch.equal(a, b) for a, b in zip(saved[0]["model"].values(), saved[1]["model"].values())
    )
    settings = json.loads((resumed_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["steps"] == 40


def test_resume_refuses_other_settings_and_a_finished_run_does_nothing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    quick_run(tmp_path, steps=20, run_dir=run_dir)
    with pytest.raises(ValueError, match="settings differ"):
        quick_run(tmp_path, steps=20, seed=1, run_dir=run_dir, resume=True)
    with pytest.raises(ValueError, match="settings differ"):  # the total steps shape the cosine
        quick_run(tmp_path, steps=30, run_dir=run_dir, resume=True)
    names = ("settings.json", "metrics.jsonl", "routing.jsonl")
    logs = [(run_dir / name).read_text(encoding="utf-8") for name in names]
    assert not quick_run(tmp_path, steps=20, run_dir=run_dir, resume=True)
    # The checkpoint interval is free to change, and a finished run's files stay as they are all the same.
    assert not quick_run(tmp_path, steps=20, run_dir=run_dir, resume=True, checkpoint_every=20)
    assert len(records_of(run_dir / "metrics.jsonl")) == 2
    (run_dir / checkpoint.FILE).unlink()  # deleted for the disk: finished all the same
    assert not quick_run(tmp_path, steps=20, run_dir=run_dir, resume=True)  # not trained again
    with pytest.raises(ValueError, match="settings differ"):  # nor started over at another length
        quick_run(tmp_path, steps=30, run_dir=run_dir, resume=True)
    assert logs == [(run_dir / name).read_text(encoding="utf-8") for name in names]


def test_a_run_started_over_cannot_resume_from_the_run_it_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, straight_dir = tmp_path / "run", tmp_path / "straight"
    quick_run(tmp_path, steps=20, run_dir=run_dir)  # leaves its checkpoint and its early save
    (run_dir / "annex.json").write_text("{}", encoding="utf-8")  # and, say, its evaluation

    def die(*_args: object, **_kwargs: object) -> Path:
        """The run started over is killed before its first checkpoint is written."""
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "save", die)
        with pytest.raises(KeyboardInterrupt):
            quick_run(tmp_path, steps=20, seed=1, run_dir=run_dir)
    assert not (run_dir / checkpoint.FILE).exists() and not (run_dir / "annex.json").exists()
    resumed = quick_run(tmp_path, steps=20, seed=1, run_dir=run_dir, resume=True)
    assert [r["step"] for r in resumed] == [
        10,
        20,
    ]  # from its own first step, not the old run's last
    quick_run(tmp_path, steps=20, seed=1, run_dir=straight_dir)
    assert records_of(run_dir / "metrics.jsonl") == records_of(straight_dir / "metrics.jsonl")


def test_a_relaunch_drops_the_line_a_crash_cut_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    straight_dir, torn_dir = tmp_path / "straight", tmp_path / "torn"
    quick_run(tmp_path, steps=20, run_dir=straight_dir)
    save = checkpoint.save

    def save_then_die(
        run_dir: Path, model: nn.Module, optimizers: dict[str, Optimizer], *, step: int
    ) -> Path:
        save(run_dir, model, optimizers, step=step)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "save", save_then_die)
        with pytest.raises(KeyboardInterrupt):
            quick_run(tmp_path, steps=20, run_dir=torn_dir)
    names = ("metrics.jsonl", "routing.jsonl")
    for name in names:  # as if it had been killed later, in the middle of writing a line
        with (torn_dir / name).open("a", encoding="utf-8") as file:
            file.write('{"step": 2')
    assert not diverged(torn_dir)
    assert [r["step"] for r in quick_run(tmp_path, steps=20, run_dir=torn_dir, resume=True)] == [20]
    assert records_of(torn_dir / "metrics.jsonl") == records_of(straight_dir / "metrics.jsonl")
    routing = [(d / "routing.jsonl").read_text(encoding="utf-8") for d in (torn_dir, straight_dir)]
    assert routing[0] == routing[1]


def test_a_run_that_diverges_ends_there_marked_and_a_relaunch_leaves_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "diverged"  # a rate a million times too high: NaN gradients by step 6
    records = quick_run(tmp_path, steps=40, run_dir=run_dir, schedule="constant", lr_multiplier=1e6)
    last = records[-1]
    assert [r["step"] for r in records] == [last["step"]] and last["step"] < 10  # off the log grid
    assert math.isnan(last["grad_norm"]) and "val_loss" in last and math.isnan(last["val_loss"])
    assert len(last["losses"]) == last["step"]  # every step up to the one it ended on
    assert diverged(run_dir) and not diverged(tmp_path / "no-such-run")
    assert (run_dir / "routing.jsonl").read_text(encoding="utf-8") == ""  # no reading of NaNs
    assert (run_dir / checkpoint.EARLY_FILE).exists()  # step 1 was still a model
    assert not (run_dir / checkpoint.FILE).exists()
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    again = quick_run(
        tmp_path, steps=40, run_dir=run_dir, schedule="constant", lr_multiplier=1e6, resume=True
    )
    assert not again and (run_dir / "metrics.jsonl").read_text(encoding="utf-8") == lines
    with pytest.raises(ValueError, match="settings differ.*lr_multiplier"):  # still the same run
        quick_run(tmp_path, steps=40, run_dir=run_dir, schedule="constant", resume=True)


def test_a_loss_that_is_not_a_number_ends_the_run_after_its_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, calls = tmp_path / "poisoned", [0]

    def poisoned_step(model: LanguageModel, *args: Any, **kwargs: Any) -> object:
        """The real step, on a model whose head gets one NaN before step 25."""
        calls[0] += 1
        if calls[0] == 25:
            with torch.no_grad():
                model.lm_head.weight[0, 0] = math.nan
        return train_step(model, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr("tfg.train.train_step", poisoned_step)
        records = quick_run(tmp_path, steps=40, run_dir=run_dir)
    assert [r["step"] for r in records] == [10, 20, 25]
    assert math.isnan(records[-1]["loss"]) and math.isnan(records[-1].get("val_loss", 0.0))
    assert len(records[-1]["losses"]) == 5 and "val_loss" in records[1]
    routing = (run_dir / "routing.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in routing] == [10, 20]
    assert torch.load(run_dir / checkpoint.FILE, weights_only=True)["step"] == 20  # still finite
    assert not quick_run(tmp_path, steps=40, run_dir=run_dir, resume=True)  # over, not resumed
    assert [r["step"] for r in records_of(run_dir / "metrics.jsonl")] == [10, 20, 25]


def test_recompute_gives_the_same_run_bit_for_bit(tmp_path: Path) -> None:
    kept_dir, recomputed_dir = tmp_path / "kept", tmp_path / "recomputed"
    quick_run(tmp_path, steps=20, partition="muon", run_dir=kept_dir)
    quick_run(tmp_path, steps=20, partition="muon", run_dir=recomputed_dir, recompute=2)
    assert records_of(recomputed_dir / "metrics.jsonl") == records_of(kept_dir / "metrics.jsonl")
    routing = [
        (d / "routing.jsonl").read_text(encoding="utf-8") for d in (recomputed_dir, kept_dir)
    ]
    assert routing[0] == routing[1]
    saved = [torch.load(d / checkpoint.FILE, weights_only=True) for d in (recomputed_dir, kept_dir)]
    assert all(
        torch.equal(a, b) for a, b in zip(saved[0]["model"].values(), saved[1]["model"].values())
    )
    settings = json.loads((recomputed_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["recompute"] == 2
    with pytest.raises(ValueError, match="settings differ.*recompute"):  # part of the run
        quick_run(tmp_path, steps=20, partition="muon", run_dir=recomputed_dir, resume=True)


def test_resume_refuses_another_set_of_training_shards(tmp_path: Path) -> None:
    run_dir, data_dir = tmp_path / "run", write_shards(tmp_path)
    quick_run(tmp_path, steps=20, run_dir=run_dir)
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    batches = load_batches("train", 8, LAPTOP.context_len, seed=0, data_dir=data_dir)
    assert settings["train_windows"] == len(batches.windows)
    shard = data_dir / "train_00000.npy"  # one more shard: another data order
    (data_dir / "train_00001.npy").write_bytes(shard.read_bytes())
    with pytest.raises(ValueError, match="settings differ.*train_windows"):
        quick_run(tmp_path, steps=20, run_dir=run_dir, resume=True)


def test_a_compiled_run_trains_and_keeps_its_telemetry(tmp_path: Path) -> None:
    """The training forward through torch.compile: the loss tracks the eager run's, the routing readings still arrive (hooks would not fire), and the flag is part of the run's settings."""
    eager = quick_run(tmp_path, steps=10, run_dir=tmp_path / "eager")
    compiled = quick_run(tmp_path, steps=10, run_dir=tmp_path / "compiled", compiled=True)
    assert [r["step"] for r in compiled] == [10]
    # The compiler's stand-in for the grouped kernel takes bf16 only, so on a CPU the experts run in bf16 under it, and the two runs agree only to that precision.
    assert compiled[0]["loss"] == pytest.approx(eager[0]["loss"], rel=0.05)
    assert "val_loss" in compiled[0] and "val_loss" in eager[0]
    assert compiled[0]["val_loss"] == pytest.approx(eager[0]["val_loss"], rel=0.05)
    lines = (tmp_path / "compiled" / "routing.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    readings = json.loads(lines[0])["mixtures"]
    assert len(readings) == LAPTOP.num_blocks - LAPTOP.dense_blocks
    eager_readings = json.loads((tmp_path / "eager" / "routing.jsonl").read_text(encoding="utf-8"))[
        "mixtures"
    ]
    for compiled_reading, eager_reading in zip(readings, eager_readings):  # the stored tensors
        for name in ("residual_max", "expert_output_max"):
            assert compiled_reading[name] == pytest.approx(eager_reading[name], rel=0.2), name
        # A fourth moment that a few tokens dominate this early: the routes bf16 flips under the compiler on a GPU move it by half, so both being finite is all that holds everywhere.
        assert math.isfinite(compiled_reading["expert_kurtosis_max"])
        assert math.isfinite(eager_reading["expert_kurtosis_max"])
    assert all(sum(reading["load"]) == pytest.approx(1.0) for reading in readings)
    groups = json.loads(lines[0])["groups"]
    assert sorted(groups) == sorted((*GROUPS, GAINS))
    assert all(0 < reading["angular_step"] < 1 for reading in groups.values())
    settings = json.loads((tmp_path / "compiled" / "settings.json").read_text(encoding="utf-8"))
    assert settings["compiled"] is True


def test_a_constant_schedule_run_can_be_trained_longer(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    quick_run(tmp_path, steps=20, run_dir=run_dir, schedule="constant")
    more = quick_run(tmp_path, steps=30, run_dir=run_dir, schedule="constant", resume=True)
    assert [r["step"] for r in more] == [30] and more[0]["lr_factor"] == 1.0
    assert [r["step"] for r in records_of(run_dir / "metrics.jsonl")] == [10, 20, 30]
    files = ("settings.json", "metrics.jsonl", "routing.jsonl")
    kept = [(run_dir / name).read_text(encoding="utf-8") for name in files]
    with pytest.raises(ValueError, match="settings differ.*steps"):  # finished: not cut shorter
        quick_run(tmp_path, steps=20, run_dir=run_dir, schedule="constant", resume=True)
    (run_dir / checkpoint.FILE).unlink()  # with no checkpoint to go on from it keeps its length
    with pytest.raises(ValueError, match="settings differ.*steps"):
        quick_run(tmp_path, steps=40, run_dir=run_dir, schedule="constant", resume=True)
    assert not quick_run(tmp_path, steps=30, run_dir=run_dir, schedule="constant", resume=True)
    assert kept == [(run_dir / name).read_text(encoding="utf-8") for name in files]


def test_a_run_is_not_cut_to_a_step_it_has_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, save = tmp_path / "run", checkpoint.save

    def save_then_die(
        run_dir: Path, model: nn.Module, optimizers: dict[str, Optimizer], *, step: int
    ) -> Path:
        save(run_dir, model, optimizers, step=step)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "save", save_then_die)
        with pytest.raises(KeyboardInterrupt):
            quick_run(tmp_path, steps=40, run_dir=run_dir, schedule="constant")
    # Step 10 is logged without a validation loss: a run that ended there would have none.
    with pytest.raises(ValueError, match="checkpointed at step 10, not short of the 10"):
        quick_run(tmp_path, steps=10, run_dir=run_dir, schedule="constant", resume=True)
    shorter = quick_run(tmp_path, steps=30, run_dir=run_dir, schedule="constant", resume=True)
    assert [r["step"] for r in shorter] == [20, 30] and "val_loss" in shorter[-1]


def test_a_run_killed_in_its_last_save_trains_its_tail_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    straight_dir, run_dir, save = tmp_path / "straight", tmp_path / "run", checkpoint.save
    quick_run(tmp_path, steps=40, run_dir=straight_dir)

    def die_in_the_last(
        run_dir: Path, model: nn.Module, optimizers: dict[str, Optimizer], *, step: int
    ) -> Path:
        if step == 40:
            raise KeyboardInterrupt
        return save(run_dir, model, optimizers, step=step)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "save", die_in_the_last)
        with pytest.raises(KeyboardInterrupt):
            quick_run(tmp_path, steps=40, run_dir=run_dir)
    # The log is whole and the model on disk is the one of step 30: not finished yet.
    assert records_of(run_dir / "metrics.jsonl")[-1]["step"] == 40
    assert checkpoint.saved_step(run_dir) == 30
    tail = quick_run(tmp_path, steps=40, run_dir=run_dir, resume=True)
    assert [r["step"] for r in tail] == [40] and checkpoint.saved_step(run_dir) == 40
    assert records_of(run_dir / "metrics.jsonl") == records_of(straight_dir / "metrics.jsonl")
    saved = [torch.load(d / "checkpoint.pt", weights_only=True) for d in (run_dir, straight_dir)]
    assert all(
        torch.equal(a, b) for a, b in zip(saved[0]["model"].values(), saved[1]["model"].values())
    )
    assert not quick_run(tmp_path, steps=40, run_dir=run_dir, resume=True)  # finished now


def test_a_log_that_stops_short_of_the_checkpoint_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, save = tmp_path / "run", checkpoint.save

    def save_then_die(
        run_dir: Path, model: nn.Module, optimizers: dict[str, Optimizer], *, step: int
    ) -> Path:
        save(run_dir, model, optimizers, step=step)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "save", save_then_die)
        with pytest.raises(KeyboardInterrupt):
            quick_run(tmp_path, steps=40, run_dir=run_dir, schedule="constant")
    routing = (run_dir / "routing.jsonl").read_text(encoding="utf-8")
    (run_dir / "routing.jsonl").write_text("", encoding="utf-8")  # the machine lost the record
    with pytest.raises(ValueError, match="step 10 and its routing.jsonl stops at None"):
        quick_run(tmp_path, steps=40, run_dir=run_dir, resume=True, schedule="constant")
    (run_dir / "routing.jsonl").write_text(routing, encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="step 10 and its metrics.jsonl stops at None"):
        quick_run(tmp_path, steps=40, run_dir=run_dir, resume=True, schedule="constant")
    # The same when the checkpoint stands at the last step, ahead of a cut's refusal.
    with pytest.raises(ValueError, match="step 10 and its metrics.jsonl stops at None"):
        quick_run(tmp_path, steps=10, run_dir=run_dir, resume=True, schedule="constant")


def test_eval_interval_must_align_with_log_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multiple"):
        train(
            tier="laptop",
            steps=10,
            warmup_steps=2,
            log_every=4,
            eval_every=10,
            run_dir=tmp_path,
            data_dir=write_shards(tmp_path),
        )


def test_the_kaggle_tier_trains_from_the_command_line(tmp_path: Path) -> None:
    """Two steps at batch 2 of the tier that runs on the free GPU, through the tyro entry point.

    On AdamW: Muon's bf16 Newton-Schulz is emulated on a CPU and takes minutes at this shape.
    """
    data_dir = write_shards(tmp_path)
    records = main(
        [
            "--tier",
            "kaggle_t4",
            "--steps",
            "2",
            "--warmup-steps",
            "1",
            "--batch-size",
            "2",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--eval-batches",
            "1",
            "--checkpoint-every",
            "1",
            "--no-balanced",
            "--router-lr-multiplier",
            "2",
            "--run-dir",
            str(tmp_path / "run"),
            "--data-dir",
            str(data_dir),
        ]
    )
    assert [r["step"] for r in records] == [1, 2]
    assert all(math.isfinite(r["loss"]) and "val_loss" in r for r in records)
    assert all(len(r["grad_norms"]["adamw"]) == 1 for r in records)  # a log step every step
    assert all(r["losses"] == [r["loss"]] for r in records)
    settings = json.loads((tmp_path / "run" / "settings.json").read_text(encoding="utf-8"))
    assert settings["tier"] == "kaggle_t4" and settings["batch_size"] == 2
    assert settings["balanced"] is False and settings["router_lr_multiplier"] == 2.0
    assert [settings[key] for key in ("log_every", "eval_every", "eval_batches")] == [1, 1, 1]
    assert (tmp_path / "run" / "checkpoint.pt").exists()
    resumed = main(
        [
            "--tier",
            "kaggle_t4",
            "--steps",
            "2",
            "--warmup-steps",
            "1",  # with the batch size and the intervals, settings a resumed run may not change
            "--batch-size",
            "2",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--eval-batches",
            "1",
            "--no-balanced",
            "--router-lr-multiplier",
            "2",
            "--run-dir",
            str(tmp_path / "run"),
            "--data-dir",
            str(data_dir),
            "--resume",
        ]
    )
    assert not resumed  # already finished


def test_an_unknown_tier_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="laptop"):
        train(tier="tpu", run_dir=tmp_path, data_dir=write_shards(tmp_path))
