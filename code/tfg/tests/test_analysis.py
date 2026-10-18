"""Tests of the analysis on hand-written runs, then on the output of real short training runs."""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tfg.analysis import (
    INVENTORY,
    SETTINGS,
    Memo,
    arm_of,
    collect,
    endpoint_names,
    endpoints,
    matched,
    metrics_frame,
    plot,
    plot_specialization,
    report,
    spike_table,
    spikes,
    status_of,
    steps_of,
    summarize,
    window,
)
from tfg.annex import evaluate as evaluate_annex
from tfg.claims import CLAIMS, CONFIRMATORY, claims, compare, contrasts, decide
from tfg.optim import GROUPS
from tfg.panel import evaluate
from tfg.tests.synthetic import QUICK, write_panel, write_run, write_shards
from tfg.tier import TIERS
from tfg.train import arm, train


def test_endpoints_integrate_over_mixtures_and_steps(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2, 0.4, 0.6])
    values = endpoints(run_dir)
    assert values["arm"] == values["partition"] == "adamw"
    assert values["seed"] == 0 and values["steps"] == 30
    assert values["val_loss"] == 2.0
    assert values["bias_span"] == pytest.approx(0.5 + 0.05)  # mixtures averaged, then the steps
    assert values["bias_span_final"] == pytest.approx(0.6 + 0.05)  # after the warmup: 20 and 30
    assert values["load_churn"] == pytest.approx(0.2)
    assert values["update_load_correlation"] == pytest.approx(0.5)
    assert "load" not in values  # vectors are not endpoints
    assert values["weight_norm_router"] == pytest.approx(1.15)  # once per step: 1.1, 1.2
    assert values["angular_step_experts"] == pytest.approx(0.025)
    assert values["collapsed_experts"] == values["collapsed_experts_final"] == 0.0
    assert values["bias_span_early"] == pytest.approx(0.2 + 0.05)  # none within 5%: the nearest
    assert values["load_churn_early"] == 0.0
    assert values["weight_norm_router_early"] == pytest.approx(1.0)
    assert values["bias_span_warmup"] == pytest.approx(0.2 + 0.05)  # step 10 alone
    assert values["load_churn_warmup"] == 0.0
    assert values["weight_norm_router_warmup"] == pytest.approx(1.0)
    assert values["angular_step_experts_warmup"] == pytest.approx(0.01)
    undefined = values["update_load_correlation_warmup"]  # undefined at its one step
    assert isinstance(undefined, float) and math.isnan(undefined)
    assert values["grad_norm_p999_adamw"] == pytest.approx(0.5)  # 30 norms, the fillers largest
    assert values["clip_fraction_adamw"] == 0.0  # none above 1.0
    assert values["grad_norm_median_adamw"] == 0.5  # after the warmup: steps 11 to 30
    assert values["loss_initial"] == 4.0 and values["spikes"] == values["spikes_raw"] == 0
    assert values["bias_span_max"] == pytest.approx(0.6 + 0.1)  # the higher mixture's, at the end
    assert values["bias_span_first"] == pytest.approx(0.5)  # the first mixture alone
    assert values["bias_span_first_final"] == pytest.approx(0.6)
    assert values["bias_span_first_early"] == values["bias_span_first_warmup"] == 0.2
    assert values["residual_max_first"] == pytest.approx(1.15)  # the rest add one per mixture
    assert values["usage_perplexity_first_final"] == 2.0
    assert values["muon_lr_multiplier"] == 1.0  # a run from before the setting: its value then
    assert endpoints(run_dir, until=20)["bias_span_max"] == pytest.approx(0.4 + 0.1)
    assert values["status"] == "finished" and values["train_windows"] is None  # an older run
    assert not any(key.endswith(("_muon", "_momentum")) for key in values)


def test_collapsed_experts_have_under_a_tenth_of_their_share(tmp_path: Path) -> None:
    loads = [[0.5, 0.5], [0.96, 0.04], [1.0, 0.0]]  # the fair share is 0.5, a tenth 0.05
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2] * 3, loads=loads, val_losses=[2.0] * 3)
    values = endpoints(run_dir)
    assert values["collapsed_experts"] == 1.0  # one at each step after the warmup
    assert values["collapsed_experts_final"] == 1.0
    assert values["collapsed_experts_early"] == values["collapsed_experts_warmup"] == 0.0
    assert endpoints(run_dir, until=20)["collapsed_experts_final"] == 1.0


def test_a_run_logged_without_group_readings_keeps_its_other_endpoints(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2, 0.4])
    path = run_dir / "routing.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        del line["groups"]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    values = endpoints(run_dir)
    assert values["bias_span"] == pytest.approx(0.4 + 0.05)
    assert not any(key.startswith(("weight_norm", "angular_step")) for key in values)


def test_the_gradient_endpoints_read_every_step(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2] * 12, val_losses=[2.0] * 12)
    values = endpoints(run_dir)  # 120 steps, the log steps' norms up to 1.2
    assert values["clip_fraction_adamw"] == pytest.approx(2 / 120)  # the log steps at 1.1, 1.2
    p999 = values["grad_norm_p999_adamw"]
    assert isinstance(p999, float) and 1.1 < p999 <= 1.2
    cut = endpoints(run_dir, until=100)
    p999 = cut["grad_norm_p999_adamw"]
    assert cut["clip_fraction_adamw"] == 0.0 and isinstance(p999, float) and p999 <= 1.0


def test_a_diverged_run_keeps_its_undefined_validation_loss(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2, 0.4], val_losses=[2.0, math.nan])
    val_loss = endpoints(run_dir)["val_loss"]
    assert isinstance(val_loss, float) and math.isnan(val_loss)  # not the last finite one
    assert endpoints(run_dir, until=10)["val_loss"] == 2.0
    assert endpoints(run_dir, until=10)["status"] == "diverged"  # of the whole run


def test_the_status_tells_finished_unfinished_and_the_two_ways_to_diverge(tmp_path: Path) -> None:
    assert status_of(write_run(tmp_path, "adamw", 0, spans=[0.2, 0.4])) == "finished"
    stuck = write_run(tmp_path, "adamw", 1, spans=[0.2, 0.4], val_losses=[4.0])  # no better than
    assert status_of(stuck) == "diverged"  # its first training loss, 4.0: counted as diverged
    marked = write_run(tmp_path, "adamw", 2, spans=[0.2, 0.4], val_losses=[math.nan])
    assert status_of(marked) == "diverged"  # the trainer's mark
    cut = write_run(tmp_path, "adamw", 3, spans=[0.2, 0.4], val_losses=[5.0, 3.0])
    settings = json.loads((cut / "settings.json").read_text(encoding="utf-8"))
    (cut / "settings.json").write_text(json.dumps(settings | {"steps": 40}), encoding="utf-8")
    assert status_of(cut) == "unfinished"  # its log stops short of its steps
    metrics = (cut / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    (cut / "metrics.jsonl").write_text(metrics[0] + "\n", encoding="utf-8")  # val_loss 5.0 so far
    assert status_of(cut) == "unfinished"  # above its first loss, but it is not over
    (cut / "metrics.jsonl").write_text("", encoding="utf-8")
    assert status_of(cut) == "unfinished"


def test_a_run_that_diverged_before_its_first_reading_still_has_a_row(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "adamw", 0, spans=[0.2, 0.4])
    run_dir = write_run(runs_dir, "muon", 0, spans=[0.2, 0.4])
    record: dict[str, object] = {
        "step": 6,
        "loss": math.nan,
        "lr_factor": 0.6,
        "grad_norm": math.nan,
    }
    record |= {"grad_norms": {"adamw": [0.5] * 6}, "losses": [4.0] * 5 + [math.nan]}
    record |= {"val_loss": math.nan}
    (run_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (run_dir / "routing.jsonl").write_text("", encoding="utf-8")
    values = endpoints(run_dir)
    assert values["status"] == "diverged" and values["steps"] == 6
    assert "bias_span" not in values  # nothing to integrate
    for name in ("bias_span_final", "bias_span_max", "usage_perplexity_final", "val_loss"):
        value = values[name]
        assert isinstance(value, float) and math.isnan(value)
    assert values["spikes"] == 0 and values["loss_initial"] == 4.0
    table = report(runs_dir)  # the tables and the figures take it
    assert table.set_index("run")["status"].to_dict() == {
        "adamw-s0": "finished",
        "muon-s0": "diverged",
    }
    decided = pd.read_csv(runs_dir / "analysis" / "contrasts.csv").set_index(["claim", "view"])
    assert decided.loc[("S", "tokens"), "verdict"] == "incomplete"


def test_the_tokens_of_a_run_count_the_batch_it_was_given(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2, 0.4])
    assert endpoints(run_dir)["tokens"] == 20 * TIERS["laptop"].tokens_per_step
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    (run_dir / "settings.json").write_text(json.dumps(settings | {"batch_size": 4}))
    assert endpoints(run_dir)["tokens"] == 20 * 4 * TIERS["laptop"].context_len  # `--batch-size 4`


def test_a_run_with_no_record_yet_is_left_out_and_a_torn_last_line_is_not_read(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = write_run(runs_dir, "adamw", 0, spans=[0.2, 0.4])
    started = write_run(runs_dir, "muon", 0, spans=[0.2, 0.4])
    for name in ("metrics.jsonl", "routing.jsonl"):
        (started / name).write_text("", encoding="utf-8")  # made, and not yet at its first log line
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"step": 30, "lo')  # the box went away in the middle of a write
    assert status_of(started) == "unfinished" and status_of(run_dir) == "finished"
    assert collect(runs_dir)["run"].tolist() == ["adamw-s0"]
    assert metrics_frame(run_dir)["step"].tolist() == [10, 20]
    report(runs_dir)


def test_a_run_not_evaluated_yet_has_a_row_and_stays_out_of_the_comparisons(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "adamw", 0, spans=[0.2, 0.4])
    run_dir = write_run(runs_dir, "muon", 0, spans=[0.2, 0.4])
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    (run_dir / "settings.json").write_text(json.dumps(settings | {"steps": 400}), encoding="utf-8")
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [{k: v for k, v in json.loads(line).items() if k != "val_loss"} for line in lines]
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    values = endpoints(run_dir)
    val_loss = values["val_loss"]
    assert values["status"] == "unfinished" and isinstance(val_loss, float) and math.isnan(val_loss)
    assert metrics_frame(run_dir)["val_loss"].isna().all()
    report(runs_dir)  # the tables and the figures take it
    assert not (runs_dir / "analysis" / "comparisons.csv").exists()  # no pair of finished runs
    write_run(runs_dir, "momentum", 0, spans=[0.2, 0.4])
    report(runs_dir)
    comparisons = pd.read_csv(runs_dir / "analysis" / "comparisons.csv")
    assert set(comparisons["arm"]) == {"momentum"}  # the run under way is at another token count


def test_a_loss_that_is_no_number_hides_no_spike_before_it() -> None:
    losses = [4.0 + 0.05 * (-1) ** step for step in range(1, 301)]
    losses[199] += 1.0  # step 200
    losses[279] += 1.0  # step 280, whose window holds the run's last step
    for last in (math.nan, math.inf):  # what a run that diverged ends on
        found = spikes(pd.Series([*losses, last], index=range(1, 302)))
        assert found.index[found["spike"]].tolist() == [200, 280]
        assert found.index[found["spike_raw"]].tolist() == [200, 280]
        assert found["loss"].isna().tolist() == [False] * 300 + [True]  # and is itself no spike


def test_steps_of_unrolls_the_windows_the_log_lines_carry() -> None:
    windows = pd.Series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=[3, 6])
    unrolled = steps_of(windows)
    assert unrolled.index.tolist() == [1, 2, 3, 4, 5, 6]
    assert unrolled.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert steps_of(pd.Series([[7.0]], index=[1])).to_dict() == {1: 7.0}


def test_spikes_need_the_floor_and_the_deviations_but_raw_spikes_only_the_floor(
    tmp_path: Path,
) -> None:
    losses = [4.0 + 0.05 * (-1) ** step for step in range(1, 121)]  # a noise of ±0.05
    losses[5] += 1.0  # step 6: over both, but in the warmup of ten steps
    losses[29] += 0.2  # step 30: over the floor, but within five robust deviations
    losses[89] += 1.0  # step 90: over both
    run_dir = write_run(
        tmp_path, "adamw", 0, spans=[0.2] * 12, val_losses=[2.0] * 12, losses=losses
    )
    every = spikes(steps_of(metrics_frame(run_dir).set_index("step")["losses"]))
    assert every.index.tolist() == list(range(1, 121))
    assert every.index[every["spike"]].tolist() == [6, 90]
    assert every.index[every["spike_raw"]].tolist() == [6, 30, 90]
    assert every.loc[90, "excess"] == pytest.approx(1.05)
    assert every.loc[1, "loss"] == pytest.approx(3.95)
    table = spike_table(run_dir)
    assert table.index.tolist() == list(range(11, 121))  # after the warmup
    assert table.index[table["spike"]].tolist() == [90]
    assert table.index[table["spike_raw"]].tolist() == [30, 90]
    assert spike_table(run_dir, until=50).index.tolist() == list(range(11, 51))
    values = endpoints(run_dir)
    assert values["spikes"] == 1 and values["spikes_raw"] == 2
    assert values["loss_initial"] == pytest.approx(3.95)
    assert endpoints(run_dir, until=50)["spikes_raw"] == 1


def test_the_panel_divergence_is_averaged_over_mixtures_when_evaluated(tmp_path: Path) -> None:
    evaluated = write_run(tmp_path, "adamw", 0, spans=[0.2], panel=[0.1, 0.3])
    values = endpoints(evaluated)
    assert values["specialization"] == pytest.approx(0.2)
    assert values["specialization_nonspace"] == pytest.approx(0.1)
    assert "specialization" not in endpoints(write_run(tmp_path, "adamw", 1, spans=[0.2]))
    assert collect(tmp_path)["specialization"].isna().tolist() == [False, True]


def test_the_annex_readings_are_averaged_over_mixtures_when_evaluated(tmp_path: Path) -> None:
    evaluated = write_run(tmp_path, "adamw", 0, spans=[0.2], annex=[0.2, 0.4])
    values = endpoints(evaluated)
    assert values["saturation"] == pytest.approx(0.3)
    assert values["routing_purity"] == pytest.approx(0.15)
    assert "gradient_alignment" not in values  # a list per mixture is no endpoint
    assert "coactivation_max" not in values  # a reading the file lacks is left out
    assert "saturation" not in endpoints(write_run(tmp_path, "adamw", 1, spans=[0.2]))
    assert collect(tmp_path)["saturation"].isna().tolist() == [False, True]


def test_the_specialization_figure_takes_one_seed_per_arm(tmp_path: Path) -> None:
    dirs = [
        write_run(tmp_path, partition, seed, spans=[0.2], panel=[0.1, 0.3 + seed])
        for partition in ("adamw", "muon")
        for seed in (0, 1)
    ]
    dirs.append(write_run(tmp_path, "momentum", 0, spans=[0.2]))  # never evaluated: left out
    arms = {d: arm_of(json.loads((d / "settings.json").read_text(encoding="utf-8"))) for d in dirs}
    palette = dict.fromkeys(arms.values(), "C0")
    files = plot_specialization(sorted(dirs), arms=arms, palette=palette, out_dir=tmp_path)
    assert [f.name for f in files] == ["specialization_heatmap.png", "specialization_seeds.png"]
    assert all(f.exists() and f.stat().st_size > 0 for f in files)
    assert not plot_specialization(dirs[-1:], arms=arms, palette=palette, out_dir=tmp_path)


def test_the_figures_and_tables_of_the_plan_come_from_the_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    for partition in ("adamw", "muon"):
        for seed in (0, 1):
            write_run(runs_dir, partition, seed, spans=[0.2 + 0.1 * seed] * 3, panel=[0.1, 0.2])
    write_run(runs_dir, "adamw", 0, spans=[0.0] * 3, balanced=False)
    write_run(runs_dir, "momentum", 0, spans=[0.2] * 3, val_losses=[math.nan])
    other = tmp_path / "other"
    write_run(other, "adamw", 0, spans=[0.3] * 3, tier="kaggle_t4")
    table = report(runs_dir, other_tier=other)
    assert table["tokens"].tolist() == [30 * TIERS["laptop"].tokens_per_step] * 6
    assert table.set_index("run")["status"].to_dict() == {
        "adamw-s0": "finished",
        "adamw-s1": "finished",
        "adamw-unbalanced-s0": "finished",
        "momentum-s0": "diverged",
        "muon-s0": "finished",
        "muon-s1": "finished",
    }
    assert (table["usage_perplexity_final"] == 2.0).all()
    names = {path.name for path in (runs_dir / "analysis").iterdir()}
    assert {
        "inventory.csv",
        "contrasts.csv",
        "claims.csv",
        "bias_span.png",
        "bias_span_blocks.png",
        "usage_perplexity_unbalanced.png",
        "early_window.png",
        "exploratory.png",
        "update_load_correlation_seeds.png",
        "views.png",
        "tiers.png",
        "specialization_seeds.png",
    } <= names
    assert {name.replace(".png", ".pdf") for name in names if name.endswith(".png")} <= names
    assert {"bias_span_first_seeds.png", "load_churn_warmup_seeds.png"} <= names
    never_logged = {"update_load_scatter.png", "geometry.png"}
    assert not (never_logged | {"router_entropy.png", "router_row_dispersion.png"}) & names
    inventory = pd.read_csv(runs_dir / "analysis" / "inventory.csv")
    assert list(inventory.columns) == INVENTORY and len(inventory) == 6
    decided = pd.read_csv(runs_dir / "analysis" / "contrasts.csv")
    assert set(decided["endpoint"]) <= {c.endpoint for c in CLAIMS}
    both = ["tokens", "loss"]  # (a) and (b) are there: every claim they take part in
    assert decided.groupby("claim", sort=False)["view"].apply(list).to_dict() == {
        "S": both,
        "G": both,
        "R": both,
        "C4": ["tokens"],
        "C5": ["tokens"],
        "C6": both,
        "C7": both,
    }
    assert set(decided.loc[decided["claim"] == "S", "higher"]) == {"muon"}
    assert list(decided.columns[-4:]) == ["effect", "signs", "agreeing", "verdict"]
    assert (decided["verdict"] == "incomplete").all()  # two seeds are not the plan's three
    verdicts = pd.read_csv(runs_dir / "analysis" / "claims.csv", keep_default_na=False)
    assert verdicts["claim"].tolist() == list(dict.fromkeys(decided["claim"]))
    exploratory = verdicts.loc[~verdicts["confirmatory"].astype(bool), "claim"].tolist()
    assert exploratory == ["C7"]
    assert verdicts.set_index("claim")["loss"].loc[["S", "C4", "C5"]].tolist() == [
        "incomplete",
        "",
        "",
    ]
    assert not (runs_dir / "analysis" / "confirmatory.csv").exists()


def test_the_windows_are_the_warmup_and_the_five_percent_after_it(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2, 0.4] + [1.0] * 38)  # 400 steps
    values = endpoints(run_dir)
    assert values["bias_span_early"] == pytest.approx(0.7 + 0.05)  # steps 20 and 30
    assert values["load_churn_early"] == pytest.approx(0.2)
    assert values["bias_span_warmup"] == pytest.approx(0.2 + 0.05)  # step 10
    assert values["load_churn_warmup"] == 0.0
    assert values["bias_span"] == pytest.approx((0.4 + 38.0) / 39 + 0.05)  # steps 20 to 400


def test_window_takes_the_steps_inside_or_else_the_nearest_reading() -> None:
    per_step = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=[5, 10, 20])
    assert window(per_step, 4, 20)["value"].tolist() == [1.0, 2.0, 3.0]
    assert window(per_step, 5, 10)["value"].tolist() == [2.0]  # open below, closed above
    assert window(per_step, 0, 3)["value"].tolist() == [1.0]  # before the first: the first
    assert window(per_step, 12, 15)["value"].tolist() == [2.0]  # between: the nearer side
    assert window(per_step, 20, math.inf)["value"].tolist() == [3.0]  # after the last: the last


def test_endpoints_can_stop_at_a_step(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, "adamw", 0, spans=[0.2, 0.4, 0.6], val_losses=[3.0, 2.5, 2.0])
    values = endpoints(run_dir, until=20)
    assert values["steps"] == 20 and values["val_loss"] == 2.5
    assert values["bias_span"] == pytest.approx(0.4 + 0.05)  # step 20 alone
    assert values["bias_span_final"] == pytest.approx(0.4 + 0.05)
    assert values["load_churn"] == pytest.approx(0.2)


def test_matched_loss_cuts_both_runs_of_a_pair_at_the_worse_final_loss(tmp_path: Path) -> None:
    write_run(tmp_path, "adamw", 0, spans=[0.4, 0.6, 0.8])  # val_loss 2.0 at the end
    write_run(tmp_path, "adamw", 1, spans=[0.4, 0.6, 0.8], val_losses=[3.3, 3.1, 3.0])
    write_run(tmp_path, "muon", 0, spans=[0.1, 0.2, 0.3], val_losses=[2.5, 2.0, 1.5])  # the better
    write_run(tmp_path, "muon", 1, spans=[0.1, 0.2, 0.3], val_losses=[4.0, 3.5, 3.2])  # the worse
    write_run(tmp_path, "muon", 7, spans=[0.1, 0.2, 0.3], val_losses=[1.0, 1.0, 1.0])  # no partner
    table = collect(tmp_path)
    memo: Memo = {}
    cut = matched(table, tmp_path, ("muon", "adamw"), memo=memo).set_index("run")
    assert list(cut.columns) == [*table.columns[1:], "matched_step"]
    assert cut["matched_step"].to_dict() == {
        "adamw-s0": 30,  # the worse of its pair: whole
        "muon-s0": 20,  # where it first reaches 2.0
        "adamw-s1": 20,  # where it first reaches the other arm's 3.2
        "muon-s1": 30,
    }
    assert sorted(memo) == [("adamw-s0", 30), ("adamw-s1", 20), ("muon-s0", 20), ("muon-s1", 30)]
    assert cut.loc["muon-s0", "val_loss"] == 2.0 and cut.loc["muon-s0", "steps"] == 20
    assert cut.loc["muon-s0", "bias_span"] == pytest.approx(0.2 + 0.05)  # step 20 alone
    assert cut.loc["muon-s0", "bias_span_final"] == pytest.approx(0.2 + 0.05)
    assert cut.loc["adamw-s1", "val_loss"] == 3.1 and cut.loc["adamw-s1", "steps"] == 20
    assert cut.loc["adamw-s1", "bias_span"] == pytest.approx(0.6 + 0.05)
    whole = ["adamw-s0", "muon-s1"]
    assert cut.drop(columns="matched_step").loc[whole].equals(table.set_index("run").loc[whole])
    again = matched(table, tmp_path, ("adamw", "muon"), memo=memo).set_index("run")
    assert again.loc[cut.index].equals(cut) and len(memo) == 4  # the same cut, from the memo
    cut = cut.reset_index()
    comparison = compare(cut, "muon", "adamw", names=endpoint_names(cut)).set_index("endpoint")
    assert comparison.loc["matched_step", "difference"] == 0.0  # one cut by ten steps in each arm
    assert comparison.loc["val_loss", "difference"] == pytest.approx(0.05)
    spans = (0.25 - 0.75) / 2 + (0.30 - 0.65) / 2  # seed 0, then seed 1
    assert comparison.loc["bias_span", "difference"] == pytest.approx(spans)
    diverged = write_run(tmp_path, "momentum", 0, spans=[0.1, 0.2, 0.3], val_losses=[math.nan])
    table = collect(tmp_path)
    cut = matched(table, tmp_path, ("momentum", "adamw")).set_index("run")
    assert cut["matched_step"].to_dict() == {"adamw-s0": 30, diverged.name: 30}  # no loss: whole


def claim_runs(runs_dir: Path, rises: list[float]) -> None:
    """(a) and (b) at one seed per rise: the span of (b) higher by that much in every mixture, its final loss lower by 0.1."""
    for seed, rise in enumerate(rises):
        loss = 2.0 + 0.1 * seed
        write_run(runs_dir, "adamw", seed, spans=[0.4, 0.6], val_losses=[loss])
        write_run(runs_dir, "muon", seed, spans=[0.4 + rise, 0.6 + rise], val_losses=[loss - 0.1])


def contrast_of(decided: pd.DataFrame, claim: str, view: str) -> dict[str, Any]:
    """The one contrast of ``claim`` in ``view``, as a record."""
    rows = decided[(decided["claim"] == claim) & (decided["view"] == view)]
    assert len(rows) == 1
    return {str(key): value for key, value in rows.iloc[0].to_dict().items()}


def test_a_claim_passes_when_every_seed_agrees_in_every_view(tmp_path: Path) -> None:
    claim_runs(tmp_path, [0.1, 0.2, 0.15])
    table = collect(tmp_path)
    cuts = {("adamw", "muon"): matched(table, tmp_path, ("adamw", "muon"))}
    decided = contrasts(table, cuts).set_index(["claim", "view"])
    span = contrast_of(decided.reset_index(), "S", "tokens")
    assert span["lower"] == "adamw" and span["higher"] == "muon" and span["signs"] == "---"
    assert span["seeds"] == span["expected_seeds"] == 3 and span["high"] < 0
    assert span["difference"] == pytest.approx(-0.15) and span["verdict"] == "pass"
    assert decided.loc[("S", "loss"), "verdict"] == "pass"
    assert decided.loc[("R", "tokens"), "verdict"] == "pass"  # every mixture's span, too
    assert decided.loc[("C5", "tokens"), "verdict"] == "pass"  # the loss, at equal tokens only
    assert ("C5", "loss") not in decided.index and ("C3", "tokens") not in decided.index
    assert decided.loc[("C6", "tokens"), "verdict"] == "fail"  # the same correlation in both arms
    assert decided.loc[("G", "tokens"), "verdict"] == "incomplete"  # never logged: no seeds
    verdicts = claims(decided.reset_index()).set_index("claim")
    assert verdicts.loc["S"].to_dict() == {
        "confirmatory": True,
        "endpoint": "bias_span_first",
        "contrasts": 2,
        "tokens": "pass",
        "loss": "pass",
        "verdict": "pass",
    }
    assert verdicts.loc["C5", "loss"] == "" and verdicts.loc["C5", "verdict"] == "pass"
    assert verdicts.loc["C4", "verdict"] == "incomplete"  # no run was evaluated on the panel
    assert not verdicts.loc["C7", "confirmatory"]
    assert "C3" not in verdicts.index  # no run without the balancer
    assert not {"O", "O-router", "U"} & set(
        decided.reset_index()["claim"]
    )  # the confirmation tier's
    assert {c.name for c in CLAIMS if c.confirmatory} == {
        "S", "S-rate", "S-parts", "G", "G-parts", "R", "M", "C3", "C4", "C5", "C6",
        "O", "O-router", "U", "S-fast", "L", "O-fast", "S-loss", "G-loss", "L-loss",
    }  # fmt: skip


def test_the_claims_are_the_ones_the_plan_registers() -> None:
    """Section 7.2 of the plan, row by row: the endpoint, the arms predicted lower and higher, the views."""
    backbone_on_adamw, backbone_on_muon = ("adamw", "muon_router"), ("muon", "muon_except_router")
    router_on_muon = ("muon", "muon_router")
    experts_on_muon = ("muon", "muon_except_router", "muon_experts")
    experts_on_adamw = (
        "adamw",
        "muon_attention",
        "muon_router",
        "muon_dense",
        "muon_attention_dense",
    )
    parts = ("muon_experts", "muon_attention", "muon_dense", "muon_attention_dense")
    both, tokens = ("tokens", "loss"), ("tokens",)
    first, similarity = "bias_span_first", "router_input_similarity_first"
    overload, fast = "max_load_ratio_first_early", ("adamw-balancer4x", "muon-balancer4x")
    recipe = tuple(
        f"{partition}-balanceloss0.0001" for partition in ("adamw", "muon", "muon_except_router")
    )
    registered = {
        "S": (first, backbone_on_adamw, backbone_on_muon, both),
        "S-rate": (first, ("adamw",), ("muon-muonlr0.5x",), both),
        "S-parts": (first, ("adamw",), ("muon_experts", "muon_attention_dense"), both),
        "G": (similarity, backbone_on_adamw, backbone_on_muon, both),
        "G-parts": (similarity, parts, backbone_on_muon, both),
        "R": ("bias_span", ("adamw",), ("muon",), both),
        "M": ("load_churn_warmup", ("adamw", "muon"), ("muon_except_router",), both),
        "C3": ("usage_perplexity_final", ("adamw-unbalanced",), ("muon-unbalanced",), both),
        "C4": ("specialization", ("adamw",), ("muon",), tokens),
        "C5": ("val_loss", ("muon",), ("adamw",), tokens),
        "C6": ("update_load_correlation", experts_on_muon, experts_on_adamw, both),
        "O": (overload, ("adamw",), backbone_on_muon, both),
        "O-router": (overload, ("muon",), ("muon_except_router",), both),
        "U": ("router_update_common_share_first", ("adamw", "muon"), ("muon_except_router",), both),
        "S-fast": (first, fast[:1], fast[1:], both),
        "L": (first, fast[1:], ("muon",), both),
        "O-fast": (overload, fast[1:], ("muon",), both),
        "S-loss": (first, recipe[:1], recipe[1:], both),
        "G-loss": (similarity, recipe[:1], recipe[1:], both),
        "L-loss": (first, recipe[1:2], ("muon",), both),
        "O-loss": (overload, recipe[1:2], ("muon",), both),
        "L-loss-router": (first, recipe[2:], ("muon_except_router",), both),
        "C7": (
            "router_row_dispersion_early",
            router_on_muon,
            ("adamw", "muon_except_router", *parts),
            both,
        ),
    }
    assert {c.name: (c.endpoint, c.lower, c.higher, c.views) for c in CLAIMS} == registered
    assert [c.name for c in CLAIMS if not c.confirmatory] == ["O-loss", "L-loss-router", "C7"]
    assert {c.name: c.tiers for c in CLAIMS if c.tiers} == {  # Amendment 3: at twice the size
        name: ("confirmation",) for name in ("O", "O-router", "U")
    }
    assert CONFIRMATORY[0] == first  # the primary endpoint first


def test_a_claim_registered_for_one_tier_is_read_at_that_tier_alone(tmp_path: Path) -> None:
    read: dict[str, set[str]] = {}
    for tier in ("laptop", "confirmation"):
        for seed in range(3):
            for partition, rise in (("adamw", 0.0), ("muon", 0.1), ("muon_except_router", 0.2)):
                spans = [0.4 + rise, 0.6 + rise]
                write_run(
                    tmp_path / tier, partition, seed, spans=spans, val_losses=[2.0], tier=tier
                )
        read[tier] = set(contrasts(collect(tmp_path / tier), {})["claim"])
    assert {"O", "O-router", "U"} <= read["confirmation"]
    assert not {"O", "O-router", "U"} & read["laptop"]
    assert {"S", "G", "R", "M"} <= read["laptop"] & read["confirmation"]  # the claims of every tier


def test_the_balancer_claims_set_the_faster_bias_against_the_slower_one(tmp_path: Path) -> None:
    for seed in range(3):
        for balancer, spans in ((1.0, (0.4, 0.8)), (4.0, (0.4, 0.6 + 0.01 * seed))):
            for partition, span in zip(("adamw", "muon"), spans):
                write_run(
                    tmp_path,
                    partition,
                    seed,
                    spans=[span, span],
                    balancer_multiplier=balancer,
                    val_losses=[2.0],
                )
    decided = contrasts(collect(tmp_path), {})
    tokens = decided[decided["view"] == "tokens"].set_index("claim")
    assert tokens.loc["S-fast", ["lower", "higher", "verdict"]].tolist() == [
        "adamw-balancer4x",
        "muon-balancer4x",
        "pass",
    ]
    assert tokens.loc["L", ["lower", "higher", "verdict"]].tolist() == [
        "muon-balancer4x",
        "muon",
        "pass",
    ]
    assert tokens.loc["O-fast", "verdict"] == "incomplete"  # the synthetic log has no load ratio


def test_the_frontier_reads_the_tuning_block_sweeps_in_rate_order(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    for seed in (0, 1):
        write_run(runs_dir, "adamw", seed, spans=[0.4] * 3)
        write_run(runs_dir, "muon", seed, spans=[0.2] * 3)
    tuning_dir = tmp_path / "tuning"
    swept = {  # optimizer, rate: the integrated span, the final loss; a NaN loss is a diverged run
        ("adamw", 1e-3): (0.4, 2.0),
        ("adamw", 2e-3): (0.5, 1.5),  # the winner
        ("adamw", 2.5e-4): (0.3, 3.0),  # the lowest rate, whose name sorts last
        ("muon", 4e-3): (0.2, 1.0),
        ("momentum", 5e-4): (0.2, 2.5),
        ("momentum", 4e-3): (0.6, math.nan),
    }
    for (optimizer, rate), (span, loss) in swept.items():
        run = write_run(tmp_path / optimizer, optimizer, 0, spans=[span] * 3, val_losses=[loss])
        (tuning_dir / optimizer).mkdir(parents=True, exist_ok=True)
        run.rename(tuning_dir / optimizer / f"lr{rate:g}")
    (tuning_dir / "muon" / "lr0.008").mkdir()  # a run that has logged nothing yet
    report(runs_dir, tuning_dir=tuning_dir)
    frontier = pd.read_csv(runs_dir / "analysis" / "frontier.csv")
    assert list(frontier.columns) == [
        "optimizer",
        "rate",
        "run",
        "status",
        "steps",
        "val_loss",
        "bias_span",
        "bias_span_first",
        "bias_span_final",
    ]
    assert list(zip(frontier["optimizer"], frontier["rate"])) == [
        ("adamw", 2.5e-4),
        ("adamw", 1e-3),
        ("adamw", 2e-3),
        ("muon", 4e-3),
        ("momentum", 5e-4),
        ("momentum", 4e-3),
    ]
    assert frontier["run"].tolist()[:3] == ["lr0.00025", "lr0.001", "lr0.002"]
    assert frontier["status"].tolist() == ["finished"] * 5 + ["diverged"]
    assert frontier["bias_span"].tolist() == pytest.approx([0.35, 0.45, 0.55, 0.25, 0.25, 0.65])
    assert frontier["bias_span_first"].tolist() == pytest.approx([0.3, 0.4, 0.5, 0.2, 0.2, 0.6])
    assert frontier["val_loss"].tolist()[:5] == [3.0, 2.0, 1.5, 1.0, 2.5]
    assert math.isnan(frontier["val_loss"].iloc[-1])
    assert (runs_dir / "analysis" / "frontier.png").exists()
    assert (runs_dir / "analysis" / "frontier.pdf").exists()
    with pytest.raises(FileNotFoundError, match="no sweep runs"):
        report(runs_dir, tuning_dir=tmp_path / "empty")


def test_a_seed_that_disagrees_or_is_missing_keeps_a_claim_from_passing(tmp_path: Path) -> None:
    claim_runs(tmp_path / "disagrees", [0.1, 0.2, -0.05])
    table = collect(tmp_path / "disagrees")
    span = contrast_of(contrasts(table, {}), "S", "tokens")
    assert span["signs"] == "--+" and span["agreeing"] == 2 and span["verdict"] == "direction"
    decided = claims(contrasts(table, {})).set_index("claim")
    assert decided.loc["S", "tokens"] == decided.loc["S", "verdict"] == "direction"  # no pass
    assert decide(pd.Series(["pass", "direction", "incomplete"])) == "direction"
    assert decide(pd.Series(["pass", "direction", "fail"])) == "fail"
    assert contrasts(table, {}).set_index(["claim", "view"]).loc[("S", "loss"), "seeds"] == 0

    claim_runs(tmp_path / "opposite", [-0.1, -0.2, -0.05])
    span = contrasts(collect(tmp_path / "opposite"), {}).set_index(["claim", "view"])
    assert span.loc[("S", "tokens"), "verdict"] == "fail"

    claim_runs(tmp_path / "two", [0.1, 0.2])
    span = contrasts(collect(tmp_path / "two"), {}).set_index(["claim", "view"])
    assert span.loc[("S", "tokens"), "signs"] == "--"
    assert span.loc[("S", "tokens"), "verdict"] == "incomplete"  # under the plan's three seeds

    claim_runs(tmp_path / "diverged", [0.1, 0.2, 0.15, 0.1])
    write_run(tmp_path / "diverged", "momentum", 0, spans=[0.4, 0.6])
    (tmp_path / "diverged" / "muon-s3" / "metrics.jsonl").write_text(
        (tmp_path / "diverged" / "momentum-s0" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .replace("2.0", "NaN"),
        encoding="utf-8",
    )
    table = collect(tmp_path / "diverged")
    assert table.set_index("run").loc["muon-s3", "status"] == "diverged"
    span = contrast_of(contrasts(table, {}), "S", "tokens")
    assert span["seeds"] == 3 and span["expected_seeds"] == 4 and span["signs"] == "---"
    assert span["verdict"] == "incomplete"  # a seed both arms have is not there to agree


def test_runs_on_different_training_sets_are_not_collected_together(tmp_path: Path) -> None:
    for seed, windows in enumerate([100, 100, 200]):
        run_dir = write_run(tmp_path, "adamw", seed, spans=[0.2])
        settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
        settings["train_windows"] = windows
        (run_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        if seed == 1:
            assert collect(tmp_path)["train_windows"].tolist() == [100, 100]
    with pytest.raises(ValueError, match="different training sets"):
        collect(tmp_path)


def test_compare_pairs_seeds_and_counts_the_agreeing_signs(tmp_path: Path) -> None:
    for seed, drop in enumerate([0.1, 0.2, 0.15]):
        write_run(tmp_path, "adamw", seed, spans=[0.4, 0.6], panel=[0.2])
        panel = [0.1] if seed < 2 else None  # one seed without a panel evaluation
        write_run(tmp_path, "muon", seed, spans=[0.4 - drop, 0.6 - drop], panel=panel)
    write_run(tmp_path, "muon", 7, spans=[0.0, 0.0])  # no AdamW partner, so it is left out
    write_run(tmp_path, "adamw", 0, spans=[0.0, 0.0], balanced=False)  # its own arm
    table = collect(tmp_path)
    assert len(table) == 8 and list(table.columns[: len(SETTINGS)]) == SETTINGS
    summary = summarize(table)
    assert summary.loc["adamw", ("bias_span", "count")] == 3
    assert summary.loc["adamw", ("bias_span", "mean")] == pytest.approx(0.65)  # step 20 alone
    assert summary.loc["adamw-unbalanced", ("bias_span", "count")] == 1
    names = endpoint_names(table)
    unbalanced = compare(table, "adamw-unbalanced", "adamw", names=names).set_index("endpoint")
    assert unbalanced.loc["bias_span", "seeds"] == 1
    assert unbalanced.loc["bias_span", "difference"] == pytest.approx(-0.6)  # 0.05 at zero span
    assert "specialization" not in unbalanced.index  # no pair has it

    comparison = compare(table, "muon", "adamw", names=names).set_index("endpoint").to_dict("index")
    span = comparison["bias_span"]
    assert span["seeds"] == 3 and span["agreeing"] == 3 and span["signs"] == "---"
    assert span["difference"] == pytest.approx(-0.15)
    assert span["low"] <= span["difference"] <= span["high"]
    assert span["effect"] == pytest.approx(-0.15 / math.sqrt((0.0 + 0.05**2) / 2))  # one spread
    assert unbalanced.loc["bias_span", "signs"] == "-"
    assert pd.isna(unbalanced.loc["bias_span", "effect"])  # one seed: no spread
    assert comparison["val_loss"]["difference"] == 0.0
    assert comparison["specialization"]["seeds"] == 2  # the pairs that have it
    assert comparison["specialization"]["difference"] == pytest.approx(-0.1)
    with pytest.raises(ValueError, match="share no seed"):
        compare(table[table["seed"] == 7], "muon", "adamw", names=names)


def test_report_on_real_training_runs(tmp_path: Path) -> None:
    data_dir = write_shards(tmp_path)
    runs_dir = tmp_path / "runs"
    for partition, balanced in (("adamw", True), ("momentum", True), ("adamw", False)):
        train(
            tier="laptop",
            partition=partition,
            balanced=balanced,
            steps=10,
            adamw_lr=3e-3,
            muon_lr=3e-3,
            **QUICK,
            run_dir=runs_dir / arm(partition, balanced=balanced, router_lr_multiplier=1.0),
            data_dir=data_dir,
        )
    panel_dir = write_panel(tmp_path)
    for name in ("adamw", "momentum"):  # the unbalanced run stays unevaluated
        evaluate(runs_dir / name, panel_dir=panel_dir)
        evaluate_annex(runs_dir / name, panel_dir=panel_dir)
    table = report(runs_dir)
    assert sorted(table["arm"]) == ["adamw", "adamw-unbalanced", "momentum"]
    assert table.set_index("arm").loc["adamw-unbalanced", "bias_span_final"] == 0.0
    assert (table["status"] == "finished").all() and (table["tokens"] == 10 * 512).all()
    names = {path.name for path in (runs_dir / "analysis").iterdir()}
    assert {"update_load_scatter.png", "update_load_scatter_spectral.png"} <= names
    assert {"router_entropy.png", "router_row_dispersion.png", "exploratory.png"} <= names
    assert {"early_window.png", "views.png", "geometry.png"} <= names
    assert {"bias_span_first_seeds.png", "router_input_similarity_first_seeds.png"} <= names
    assert table.set_index("arm")["specialization"].isna().to_dict() == {
        "adamw": False,
        "adamw-unbalanced": True,
        "momentum": False,
    }
    comparisons = pd.read_csv(runs_dir / "analysis" / "comparisons.csv")
    assert sorted(set(comparisons["arm"])) == ["adamw-unbalanced", "momentum"]
    assert list(comparisons.columns[:4]) == ["matching", "arm", "reference", "endpoint"]
    assert sorted(set(comparisons["matching"])) == ["loss", "tokens"]
    by_matching = comparisons.groupby("matching")["endpoint"].apply(set)
    assert {"load_churn_early", "router_entropy_early", "val_loss"} <= by_matching["tokens"]
    assert {"load_churn_warmup", "bias_span_warmup", "residual_max_warmup"} <= (
        by_matching["tokens"]
    )
    assert {"router_row_similarity", "attention_logit_max"} <= by_matching["tokens"]
    assert {"grad_norm_p999_adamw", "clip_fraction_adamw", "spikes", "spikes_raw"} <= (
        by_matching["tokens"]
    )
    assert {"residual_max", "expert_kurtosis_max"} <= by_matching["tokens"]
    assert {"router_logit_norm", "router_svd_entropy", "collapsed_experts"} <= (
        by_matching["tokens"]
    )
    assert {"saturation", "routing_purity", "expert_similarity_mean"} <= by_matching["tokens"]
    assert {f"{name}_{group}" for name in ("weight_norm", "angular_step") for group in GROUPS} <= (
        by_matching["tokens"]
    )
    assert (0 <= table["clip_fraction_adamw"]).all() and (table["clip_fraction_adamw"] <= 1).all()
    assert table.set_index("arm")["grad_norm_median_momentum"].isna().to_dict() == {
        "adamw": True,
        "adamw-unbalanced": True,
        "momentum": False,
    }
    assert by_matching["loss"] == by_matching["tokens"] | {"matched_step"}
    by_arm = comparisons.groupby("arm")["endpoint"].apply(set)
    assert (
        "specialization" in by_arm["momentum"]
        and "specialization" not in by_arm["adamw-unbalanced"]
    )
    assert "saturation" in by_arm["momentum"] and "saturation" not in by_arm["adamw-unbalanced"]
    annex = pd.read_csv(runs_dir / "analysis" / "annex.csv", header=[0, 1], index_col=0)
    assert list(annex.index) == ["adamw", "momentum"]  # the arms with the readings
    assert set(annex.columns.get_level_values(0)) >= {"saturation", "gradient_alignment_mean"}
    assert (annex[("saturation", "count")] == 1).all()
    cut = pd.read_csv(runs_dir / "analysis" / "endpoints_matched.csv")
    assert (cut["matched_step"] == 10).all()  # the only evaluation is the last step
    out_dir = runs_dir / "analysis"
    assert {p.name for p in out_dir.iterdir()} >= {
        "endpoints.csv",
        "endpoints_matched.csv",
        "summary.csv",
        "comparisons.csv",
        "loss.png",
        "bias_span.png",
        "bias_span_blocks.png",  # the per-block view of every routing curve
        "residual_max.png",
        "attention_logit_max_blocks.png",
        "specialization.png",
        "saturation.png",
        "annex.csv",
        "specialization_heatmap.png",
        "specialization_seeds.png",
    }
    assert not (out_dir / "loss_blocks.png").exists()  # a run curve, not a mixture's
    assert not (out_dir / "angular_step_router_blocks.png").exists()  # a group's, not a mixture's
    assert collect(runs_dir).equals(table)  # the analysis folder is not mistaken for a run
    files = plot(runs_dir, tmp_path / "plots")
    assert all(f.exists() and f.stat().st_size > 0 for f in files)
