"""The status of an experiment (`--status`): per matchup, or per training run, epoch and phase, the games saved of those planned and the default
metrics so far, as the analysis computes them, read from the saved games without playing one."""

from __future__ import annotations

import asyncio
import json
import sys

from plyllm import LLMConfig
from test_learnable import _Stub, _StubParams

from plybench.analysis.stats.compute import compute_matchup_metrics
from plybench.analysis.training import TrainingAnalysis
from plybench.app import PlyBench
from plybench.cli import run
from plybench.common.enums import MetricName
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.status import benchmark_status, training_status
from plybench.harness.training.training import TrainingHarness
from plybench.player.spec import PlayerSpec

UNIFORM = "random:distribution=uniform"
OPTIMAL = "optimal:stochastic=True"


def _op() -> PlyBench:
    # `stub:` is the learnable player of test_learnable
    op = PlyBench(LLMConfig())
    op.registry.register_player(PlayerSpec("stub", _StubParams, lambda game, cfg, pid: _Stub(cfg, pid)))
    return op


def test_a_benchmark_status_counts_the_saved_games_and_matches_the_analysis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("status", _op(), ["tic_tac_toe:", "nim:"], [UNIFORM], [OPTIMAL], 6, seed=0)
    status = benchmark_status(benchmark)
    assert [(matchup["planned_games"], matchup["saved_games"], matchup["metrics"]) for matchup in status["matchups"]] == [(6, 0, None), (6, 0, None)]

    asyncio.run(benchmark.run(sync=True))
    status = benchmark_status(benchmark)
    for matchup, tracker in zip(status["matchups"], benchmark.get_results(progress=False).trackers, strict=True):
        assert (matchup["game"], matchup["player"], matchup["opponent"], matchup["saved_games"]) == (tracker.game.to_string(), tracker.i.to_string(), tracker.o.to_string(), 6)
        expected = compute_matchup_metrics(tracker).combined.metrics
        for name in (MetricName.WIN_RATE, MetricName.DRAW_RATE, MetricName.LOSS_RATE, MetricName.MOVES_PER_GAME):
            assert matchup["metrics"][name.value]["value"] == expected[name].value
        wilson = expected[MetricName.WIN_RATE].wilson
        assert wilson is not None and (matchup["metrics"]["win_rate"]["lower"], matchup["metrics"]["win_rate"]["upper"]) == (wilson.lower, wilson.upper)
    # tokens a random player never reports: no NaN in the JSON
    json.dumps(status, allow_nan=False)


def test_a_training_status_has_every_epoch_and_matches_the_learning_curve(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    schedule = "num_epochs=2,num_training_games=3,num_test_games=4"
    harness = TrainingHarness("status", _op(), ["tic_tac_toe:"], ["stub:"], [UNIFORM], [UNIFORM, OPTIMAL], [schedule], 2, seed=0)
    status = training_status(harness)
    assert status["testers"] == [UNIFORM, OPTIMAL] and len(status["runs"]) == 2
    epochs = status["runs"][0]["epochs"]
    # epoch 0 is only evaluated
    assert [epoch["epoch"] for epoch in epochs] == [0, 1, 2] and epochs[0]["train"] is None
    assert epochs[1]["train"] == {"planned_games": 3, "saved_games": 0, "metrics": None}
    assert {tester: phase["saved_games"] for tester, phase in epochs[0]["tests"].items()} == {UNIFORM: 0, OPTIMAL: 0}

    asyncio.run(harness.run(sync=True))
    status = training_status(harness)
    analysis = TrainingAnalysis(harness.get_results(progress=False))
    for run_status, run_config in zip(status["runs"], harness.runs, strict=True):
        assert run_status["replicate"] == run_config.replicate and run_status["training_config"] == schedule
        for tester in [harness.op.registry.player_config(value) for value in harness.tester_configs]:
            curve = analysis.curve(run_config, tester).values(MetricName.WIN_RATE)
            assert [epoch["tests"][tester.to_string()]["metrics"]["win_rate"]["value"] for epoch in run_status["epochs"]] == curve
        assert all(epoch["train"]["saved_games"] == 3 for epoch in run_status["epochs"][1:])
    json.dumps(status, allow_nan=False)


def test_run_prints_the_status_as_json_without_playing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["plybench-run", "--name", "status", "--games", "tic_tac_toe:", "--players", UNIFORM, "--opponents", UNIFORM, "--status"])
    run.main()
    status = json.loads(capsys.readouterr().out)
    assert status["kind"] == "benchmark" and status["matchups"][0]["saved_games"] == 0
    # the path builder makes the empty folders, as for --check; no game and no metadata is written
    assert not list((tmp_path / "results").rglob("*.json"))
