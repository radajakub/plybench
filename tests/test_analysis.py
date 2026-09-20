"""Phase 6a: the in-memory statistics layer. The CI primitives behave (Wilson / SEM / t / bootstrap),
and `BenchmarkAnalysis` composes the same benchmark metrics with confidence intervals over a finished
(offline, bot-only) benchmark run."""

from __future__ import annotations

import asyncio
import json

from plybench.analysis import BenchmarkAnalysis
from plybench.analysis.statistics.bundle import mean_bundle, ratio_bundle
from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.intervals import bootstrap_ci, sem_ci, t_ci, wilson_ci
from plybench.app import PlyBench
from plybench.common.enums import MetricName
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.llm import LLMConfig

op = PlyBench(LLMConfig())
registry = op.registry

_FIELD_METRICS = {
    MetricName.WIN_RATE,
    MetricName.DRAW_RATE,
    MetricName.LOSS_RATE,
    MetricName.FAIL_RATE,
    MetricName.SCORE,
    MetricName.MOVES_PER_GAME,
    MetricName.INPUT_TOKENS_PER_GAME,
    MetricName.OUTPUT_TOKENS_PER_GAME,
    MetricName.INPUT_TOKENS_PER_MOVE,
    MetricName.OUTPUT_TOKENS_PER_MOVE,
}


# --- statistics primitives -------------------------------------------------------------------
def test_distribution_basics():
    dist = Distribution([1, 1, 0, 0])
    assert (dist.n, dist.total, dist.mean, dist.ratio) == (4, 2.0, 0.5, 0.5)
    assert Distribution().n == 0 and Distribution().mean == 0.0


def test_wilson_interval_brackets_the_ratio():
    ci = wilson_ci(Distribution([1, 1, 0, 0]))
    assert ci.value == 0.5
    assert 0.0 < ci.lower < 0.5 < ci.upper < 1.0
    empty = wilson_ci(Distribution())
    assert empty.unwrap() == (0.0, 0.0, 0.0)


def test_mean_intervals_and_bootstrap_degeneracy():
    dist = Distribution([1, 2, 3, 4, 5])
    sem, t = sem_ci(dist), t_ci(dist)
    assert sem.value == 3.0 and t.value == 3.0
    # the t interval is wider than the normal (SEM) interval for the same small sample
    assert t.upper > sem.upper and t.lower < sem.lower
    # a zero-variance sample yields a point interval
    assert bootstrap_ci(Distribution([2, 2, 2])).unwrap() == (2.0, 2.0, 2.0)


def test_bundle_families_carry_the_right_intervals():
    ratio = ratio_bundle(Distribution([1, 0, 1, 1]))
    assert ratio.wilson is not None and ratio.sem is None and ratio.t is None and ratio.bootstrap is None

    mean = mean_bundle(Distribution([1, 2, 3, 4, 5]))
    assert mean.wilson is None and mean.sem is not None and mean.t is not None and mean.bootstrap is not None


# --- end-to-end over a finished benchmark ----------------------------------------------------
def _run_benchmark(tmp_path, monkeypatch, num_games=4):
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["random:distribution=uniform"], ["random:distribution=normal"], num_games)
    return asyncio.run(benchmark.run(sync=True, concurrency=1))


def test_benchmark_analysis_composes_all_metrics(tmp_path, monkeypatch):
    results = _run_benchmark(tmp_path, monkeypatch, num_games=4)
    analysis = BenchmarkAnalysis(results)

    stats = analysis.matchup(
        registry.game_config("tic_tac_toe:"),
        registry.player_config("random:distribution=uniform"),
        registry.player_config("random:distribution=normal"),
    )

    combined = stats.metrics.combined
    # tic_tac_toe is recognisable, so the recognition-rate extractor is always present (no traces -> n=0)
    assert set(combined.metrics.keys()) == _FIELD_METRICS | {MetricName.RECOGNITION_RATE}
    assert combined.metrics[MetricName.RECOGNITION_RATE].value == 0.0 and combined.metrics[MetricName.RECOGNITION_RATE].n == 0
    # colour balancing over 4 rounds -> 2 where the player started, 2 where it played second
    assert combined.n_games == 4
    assert stats.metrics.i_first.n_games == 2 and stats.metrics.i_second.n_games == 2

    # outcome rates are proportions with Wilson intervals bracketing the value
    for name in (MetricName.WIN_RATE, MetricName.DRAW_RATE, MetricName.LOSS_RATE, MetricName.FAIL_RATE):
        bundle = combined.metrics[name]
        assert bundle.wilson is not None and bundle.sem is None
        assert 0.0 <= bundle.value <= 1.0
        assert bundle.wilson.lower <= bundle.value <= bundle.wilson.upper

    # random vs random never fails and never draws a win-by-opponent-fail, so win+draw+loss = 1
    total = sum(combined.metrics[name].value for name in (MetricName.WIN_RATE, MetricName.DRAW_RATE, MetricName.LOSS_RATE))
    assert abs(total - 1.0) < 1e-9
    assert combined.metrics[MetricName.FAIL_RATE].value == 0.0

    # score is a mean metric (SEM/t/bootstrap, no Wilson)
    score = combined.metrics[MetricName.SCORE]
    assert score.wilson is None and score.sem is not None and score.t is not None and score.bootstrap is not None
    assert 0.0 <= score.value <= 1.0

    # the fastest possible loss (opponent wins on its 3rd move) leaves the second player 2 moves, so 2
    # is the true floor for the per-game mean; bots have no token concept -> 0
    assert combined.metrics[MetricName.MOVES_PER_GAME].value >= 2
    assert combined.metrics[MetricName.INPUT_TOKENS_PER_GAME].value == 0.0
    assert combined.metrics[MetricName.OUTPUT_TOKENS_PER_MOVE].value == 0.0


def test_analyze_returns_one_matchup_and_serializes(tmp_path, monkeypatch):
    results = _run_benchmark(tmp_path, monkeypatch, num_games=2)
    all_stats = BenchmarkAnalysis(results).analyze()
    assert len(all_stats) == 1
    # the in-memory stats render to plain JSON
    json.dumps(all_stats[0].to_dict())


# --- quality metrics via minimax replay (Phase 6b) -------------------------------------------
def test_quality_metrics_for_optimal_player(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # OptimalPlayer solves + caches minimax under ./cache
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["optimal:stochastic=True"], ["random:distribution=uniform"], 2)
    results = asyncio.run(benchmark.run(sync=True, concurrency=1))

    stats = BenchmarkAnalysis(results, op.registry).matchup(
        registry.game_config("tic_tac_toe:"),
        registry.player_config("optimal:stochastic=True"),
        registry.player_config("random:distribution=uniform"),
    )
    combined = stats.metrics.combined

    # the optimal player plays optimally every move -> optimality 1, zero regret
    optimality = combined.metrics[MetricName.OPTIMALITY_RATE]
    assert optimality.value == 1.0 and optimality.wilson is not None
    assert MetricName.OPTIMALITY_RATE_NON_TRIVIAL in combined.metrics
    regret = combined.metrics[MetricName.REGRET]
    assert regret.value == 0.0 and regret.sem is not None


def test_quality_gated_on_registry_and_solvability(tmp_path, monkeypatch):
    results = _run_benchmark(tmp_path, monkeypatch, num_games=2)  # random vs random on tic_tac_toe

    # no registry -> no replay -> the field-based metrics plus recognition rate (which needs no minimax)
    without_registry = BenchmarkAnalysis(results).matchup(
        registry.game_config("tic_tac_toe:"),
        registry.player_config("random:distribution=uniform"),
        registry.player_config("random:distribution=normal"),
    )
    assert set(without_registry.metrics.combined.metrics.keys()) == _FIELD_METRICS | {MetricName.RECOGNITION_RATE}

    # with a registry, a solvable game gains the quality metrics
    with_registry = BenchmarkAnalysis(results, op.registry).matchup(
        registry.game_config("tic_tac_toe:"),
        registry.player_config("random:distribution=uniform"),
        registry.player_config("random:distribution=normal"),
    )
    assert MetricName.OPTIMALITY_RATE in with_registry.metrics.combined.metrics
    assert MetricName.REGRET in with_registry.metrics.combined.metrics

    # solvability is a registry lookup: connect_four is not solvable
    assert registry.solvable("tic_tac_toe") is True
    assert registry.solvable("connect_four") is False
