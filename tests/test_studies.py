"""The composed studies: the cross-game A-vs-B comparison and the token-scaling / reasoner-vs-retriever
analysis. These only wire the general primitives together, so the tests focus on that wiring."""

from __future__ import annotations

import asyncio

import pytest

from plybench.analysis.stats.move_features import branching, sharpness
from plybench.analysis.stats.moves import MoveRecord
from plybench.analysis.studies.cross_game import compare_games
from plybench.analysis.studies.scaling import accuracy_buys_spend, token_slope, token_slope_shift
from plybench.app import PlyBench
from plybench.common.enums import MetricName, StateClass
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.llm import LLMConfig

op = PlyBench(LLMConfig())
ANY_MODEL = lambda config: True  # noqa: E731 - bots, not llm:, drive the recorded games in tests


def _move(tokens, n_legal=9, n_optimal=1, is_optimal=True):
    return MoveRecord(StateClass.DECISION, is_optimal, 0.0, None, tokens, None, n_legal, n_optimal)


# --- token scaling ------------------------------------------------------------------------------
def test_token_slope_recovers_spend_rising_with_difficulty():
    # sharpness = 1 - n_optimal/9, so n_optimal 9->1 walks sharpness 0.0 -> 0.89 while tokens rise
    moves = [_move(tokens=100 + 50 * k, n_optimal=9 - k) for k in range(9)]
    fit = token_slope(moves, sharpness)
    assert fit.defined and fit.slope is not None and fit.slope > 0
    assert fit.r == pytest.approx(1.0) and fit.n == 9


def test_token_slope_pairs_features_only_with_moves_that_have_tokens():
    # a token-less move must drop from BOTH samples, not just the y side
    moves = [_move(tokens=100, n_optimal=1), _move(tokens=None, n_optimal=5), _move(tokens=300, n_optimal=9)]
    assert token_slope(moves, branching).n == 2  # the token-less move is gone, x and y stay aligned


def test_token_slope_shift_compares_two_games_slopes():
    steep = [_move(tokens=100 + 100 * k, n_optimal=9 - k) for k in range(9)]
    flat = [_move(tokens=100, n_optimal=9 - k) for k in range(9)]
    shift = token_slope_shift(steep, flat, sharpness)
    assert shift.delta_slope is not None and shift.delta_slope > 0
    assert shift.n_a == 9 and shift.n_b == 9


def test_accuracy_buys_spend_detects_extra_spend_on_correct_moves():
    # within every difficulty level, the optimal moves cost far more tokens than the suboptimal ones
    moves = []
    for n_optimal in (1, 3, 6):
        moves += [_move(tokens=500 + i, n_optimal=n_optimal, is_optimal=True) for i in range(5)]
        moves += [_move(tokens=100 + i, n_optimal=n_optimal, is_optimal=False) for i in range(5)]

    spend = accuracy_buys_spend(moves, n_bins=3)
    assert len(spend.bins) > 1 and spend.n == len(moves)
    assert spend.combined.value is not None and spend.combined.value > 0  # spend buys accuracy
    assert spend.combined.k == len(spend.bins)


def test_accuracy_buys_spend_is_undefined_without_both_outcomes_in_a_bin():
    only_optimal = [_move(tokens=100 + i, n_optimal=1, is_optimal=True) for i in range(6)]
    spend = accuracy_buys_spend(only_optimal, n_bins=3)
    assert spend.combined.value is None and spend.combined.k == 0


# --- cross-game comparison ------------------------------------------------------------------------
def test_compare_games_reports_per_opponent_and_combined_differences(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    game_a, game_b = "tic_tac_toe:", "modified_tic_tac_toe:"
    benchmark = Benchmark("exp", op, [game_a, game_b], ["random:distribution=uniform"], ["random:distribution=normal"], 6)
    results = asyncio.run(benchmark.run(sync=True, concurrency=1))

    (diff,) = compare_games(results, op.registry, game_a, game_b, model_filter=ANY_MODEL)
    assert len(diff.per_opponent) == 1 and diff.per_opponent[0].n_moves_a > 0

    optimality = diff.average[MetricName.OPTIMALITY_RATE]
    assert optimality.mean_difference is not None and optimality.se is not None and optimality.k == 1
    # bots emit no token counts, so a token metric has nothing to combine and stays undefined
    assert diff.average[MetricName.OUTPUT_TOKENS_PER_MOVE].mean_difference is None


def test_compare_games_skips_models_without_moves_in_both_games(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["random:distribution=uniform"], ["random:distribution=normal"], 2)
    results = asyncio.run(benchmark.run(sync=True, concurrency=1))
    # game B was never recorded, so no cell can be built
    assert compare_games(results, op.registry, "tic_tac_toe:", "modified_tic_tac_toe:", model_filter=ANY_MODEL) == []
