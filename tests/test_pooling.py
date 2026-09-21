"""Pooling a metric over a matchup's opponents into the single value per (game, model) that a line
chart plots. The point of these tests is that pooling happens on observations, not on rates."""

from __future__ import annotations

import pytest

from plybench.analysis import pooling
from plybench.analysis.pooling import GameSplit, MetricOptions, pooled_bundle
from plybench.app import PlyBench
from plybench.common.enums import GameResults, MetricName
from plybench.common.paths import BenchmarkPathBuilder
from plybench.llm import LLMConfig
from plybench.trackers.game_tracker import GameEnding, GameTracker
from plybench.trackers.result_tracker import ResultTracker

op = PlyBench(LLMConfig())
PATHS = BenchmarkPathBuilder()

MODEL = "llm:actions:text:openai:gpt-5-nano:thinking_enabled=True,reasoning_effort=high"


def _game_tracker(model, opponent, result, model_starts=True):
    i_player, o_player = (model, opponent) if model_starts else (opponent, model)
    # the result is always recorded from the starting player's point of view; get_result inverts it
    # for the second player, so flip it here to keep the model's outcome as asked
    recorded = result if model_starts else result.invert()
    tracker = GameTracker(1, i_player, o_player, {}, [], None)
    tracker.ending = GameEnding(1, "", recorded)
    return tracker


def _result_tracker(opponent_string, results, model_starts=True):
    model = op.registry.player_config(MODEL)
    opponent = op.registry.player_config(opponent_string)
    game = op.registry.game_config("tic_tac_toe:")
    games = [_game_tracker(model, opponent, result, model_starts) for result in results]
    return ResultTracker("exp", model, opponent, game, len(games), set(), op.registry, PATHS, games=list(games), save_on_record=False)


# --- pooling -----------------------------------------------------------------------------------
def test_pooling_over_opponents_sums_the_observations_rather_than_averaging_the_rates():
    # 1 win in 1 game against random, 1 win in 3 against mcts. Averaging the two rates would give
    # (1.0 + 0.333)/2 = 0.667; pooling the games gives the honest 2/4.
    trackers = [
        _result_tracker("random:distribution=uniform", [GameResults.WIN]),
        _result_tracker("mcts:max_simulations=1000,rollout_count=1,uct_c=2.0", [GameResults.WIN, GameResults.LOSS, GameResults.LOSS]),
    ]
    bundle = pooled_bundle(trackers, MetricName.WIN_RATE, None)
    assert bundle is not None
    assert bundle.n == 4
    assert bundle.value == pytest.approx(0.5)


def test_pooling_a_single_opponent_matches_that_opponent_on_its_own():
    tracker = _result_tracker("random:distribution=uniform", [GameResults.WIN, GameResults.LOSS])
    alone = pooled_bundle([tracker], MetricName.WIN_RATE, None)
    assert alone is not None
    assert (alone.value, alone.n) == (pytest.approx(0.5), 2)


def test_a_ratio_metric_carries_a_wilson_interval_so_the_band_reflects_the_pooled_sample():
    small = pooled_bundle([_result_tracker("random:distribution=uniform", [GameResults.WIN, GameResults.LOSS])], MetricName.WIN_RATE, None)
    large = pooled_bundle([_result_tracker("random:distribution=uniform", [GameResults.WIN, GameResults.LOSS] * 25)], MetricName.WIN_RATE, None)
    assert small is not None and large is not None
    assert small.wilson is not None and large.wilson is not None
    # same point estimate, but more games must narrow the interval
    assert small.value == pytest.approx(large.value)
    assert (large.wilson.upper - large.wilson.lower) < (small.wilson.upper - small.wilson.lower)


# --- the shared pool ---------------------------------------------------------------------------
def test_a_pool_builds_one_suite_per_game_and_player_however_many_metrics_are_asked_for(monkeypatch):
    # building a suite for a solvable game solves or loads the minimax cache and allocates a fresh
    # replay memo, so a suite per metric makes a second panel cost what the first one cost
    builds = []
    original = pooling.matchup_suite

    def counted(tracker, registry=None, include_fails=False):
        builds.append((tracker.game.to_string(), tracker.i.hash))
        return original(tracker, registry, include_fails)

    monkeypatch.setattr(pooling, "matchup_suite", counted)
    trackers = [_result_tracker("random:distribution=uniform", [GameResults.WIN, GameResults.LOSS])]
    pool = pooling.MetricPool(None)
    for metric in (MetricName.WIN_RATE, MetricName.LOSS_RATE, MetricName.SCORE):
        assert pool.bundle(trackers, metric) is not None
    assert len(builds) == 1


def test_a_group_spanning_two_players_is_refused_rather_than_pooled_against_one_suite():
    mine = _result_tracker("random:distribution=uniform", [GameResults.WIN])
    other = ResultTracker("exp", mine.o, mine.i, mine.game, 1, set(), op.registry, PATHS, games=list(mine.games), save_on_record=False)
    with pytest.raises(ValueError, match="does not share one game and one analysed player"):
        pooled_bundle([mine, other], MetricName.WIN_RATE, None)


# --- undefined values --------------------------------------------------------------------------
def test_optimality_is_undefined_without_a_registry_rather_than_reported_as_zero():
    # optimality comes from minimax replay, which needs a registry; a zero here would read as
    # "the model never played optimally" instead of "this was never computed"
    trackers = [_result_tracker("random:distribution=uniform", [GameResults.WIN])]
    assert pooled_bundle(trackers, MetricName.OPTIMALITY_RATE_NON_TRIVIAL, None) is None


def test_no_trackers_and_no_games_are_both_undefined():
    assert pooled_bundle([], MetricName.WIN_RATE, None) is None
    assert pooled_bundle([_result_tracker("random:distribution=uniform", [])], MetricName.WIN_RATE, None) is None


def test_a_split_with_no_matching_games_is_undefined_rather_than_zero():
    # every game here was started by the model, so the "model played second" split is empty
    trackers = [_result_tracker("random:distribution=uniform", [GameResults.WIN, GameResults.WIN], model_starts=True)]
    assert pooled_bundle(trackers, MetricName.WIN_RATE, None, MetricOptions(GameSplit.I_SECOND)) is None
    combined = pooled_bundle(trackers, MetricName.WIN_RATE, None, MetricOptions(GameSplit.I_FIRST))
    assert combined is not None and combined.n == 2
