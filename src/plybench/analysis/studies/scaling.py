"""Token-scaling study: is a model's token spend driven by tactical difficulty (a reasoner) or merely
by surface position properties (a retriever)?

Composed from the general primitives — `linear_fit`/`fit_difference` for the slopes, `QuantilePartitioner`
for the difficulty bins, `mean_difference_test` and `combine_comparisons` for the spend comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from plybench.analysis.statistics.comparison import CombinedEstimate, GroupComparison, combine_comparisons, mean_difference_test
from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.regression import FitDifference, LinearFit, fit_difference, linear_fit
from plybench.analysis.stats.move_features import UNGUARDED_FEATURES, MoveFeature, decision_moves, sharpness
from plybench.analysis.stats.moves import MoveRecord, collect_moves
from plybench.analysis.stats.partition import QuantilePartitioner
from plybench.configs.player_config import PlayerConfig
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.registry import Registry


def _tokens(moves: Sequence[MoveRecord]) -> list[float]:
    return [float(m.output_tokens) for m in moves if m.output_tokens is not None]


def token_slope(moves: Sequence[MoveRecord], feature: MoveFeature) -> LinearFit:
    """OLS fit of output tokens on one move feature. Only moves carrying a token count are used, so the
    feature and token samples stay paired."""
    observed = [m for m in moves if m.output_tokens is not None]
    return linear_fit([feature(m) for m in observed], _tokens(observed))


def token_slope_shift(moves_a: Sequence[MoveRecord], moves_b: Sequence[MoveRecord], feature: MoveFeature, confidence: float = 0.95) -> FitDifference:
    return fit_difference(token_slope(moves_a, feature), token_slope(moves_b, feature), confidence)


@dataclass(frozen=True)
class SpendBin:
    """One difficulty bin for the accuracy-buys-spend test: the token spend of the analysed player's
    optimal vs suboptimal moves in that bin. `comparison` is optimal-minus-suboptimal mean tokens — a
    reasoner spends more on the moves it gets right."""

    label: str
    n_optimal: int
    n_suboptimal: int
    comparison: GroupComparison


@dataclass(frozen=True)
class AccuracySpend:
    """Whether token spend buys accuracy within a fixed difficulty (sharpness) bin, controlling for game
    phase. Per-bin optimal-minus-suboptimal token differences, combined across bins without pooling the
    moves. Positive combined value = extra spend on sharp states pays off (reasoner); ~zero = spend is
    decoupled from correctness (retriever)."""

    bins: list[SpendBin]
    combined: CombinedEstimate
    n: int


def _spend_bin(label: str, moves: Sequence[MoveRecord], confidence: float) -> SpendBin:
    optimal = Distribution(_tokens([m for m in moves if m.is_optimal]))
    suboptimal = Distribution(_tokens([m for m in moves if not m.is_optimal]))
    return SpendBin(label, optimal.n, suboptimal.n, mean_difference_test(optimal, suboptimal, confidence))


def accuracy_buys_spend(moves: Sequence[MoveRecord], n_bins: int = 3, confidence: float = 0.95) -> AccuracySpend:
    partitions = QuantilePartitioner("sharpness", sharpness, n_bins).partition(list(moves))
    bins = [_spend_bin(p.label, p.moves, confidence) for p in partitions]
    return AccuracySpend(bins, combine_comparisons([b.comparison for b in bins], confidence), len(moves))


@dataclass(frozen=True)
class ModelScaling:
    model: PlayerConfig
    n_a: int
    n_b: int
    within: dict[str, LinearFit]
    accuracy_spend: AccuracySpend
    shifts: dict[str, FitDifference]


def _pooled_decision_moves(results: BenchmarkResults, game_str: str, model: PlayerConfig, registry: Registry) -> list[MoveRecord]:
    """All of a model's decision-point moves for one game, pooled across every opponent (the token<-
    feature relationship is a within-move property, so pooling only adds power)."""
    game = next((g for g in results.game_configs if g.to_string() == game_str), None)
    if game is None:
        return []
    moves: list[MoveRecord] = []
    for opponent in results.opponent_configs:
        try:
            tracker = results.find(game, model, opponent)
        except ValueError:
            continue
        if not any(g is not None for g in tracker.games):
            continue
        moves += decision_moves(collect_moves(tracker, registry))
    return moves


def analyze_scaling(
    results: BenchmarkResults,
    registry: Registry,
    game_a: str,
    game_b: str,
    features: Sequence[tuple[str, MoveFeature]] = UNGUARDED_FEATURES,
    n_bins: int = 3,
    confidence: float = 0.95,
    model_filter: Callable[[PlayerConfig], bool] = lambda config: config.key == "llm",
) -> list[ModelScaling]:
    """Per-model token-scaling analysis on decision points. Within game A: token<-feature slopes plus the
    accuracy-buys-spend test. Across games (A vs B): the token<-feature slope shift. Moves are pooled
    across opponents per model; only models passing `model_filter` with decision moves in A are reported."""
    scalings: list[ModelScaling] = []
    for model in results.player_configs:
        if not model_filter(model):
            continue
        moves_a = _pooled_decision_moves(results, game_a, model, registry)
        if not moves_a:
            continue
        moves_b = _pooled_decision_moves(results, game_b, model, registry)
        scalings.append(
            ModelScaling(
                model=model,
                n_a=len(moves_a),
                n_b=len(moves_b),
                within={label: token_slope(moves_a, feature) for label, feature in features},
                accuracy_spend=accuracy_buys_spend(moves_a, n_bins, confidence),
                shifts={label: token_slope_shift(moves_a, moves_b, feature, confidence) for label, feature in features} if moves_b else {},
            )
        )
    return scalings
