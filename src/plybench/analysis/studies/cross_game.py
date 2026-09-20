"""Cross-game study: how a model's move quality changes between two games (A minus B), holding the
model and opponent fixed.

Composed from the general primitives — `MoveMetric` for the per-move statistics, `compare_for_family`
for each cell's test, and `combine_comparisons` to average across opponents without pooling their moves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from plybench.analysis.statistics.comparison import GroupComparison, combine_comparisons
from plybench.analysis.stats.move_metrics import DEFAULT_MOVE_METRICS, MoveMetric
from plybench.analysis.stats.moves import MoveRecord, collect_moves
from plybench.common.enums import MetricName
from plybench.configs.player_config import PlayerConfig
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.registry import Registry
from plybench.trackers.result_tracker import ResultTracker


@dataclass(frozen=True)
class MetricDiff:
    """One metric compared between game A and game B for a fixed (model, opponent). `comparison` carries
    the A-minus-B difference, its CI, its standard error and the two-sided test."""

    name: MetricName
    value_a: float
    value_b: float
    comparison: GroupComparison


@dataclass(frozen=True)
class CellDiff:
    """A single (model, opponent) cell: every metric's A-vs-B difference for that opponent alone."""

    opponent: PlayerConfig
    n_moves_a: int
    n_moves_b: int
    metrics: dict[MetricName, MetricDiff]


@dataclass(frozen=True)
class AverageDiff:
    """The across-opponent summary for one metric: the unweighted mean of the per-opponent A-minus-B
    differences (opponents are never pooled at the move level), with a combined-SE z-test on that mean."""

    name: MetricName
    mean_difference: float | None
    se: float | None
    p_value: float | None
    significant: bool
    k: int


@dataclass(frozen=True)
class ModelDiff:
    model: PlayerConfig
    per_opponent: list[CellDiff]
    average: dict[MetricName, AverageDiff]


def _metric_diff(metric: MoveMetric, moves_a: list[MoveRecord], moves_b: list[MoveRecord], confidence: float) -> MetricDiff:
    dist_a, dist_b = metric.distribution(moves_a), metric.distribution(moves_b)
    return MetricDiff(metric.name, dist_a.mean, dist_b.mean, metric.compare(moves_a, moves_b, confidence))


def _cell_diff(opponent: PlayerConfig, moves_a: list[MoveRecord], moves_b: list[MoveRecord], metrics: Sequence[MoveMetric], confidence: float) -> CellDiff:
    return CellDiff(
        opponent=opponent,
        n_moves_a=len(moves_a),
        n_moves_b=len(moves_b),
        metrics={metric.name: _metric_diff(metric, moves_a, moves_b, confidence) for metric in metrics},
    )


def _average_diff(name: MetricName, cells: list[CellDiff], confidence: float) -> AverageDiff:
    combined = combine_comparisons([cell.metrics[name].comparison for cell in cells], confidence)
    return AverageDiff(name, combined.value, combined.se, combined.p_value, combined.significant, combined.k)


def _find_moves(results: BenchmarkResults, game_key: str, model: PlayerConfig, opponent: PlayerConfig, registry: Registry) -> list[MoveRecord] | None:
    """Judged moves for one (game, model, opponent) matchup, or None when that matchup was not recorded
    or produced no completed games."""
    game = next((g for g in results.game_configs if g.to_string() == game_key), None)
    if game is None:
        return None
    try:
        tracker: ResultTracker = results.find(game, model, opponent)
    except ValueError:
        return None
    if not any(g is not None for g in tracker.games):
        return None
    return collect_moves(tracker, registry)


def compare_games(
    results: BenchmarkResults,
    registry: Registry,
    game_a: str,
    game_b: str,
    metrics: Sequence[MoveMetric] = DEFAULT_MOVE_METRICS,
    confidence: float = 0.95,
    model_filter: Callable[[PlayerConfig], bool] = lambda config: config.key == "llm",
) -> list[ModelDiff]:
    """Per-model, per-opponent difference (game A minus game B) on each move-metric, plus an across-
    opponent average that combines the per-opponent differences without pooling their moves. Only models
    passing `model_filter` and only opponents both games share (with judged moves on both) are reported."""
    diffs: list[ModelDiff] = []
    for model in results.player_configs:
        if not model_filter(model):
            continue
        cells: list[CellDiff] = []
        for opponent in results.opponent_configs:
            moves_a = _find_moves(results, game_a, model, opponent, registry)
            moves_b = _find_moves(results, game_b, model, opponent, registry)
            if not moves_a or not moves_b:
                continue
            cells.append(_cell_diff(opponent, moves_a, moves_b, metrics, confidence))
        if not cells:
            continue
        diffs.append(ModelDiff(model, cells, {metric.name: _average_diff(metric.name, cells, confidence) for metric in metrics}))
    return diffs
