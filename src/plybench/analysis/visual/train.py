"""Plot recorded training curves against an optional benchmark reference."""

from __future__ import annotations

from plybench.analysis.pooling import MetricOptions, MetricPool
from plybench.analysis.statistics.bundle import CIBundle
from plybench.analysis.training import LearningCurve
from plybench.analysis.visual.bench.labels import Overrides, metric_label, player_label
from plybench.analysis.visual.bench.presets import RATE_STEP, LineOptions, interval_of
from plybench.analysis.visual.core.axis import Axis
from plybench.analysis.visual.core.layers.base import Layer
from plybench.analysis.visual.core.layers.baseline import BaselineLayer
from plybench.analysis.visual.core.layers.line import LineLayer
from plybench.analysis.visual.core.panel import Panel
from plybench.analysis.visual.core.style import StyleOverride
from plybench.analysis.visual.core.ticks import CategoryTicks, StepTicks
from plybench.common.enums import GameResults, MetricName
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.registry import Registry


def benchmark_reference(
    results: BenchmarkResults,
    game: GameConfig,
    player: PlayerConfig,
    tester: PlayerConfig,
    metric: MetricName,
    registry: Registry | None = None,
    options: MetricOptions | None = None,
) -> CIBundle:
    """Use the benchmark matchup with the same game and evaluation opponent."""
    tracker = results.find(game, player, tester)
    bundle = MetricPool(registry, options).bundle([tracker], metric)
    if bundle is None:
        raise ValueError(f"No recorded {metric.value} for {game.to_string()}, {player.to_string()} vs {tester.to_string()}")
    return bundle


# the metrics the optimal policy's outcomes give; optimality and regret need no computation
def optimal_value(outcomes: dict[GameResults, float], metric: MetricName) -> float | None:
    win, draw, loss = outcomes[GameResults.WIN], outcomes[GameResults.DRAW], outcomes[GameResults.LOSS]
    values = {
        MetricName.WIN_RATE: win,
        MetricName.DRAW_RATE: draw,
        MetricName.LOSS_RATE: loss,
        MetricName.SCORE: win + 0.5 * draw,
        MetricName.FAIL_RATE: 0.0,
        MetricName.OPTIMALITY_RATE: 1.0,
        MetricName.OPTIMALITY_RATE_NON_TRIVIAL: 1.0,
        MetricName.REGRET: 0.0,
    }
    return values.get(metric)


def curve_labels(curves: list[LearningCurve], overrides: Overrides | None = None) -> list[str]:
    # each curve by its player (its label in the experiment, when it has one); the trainer, the training config and the replicate only where the curves
    # differ in them. Two players with the same label are told apart by their full config
    players = [player_label(curve.run.trainee, overrides) for curve in curves]
    players = [
        curve.run.trainee.to_string() if any(other == label and peer.run.trainee.hash != curve.run.trainee.hash for other, peer in zip(players, curves)) else label
        for curve, label in zip(curves, players)
    ]
    parts = [[label] for label in players]
    for describe in (lambda run: f"vs {player_label(run.trainer, overrides)}", lambda run: run.training_config.to_string(), lambda run: f"rep {run.replicate}"):
        values = [describe(curve.run) for curve in curves]
        if len(set(values)) > 1:
            for part, value in zip(parts, values):
                part.append(value)
    return [" · ".join(part) for part in parts]


def training_panel(
    curves: list[LearningCurve],
    metric: MetricName,
    reference: CIBundle | None = None,
    options: LineOptions | None = None,
    title: str = "",
    reference_label: str = "Benchmark reference",
    optimal: float | None = None,
    overrides: Overrides | None = None,
) -> Panel:
    if not curves:
        raise ValueError("a training panel needs at least one curve")
    game, tester = curves[0].run.game.to_string(), curves[0].tester.hash
    if any(curve.run.game.to_string() != game or curve.tester.hash != tester for curve in curves):
        raise ValueError("curves in one panel must share a game and tester")

    options = options or LineOptions()
    layers: list[Layer] = []
    has_data = False
    for curve, label in zip(curves, curve_labels(curves, overrides)):
        bundles = [point.metrics.combined.metrics.get(metric) for point in curve.points]
        has_data |= any(bundle is not None and bundle.n > 0 for bundle in bundles)
        layers.append(
            LineLayer(
                x=tuple(float(point.epoch) for point in curve.points),
                y=tuple(bundle.value if bundle is not None and bundle.n else None for bundle in bundles),
                band=tuple(interval_of(bundle) if bundle is not None and bundle.n else None for bundle in bundles) if options.show_ci else None,
                label=label,
                show_markers=options.show_markers,
                style=StyleOverride(linewidth=options.linewidth, markersize=options.marker_size, fill_alpha=options.ci_alpha),
            )
        )
    if not has_data:
        raise ValueError(f"No recorded training data for {metric.value} in {game} vs {curves[0].tester.to_string()}")
    if reference is not None:
        layers.append(BaselineLayer(reference.value, label=reference_label, annotate=False, in_legend=True, style=StyleOverride(color="#000000", linestyle="--")))
    if optimal is not None:
        # exact, so no band: where a learner that plays optimally converges
        layers.append(BaselineLayer(optimal, label="Optimal policy (exact)", annotate=False, in_legend=True, style=StyleOverride(color="#555555", linestyle=":")))

    is_rate = any(bundle.wilson is not None for curve in curves for point in curve.points if (bundle := point.metrics.combined.metrics.get(metric)) is not None)
    if reference is not None and reference.wilson is not None:
        is_rate = True
    y = Axis(label=metric_label(metric.value), limits=options.ylim or ((0.0, 1.0) if is_rate else None), ticks=StepTicks(RATE_STEP) if is_rate and options.ylim is None else None)
    epochs = sorted({point.epoch for curve in curves for point in curve.points})
    x = Axis(label="Epoch", ticks=CategoryTicks(tuple(str(epoch) for epoch in epochs), tuple(float(epoch) for epoch in epochs)), rotation=options.tick_rotation)
    return Panel(layers=layers, x=x, y=y, title=title)
