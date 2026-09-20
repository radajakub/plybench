"""Plot a metric across games, one line per model, from a recorded experiment.

Games form the x axis in the order they appear in the experiment config, the selected metric is on
the y axis, and every model gets its own line. Opponents are pooled at the observation level by
default (not averaged per opponent), so each point is one value per model per game; restrict them
with --opponents or split them into a grid of panels with --panel-by opponent.

Colour identifies the provider, the linestyle ranks the model's tier within that provider (nano <
mini < full, flash-lite < flash < pro), and the marker is the reasoning effort. A categorical palette
carries eight hues and an experiment enables far more models than that, which is why the three
channels are split up this way.

The linestyle ladder covers only the models actually drawn, so plotting three tiers of one provider
reads as solid (the strongest) / dashed / dotted whatever the experiment contains -- narrowing
--players therefore restyles the lines. Colours instead come from the full experiment config, so
filtering never recolours the models that remain.

Use --color-by player to give each model its own hue instead; that only works for a hand-picked
selection of at most eight, and those colours follow the selection rather than the experiment.

Metrics that do not apply to a game (optimality on an unsolvable game such as connect_four) are
left as gaps in the line rather than drawn as zero.

Examples:
    uv run python scripts/plot.py --experiment ttt

    uv run python scripts/plot.py --experiment ttt --metric win_rate --panel-by opponent --ncols 3

    uv run python scripts/plot.py --experiment ttt --color-by player --no-ci \\
        --players "llm:actions:text:claude:claude-opus-5:thinking_enabled=True,reasoning_effort=high" \\
                  "llm:actions:text:openai:gpt-5.4:thinking_enabled=True,reasoning_effort=high"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypeVar, cast

sys.path.insert(0, str(Path(__file__).parent))
from _shared import add_source_args, benchmark_from_args, build_op  # noqa: E402

from plybench.analysis.pooling import GameSplit, MetricOptions  # noqa: E402
from plybench.analysis.visual import (  # noqa: E402
    ColorBy,
    Figure,
    Layout,
    LegendSpec,
    LineOptions,
    Panel,
    Series,
    SeriesRequest,
    Style,
    StyleEncoder,
    build_series_batch,
    encoder_keys,
    indistinguishable,
    metric_label,
    metric_panel,
    player_label,
    render,
)
from plybench.app import PlyBench  # noqa: E402
from plybench.common.enums import MetricName  # noqa: E402
from plybench.common.paths import BenchmarkPathBuilder  # noqa: E402
from plybench.configs.benchmark_config import BenchmarkConfig  # noqa: E402
from plybench.configs.player_config import PlayerConfig  # noqa: E402
from plybench.harness.benchmark.results import BenchmarkResults  # noqa: E402
from plybench.registry import Registry  # noqa: E402
from plybench.utils.enums import ExtendedEnum  # noqa: E402

PANEL_BY = ("none", "opponent", "metric")


E = TypeVar("E", bound=ExtendedEnum)


def _enum(cls: type[E], value: str) -> E:
    resolved = cls.from_value(value)
    if resolved is None:
        raise SystemExit(f"unknown {cls.__name__.lower()} {value!r}; expected one of {', '.join(cls.values())}")
    return cast(E, resolved)


def _metrics(args: argparse.Namespace) -> list[MetricName]:
    if args.panel_by != "metric":
        if args.metrics != parser_default_metrics():
            print("warning: --metrics applies only to --panel-by metric; plotting --metric alone")
        return [_enum(MetricName, args.metric)]
    return list(dict.fromkeys(_enum(MetricName, name) for name in args.metrics))


def parser_default_metrics() -> list[str]:
    return [MetricName.OPTIMALITY_RATE_NON_TRIVIAL.value]


def _opponent_groups(results: BenchmarkResults, panel_by: str) -> list[tuple[str, list[PlayerConfig]]]:
    if panel_by != "opponent":
        return [("", results.opponent_configs)]
    return [(player_label(opponent), [opponent]) for opponent in results.opponent_configs]


def _series_groups(
    results: BenchmarkResults, registry: Registry, args: argparse.Namespace, options: MetricOptions, color_by: ColorBy
) -> list[tuple[str, MetricName, list[Series]]]:
    requests = [SeriesRequest(metric, opponents, title) for metric in _metrics(args) for title, opponents in _opponent_groups(results, args.panel_by)]
    batches = build_series_batch(results, registry, requests, results.game_configs, results.player_configs, options, color_by)

    groups: list[tuple[str, MetricName, list[Series]]] = []
    for request, series in zip(requests, batches, strict=True):
        drawable = [item for item in series if item.has_data]
        if not drawable:
            print(f"skipping panel (no data): metric={request.metric.value} {request.title or 'all opponents'}")
            continue
        # a metric grid names each panel by its metric; anything else already has the metric on the y axis
        heading = request.title if args.panel_by != "metric" else metric_label(request.metric.value)
        groups.append((heading, request.metric, drawable))
    return groups


def _palette_players(op: PlyBench, args: argparse.Namespace, results: BenchmarkResults, color_by: ColorBy) -> list[PlayerConfig]:
    if color_by is ColorBy.PLAYER or not args.experiment:
        return results.player_configs
    with open(BenchmarkPathBuilder().experiment_path(args.experiment)) as handle:
        config = BenchmarkConfig.from_dict(json.load(handle))
    return [op.registry.player_config(value) for value in config.get_player_configs()]


def _warn_indistinguishable(groups: list[tuple[str, MetricName, list[Series]]], encoder: StyleEncoder, line_options: LineOptions) -> None:
    ignore_markers = not line_options.show_markers
    collisions = {tuple(group) for _, _, series in groups for group in indistinguishable(series, encoder, ignore_markers)}
    reason = "--no-markers draws these identically" if ignore_markers else "these are drawn identically (same colour, linestyle and marker)"
    for group in sorted(collisions):
        print(f"warning: {reason}: {', '.join(group)}")


def _default_path(args: argparse.Namespace, paths: BenchmarkPathBuilder, name: str) -> Path:
    if args.out:
        return Path(args.out)
    suffix = "_".join(metric.value for metric in _metrics(args))
    return paths.plots_dir / f"{name}_{suffix}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument(
        "--metric", default=MetricName.OPTIMALITY_RATE_NON_TRIVIAL.value, choices=MetricName.values(), help="metric on the y axis (default optimality_rate_non_trivial)"
    )
    parser.add_argument("--metrics", nargs="+", default=parser_default_metrics(), choices=MetricName.values(), help="metrics to panel over (only with --panel-by metric)")
    parser.add_argument(
        "--split", default=GameSplit.COMBINED.value, choices=GameSplit.values(), help="which games to count: all, or only those the model started/answered (default combined)"
    )
    parser.add_argument("--panel-by", default="none", choices=PANEL_BY, help="compose a grid of panels along this axis (default none)")
    parser.add_argument("--color-by", default=ColorBy.PROVIDER.value, choices=ColorBy.values(), help="what the hue identifies (default provider)")
    parser.add_argument("--confidence", type=float, default=0.95, help="confidence level for the interval bands (default 0.95)")
    parser.add_argument("--include-fails", action="store_true", help="count invalid-move failures as losses")

    figure_group = parser.add_argument_group("figure")
    figure_group.add_argument("--out", help="output PNG path (default plots/benchmarks/<experiment>_<metric>.png)")
    figure_group.add_argument("--no-ci", action="store_true", help="hide the confidence bands")
    figure_group.add_argument("--no-markers", action="store_true", help="draw bare lines; the reasoning effort then has no encoding of its own")
    figure_group.add_argument("--linewidth", type=float, default=LineOptions.linewidth, help=f"line thickness in points (default {LineOptions.linewidth})")
    figure_group.add_argument("--marker-size", type=float, default=LineOptions.marker_size, help=f"marker size in points (default {LineOptions.marker_size})")
    figure_group.add_argument("--ncols", type=int, default=1, help="panels per row (default 1)")
    figure_group.add_argument("--panel-size", nargs=2, type=float, default=[7.0, 4.5], metavar=("W", "H"), help="size of one panel in inches (default 7 4.5)")
    figure_group.add_argument("--dpi", type=int, default=200, help="output resolution (default 200)")
    figure_group.add_argument("--font-size", type=float, default=10.0, help="base font size (default 10)")
    figure_group.add_argument("--legend-cols", type=int, default=3, help="columns in the shared legend (default 3)")
    figure_group.add_argument("--tick-rotation", type=float, default=0.0, help="rotate x tick labels by this many degrees")
    figure_group.add_argument("--ylim", nargs=2, type=float, metavar=("LOW", "HIGH"), help="fix the y axis range")
    figure_group.add_argument("--free-y", action="store_true", help="scale each panel's y axis to its own data instead of sharing one scale (panels stop being comparable)")
    figure_group.add_argument("--title", default="", help="figure title")
    args = parser.parse_args()

    op = build_op()
    benchmark = benchmark_from_args(op, args)
    results = benchmark.get_results()
    if not results.player_configs or not results.game_configs:
        raise SystemExit("the selection is empty; check --experiment / --games / --players")

    options = MetricOptions(_enum(GameSplit, args.split), args.confidence, args.include_fails)
    line_options = LineOptions(
        show_ci=not args.no_ci,
        show_markers=not args.no_markers,
        linewidth=args.linewidth,
        marker_size=args.marker_size,
        ylim=tuple(args.ylim) if args.ylim else None,
        tick_rotation=args.tick_rotation,
    )
    color_by = _enum(ColorBy, args.color_by)
    groups = _series_groups(results, op.registry, args, options, color_by)
    if not groups:
        raise SystemExit("nothing to plot: no selected model has recorded data for this metric")

    # linestyles rank only what is drawn, so a narrowed figure reads solid / dashed / dotted; colours
    # come from every player in the experiment, so a model keeps its hue across runs that filter
    drawn = [item.style for _, _, series in groups for item in series]
    encoder = StyleEncoder(drawn, encoder_keys(_palette_players(op, args, results, color_by), color_by))

    panels: list[Panel] = [metric_panel(series, metric, encoder, line_options, heading) for heading, metric, series in groups]
    _warn_indistinguishable(groups, encoder, line_options)

    figure = Figure(
        panels=panels,
        ncols=args.ncols,
        panel_size=Layout(args.panel_size[0], args.panel_size[1]),
        style=Style(font_size=args.font_size, dpi=args.dpi),
        suptitle=args.title,
        legend=LegendSpec(columns=args.legend_cols, show_markers=not args.no_markers),
        # panels of one metric must share a scale to be comparable; a metric grid measures a different
        # quantity per panel, so sharing there would squash every panel but the largest
        share_y=not args.free_y and args.panel_by != "metric",
    )
    path = render(figure, _default_path(args, BenchmarkPathBuilder(), benchmark.experiment))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
