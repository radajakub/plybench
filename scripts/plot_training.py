"""Plot recorded training progress, with an optional benchmark reference.

Examples:
    uv run python scripts/plot_training.py --experiment memory --metric win_rate
    uv run python scripts/plot_training.py --experiment memory --baseline-experiment ttt --baseline-player 'random:distribution=uniform'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared import add_training_source_args, build_op, training_from_args  # noqa: E402

from plybench.analysis.pooling import MetricOptions  # noqa: E402
from plybench.analysis.training import TrainingAnalysis  # noqa: E402
from plybench.analysis.visual import Figure, Layout, LegendSpec, LineOptions, Style, benchmark_reference, player_label, render, training_panel  # noqa: E402
from plybench.common.enums import MetricName  # noqa: E402
from plybench.common.paths import TrainingHarnessPathBuilder  # noqa: E402
from plybench.harness.benchmark.benchmark import Benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_source_args(parser)
    parser.add_argument("--metric", default=MetricName.WIN_RATE.value, choices=MetricName.values(), help="metric on the y axis (default win_rate)")
    parser.add_argument("--baseline-experiment", help="recorded benchmark experiment used as the horizontal reference")
    parser.add_argument("--baseline-player", help="benchmark player to use; required when the benchmark has several players")
    parser.add_argument("--confidence", type=float, default=0.95, help="confidence level for training intervals (default 0.95)")
    parser.add_argument("--include-fails", action="store_true", help="count invalid-move failures as losses")
    parser.add_argument("--out", help="output PNG path (default plots/training/<experiment>_<metric>.png)")
    parser.add_argument("--no-ci", action="store_true", help="hide training confidence bands")
    parser.add_argument("--no-markers", action="store_true", help="draw training lines without epoch markers")
    parser.add_argument("--ncols", type=int, default=1, help="panels per row (default 1)")
    parser.add_argument("--panel-size", nargs=2, type=float, default=[7.0, 4.5], metavar=("W", "H"), help="size of one panel in inches (default 7 4.5)")
    parser.add_argument("--dpi", type=int, default=200, help="output resolution (default 200)")
    parser.add_argument("--font-size", type=float, default=10.0, help="base font size (default 10)")
    parser.add_argument("--legend-cols", type=int, default=3, help="columns in the shared legend (default 3)")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("LOW", "HIGH"), help="fix the y axis range")
    parser.add_argument("--free-y", action="store_true", help="scale each panel's y axis to its own data")
    parser.add_argument("--title", default="", help="figure title")
    args = parser.parse_args()

    if args.baseline_player and not args.baseline_experiment:
        parser.error("--baseline-player requires --baseline-experiment")
    if args.ncols < 1 or min(args.panel_size) <= 0 or args.dpi < 1:
        parser.error("--ncols, --panel-size and --dpi must be positive")

    op = build_op()
    harness = training_from_args(op, args)
    results = harness.get_results()
    if not results.runs or not harness.tester_configs:
        raise SystemExit("the training selection is empty; check --experiment / --games / --testers")
    metric = MetricName.from_value(args.metric)
    assert metric is not None  # argparse choices already checked this
    testers = [op.registry.player_config(value) for value in harness.tester_configs]

    baseline_results = None
    baseline_player = None
    if args.baseline_experiment:
        benchmark = Benchmark.load_experiment(op, args.baseline_experiment)
        players = benchmark.player_configs
        if args.baseline_player:
            selected = op.registry.player_config(args.baseline_player)
            if selected.hash not in {op.registry.player_config(value).hash for value in players}:
                parser.error("--baseline-player is not in the selected benchmark experiment")
            baseline_player = selected
        elif len(players) == 1:
            baseline_player = op.registry.player_config(players[0])
        else:
            parser.error("--baseline-player is required when the benchmark has several players")
        games = list(dict.fromkeys(run.game.to_string() for run in results.runs))
        benchmark = Benchmark.load_experiment(
            op, args.baseline_experiment, game_override=games, player_override=[baseline_player.to_string()], opponent_override=harness.tester_configs
        )
        baseline_results = benchmark.get_results()

    analysis = TrainingAnalysis(results, op.registry, confidence=args.confidence, include_fails=args.include_fails)
    line_options = LineOptions(show_ci=not args.no_ci, show_markers=not args.no_markers, ylim=tuple(args.ylim) if args.ylim else None)
    panels = []
    games = {run.game.to_string(): run.game for run in results.runs}
    for game in games.values():
        for tester in testers:
            curves = [analysis.curve(run, tester) for run in results.runs if run.game.to_string() == game.to_string()]
            reference = None
            if baseline_results is not None and baseline_player is not None:
                reference = benchmark_reference(
                    baseline_results, game, baseline_player, tester, metric, op.registry, MetricOptions(confidence=args.confidence, include_fails=args.include_fails)
                )
            reference_label = f"Benchmark: {player_label(baseline_player)}" if baseline_player is not None else "Benchmark reference"
            panels.append(training_panel(curves, metric, reference, line_options, f"{game.to_string()} vs {tester.to_string()}", reference_label))

    figure = Figure(
        panels=panels,
        ncols=args.ncols,
        panel_size=Layout(*args.panel_size),
        style=Style(font_size=args.font_size, dpi=args.dpi),
        suptitle=args.title,
        legend=LegendSpec(columns=args.legend_cols, show_markers=not args.no_markers),
        share_y=not args.free_y,
    )
    path = Path(args.out) if args.out else TrainingHarnessPathBuilder().plots_dir / f"{harness.experiment}_{metric.value}.png"
    if path.exists():
        raise SystemExit(f"output already exists: {path}; choose a new --out path")
    render(figure, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
