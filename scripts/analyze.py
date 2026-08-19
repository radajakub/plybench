"""Compute and print benchmark statistics from persisted results (in memory; nothing is written).

Prints per-matchup metrics with confidence intervals. Solvable games also get optimality/regret unless
`--no-quality` is passed.

Examples:
    uv run python scripts/analyze.py --experiment my_experiment
    uv run python scripts/analyze.py --name smoke --games tic_tac_toe: --players random:distribution=uniform \\
        --opponents optimal:stochastic=True --num-games 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared import add_source_args, benchmark_from_args, build_op  # noqa: E402

from plybench.analysis import BenchmarkAnalysis  # noqa: E402
from plybench.analysis.stats.matchup_stats import MatchupStats  # noqa: E402


def _print_matchup(stats: MatchupStats) -> None:
    print(f"\n{stats.game.to_string()}  |  {stats.i.to_string()}  vs  {stats.o.to_string()}  |  n={stats.n_games}")
    for name, bundle in stats.metrics.combined.metrics.items():
        print(f"  {name.value:28s} {bundle.fmt(10, interval=True, count=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--confidence", type=float, default=0.95, help="confidence level (default 0.95)")
    parser.add_argument("--include-fails", action="store_true", help="fold the player's own failed games into loss rate")
    parser.add_argument("--no-quality", action="store_true", help="skip optimality/regret (no minimax replay)")
    args = parser.parse_args()

    op = build_op()
    results = benchmark_from_args(op, args).get_results()
    registry = None if args.no_quality else op.registry
    analysis = BenchmarkAnalysis(results, registry, confidence=args.confidence, include_fails=args.include_fails)

    matchups = [stats for stats in analysis.analyze() if stats.completed]
    for stats in matchups:
        _print_matchup(stats)
    print(f"\n{len(matchups)} matchup(s) analyzed.")


if __name__ == "__main__":
    main()
