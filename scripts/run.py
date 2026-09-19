"""Run a benchmark: play every (player x opponent x game) matchup for `num_games` rounds and persist
the results under `results/benchmarks/` (resumable — already-completed rounds are skipped).

Examples:
    uv run python scripts/run.py --experiment my_experiment
    uv run python scripts/run.py --name smoke --games tic_tac_toe: --players random:distribution=uniform \\
        --opponents optimal:stochastic=True --num-games 10 --concurrency 4

`--concurrency` is the hard ceiling on in-flight requests per provider (match it to your API rate quota);
`--rounds-concurrency` only paces rounds within each matchup and defaults to the same value.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import clankers
from clankers.core.models import describe

sys.path.insert(0, str(Path(__file__).parent))
from _shared import add_source_args, benchmark_from_args, build_op  # noqa: E402

from plybench.callbacks.benchmark_callbacks import BenchmarkCallbacks  # noqa: E402
from plybench.callbacks.console_callbacks import console_benchmark_callbacks  # noqa: E402
from plybench.callbacks.notification_callbacks import notification_benchmark_callbacks  # noqa: E402
from plybench.llm import DEFAULT_CONCURRENCY  # noqa: E402


def warn_if_unconfigured() -> None:
    # clankers builds its backend lazily and only warns when a send fails, so a missing NTFY_URL/NTFY_TOPIC
    # would otherwise go unnoticed until the first matchup ends
    try:
        _ = clankers.default().backend
    except ValueError as error:
        print(f"warning: --notify set but notifications are not configured ({error}); they will be skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--sync", action="store_true", help="run matchups sequentially instead of concurrently (rounds within a matchup still run concurrently)")
    parser.add_argument("--concurrency", type=int, help=f"max in-flight requests per provider -- set this to your API rate quota (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--rounds-concurrency", type=int, help="max concurrent rounds per matchup (default: --concurrency); the provider limit still caps actual requests")
    parser.add_argument("--notify", action="store_true", help="push a notification on each matchup end, a final summary, and on failure (needs NTFY_URL and NTFY_TOPIC)")
    args = parser.parse_args()

    concurrency = args.concurrency if args.concurrency is not None else DEFAULT_CONCURRENCY
    rounds_concurrency = args.rounds_concurrency if args.rounds_concurrency is not None else concurrency

    op = build_op(concurrency=concurrency)
    benchmark = benchmark_from_args(op, args)

    callbacks = console_benchmark_callbacks()
    notify: AbstractContextManager[object] = nullcontext()
    if args.notify:
        warn_if_unconfigured()
        notification_callbacks, progress = notification_benchmark_callbacks(benchmark.experiment)
        callbacks = BenchmarkCallbacks.combine(callbacks, notification_callbacks)
        # clankers reports the run's start, duration and outcome; the builders add how far it got, a crash included
        label = f"[{benchmark.experiment}] benchmark"
        notify = clankers.Engage(
            label,
            success=lambda: f"{label} finished: {progress()}",
            failure=lambda error: f"{label} crashed at {progress()}: {describe(error)}",
        )

    with notify:
        results = asyncio.run(benchmark.run(sync=args.sync, concurrency=rounds_concurrency, benchmark_callbacks=callbacks))

    complete = sum(1 for tracker in results.trackers if tracker.is_complete())
    print(f"\ndone: {complete}/{len(results.trackers)} matchups complete under results/benchmarks/{benchmark.experiment}/")


if __name__ == "__main__":
    main()
