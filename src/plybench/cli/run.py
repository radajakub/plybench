"""Run a benchmark: play every (player x opponent x game) matchup for `num_games` rounds and persist
the results under `results/benchmarks/` (resumable — already-completed rounds are skipped).

Examples:
    uv run plybench-run --experiment my_experiment
    uv run plybench-run --name smoke --games tic_tac_toe: --players random:distribution=uniform \\
        --opponents optimal:stochastic=True --num-games 10 --concurrency 4

`--concurrency` is the hard ceiling on in-flight requests per provider (match it to your API rate quota);
`--rounds-concurrency` only paces rounds within each matchup and defaults to the same value.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import AbstractContextManager, nullcontext

import clankers
from clankers.core.models import describe
from plyllm import DEFAULT_CONCURRENCY

from plybench.callbacks.benchmark_callbacks import BenchmarkCallbacks
from plybench.callbacks.console_callbacks import console_benchmark_callbacks
from plybench.callbacks.notification_callbacks import notification_benchmark_callbacks
from plybench.cli.shared import add_limits_args, add_source_args, benchmark_from_args, build_op, warn_if_unconfigured
from plybench.harness.plan import benchmark_plan
from plybench.harness.status import benchmark_status, check_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--sync", action="store_true", help="run matchups sequentially instead of concurrently (rounds within a matchup still run concurrently)")
    parser.add_argument(
        "--concurrency", type=int, help=f"max in-flight requests per provider -- set this to your API rate quota (default: the provider's own, {DEFAULT_CONCURRENCY} for most)"
    )
    parser.add_argument("--rounds-concurrency", type=int, help="max concurrent rounds per matchup (default: --concurrency); the provider limit still caps actual requests")
    parser.add_argument("--notify", action="store_true", help="push a notification on each matchup end, a final summary, and on failure (needs NTFY_URL and NTFY_TOPIC)")
    parser.add_argument("--quiet", action="store_true", help="print only the start and the outcomes of each matchup, not the progress of every game")
    parser.add_argument(
        "--check", action="store_true", help="check the experiment and print what it would play, then exit: nothing is played and no model is called (exit 1 on a problem)"
    )
    parser.add_argument("--json", action="store_true", help="with --check: print the plan as JSON")
    parser.add_argument("--status", action="store_true", help="print the saved games and their metrics as JSON, then exit: nothing is played (for tools that poll a run)")
    parser.add_argument(
        "--summaries",
        choices=["check", "write"],
        help="compare each matchup's stored summary (in its metadata.json) with its saved games, then exit (exit 1 on a difference); write: store the "
        "games' summary where it differs, changing nothing else",
    )
    add_limits_args(parser)
    args = parser.parse_args()

    # None leaves each provider at its own default (Metacentrum takes fewer parallel calls than the rest)
    concurrency = args.concurrency
    rounds_concurrency = args.rounds_concurrency if args.rounds_concurrency is not None else (concurrency if concurrency is not None else DEFAULT_CONCURRENCY)

    op = build_op(concurrency=concurrency, limits=args.limits)
    benchmark = benchmark_from_args(op, args)

    if args.status:
        print(json.dumps(benchmark_status(benchmark)))
        return

    if args.summaries:
        problems = check_summaries(benchmark, write=args.summaries == "write")
        print("\n".join(problems) if problems else "every summary matches its games")
        if args.summaries == "write":
            print(f"wrote {len(problems)} summaries")
        raise SystemExit(1 if problems and args.summaries == "check" else 0)

    if args.check:
        plan = benchmark_plan(benchmark)
        print(json.dumps(plan.to_dict(), indent=2) if args.json else plan.summary())
        raise SystemExit(1 if plan.problems else 0)

    callbacks = console_benchmark_callbacks(progress=not args.quiet)
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
