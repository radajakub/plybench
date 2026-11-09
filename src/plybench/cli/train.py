"""Train a learnable player: for every (trainee x trainer x game x schedule x replicate) run, play the
epochs in order and evaluate each epoch's frozen checkpoint against every tester, persisting the
results under `results/training/` (resumable — completed epochs and rounds are skipped).

Examples:
    uv run plybench-train --experiment test
    uv run plybench-train --name smoke --games tic_tac_toe: --trainees memory:eps=0.3 \\
        --trainers optimal:stochastic=True,eps=0.2 --testers optimal:stochastic=True \\
        --schedules num_epochs=10,num_training_games=50,num_test_games=50 --concurrency 4

Training phases are sequential by construction: one learner instance plays the epoch's games in order
and is updated after each. `--concurrency` is the ceiling on in-flight requests per provider;
`--rounds-concurrency` paces the rounds within one evaluation phase, and `--runs-concurrency` caps how
many runs are in flight at once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import AbstractContextManager, nullcontext

import clankers
from clankers.core.models import describe
from plyllm import DEFAULT_CONCURRENCY, LLM

from plybench.analysis.training import TrainingAnalysis
from plybench.analysis.usage import TokenUsage, calls_cost, calls_usage, learning_calls
from plybench.callbacks.console_callbacks import console_training_callbacks
from plybench.callbacks.notification_callbacks import notification_training_callbacks
from plybench.callbacks.training_callbacks import TrainingCallbacks
from plybench.cli.shared import add_limits_args, add_training_source_args, build_op, training_from_args, warn_if_unconfigured
from plybench.common.enums import MetricName
from plybench.harness.plan import training_plan
from plybench.harness.status import check_summaries, training_status
from plybench.harness.training.results import TrainingResults
from plybench.harness.training.training import TrainingHarness
from plybench.registry import Registry


def report_curves(results: TrainingResults, registry: Registry, tester_configs: list[str], metric: MetricName) -> None:
    analysis = TrainingAnalysis(results, registry)
    for tester_config in tester_configs:
        tester = registry.player_config(tester_config)
        for run in results.runs:
            curve = analysis.curve(run, tester)
            print(f"\n{run.game.path}: {run.trainee.path} vs {run.trainer.path} rep{run.replicate} — {metric.value} vs {tester_config}")
            for point in curve.points:
                bundle = point.bundle(metric)
                interval = f" [{bundle.wilson.lower:.2f}, {bundle.wilson.upper:.2f}]" if bundle.wilson is not None else ""
                print(f"  epoch {point.epoch:>2}: {bundle.value:.3f}{interval}  (n={bundle.n})")
            print(f"  delta: {curve.delta(metric):+.3f}")


def report_learning_cost(harness: TrainingHarness, llm: LLM) -> None:
    # the calls a learner made in its proposals, per epoch; the moves' tokens are in the game records, as for any player
    for run in harness.runs:
        epochs = [(index, learning_calls(harness.path_builder.learner_dir(harness.experiment, run, index))) for index in range(1, run.training_config.num_epochs + 1)]
        if not any(calls for _, calls in epochs):
            continue
        print(f"\n{run.game.path}: {run.trainee.path} vs {run.trainer.path} rep{run.replicate} -- learning calls")
        for index, calls in epochs:
            usage = sum(calls_usage(calls).values(), TokenUsage())
            cost, unpriced = calls_cost(llm, calls)
            roles = ", ".join(f"{role} {role_usage.calls}" for role, role_usage in sorted(calls_usage(calls).items()))
            priced = f"${cost:.4f}" + (f" (+{unpriced} unpriced)" if unpriced else "")
            print(f"  epoch {index:>2}: {usage.calls} calls, {usage.tokens.input_tokens} in / {usage.tokens.output_tokens} out tokens, {priced}  [{roles}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_source_args(parser)
    parser.add_argument("--sync", action="store_true", help="run the runs sequentially instead of concurrently (epochs within a run are always sequential)")
    parser.add_argument(
        "--concurrency", type=int, help=f"max in-flight requests per provider -- set this to your API rate quota (default: the provider's own, {DEFAULT_CONCURRENCY} for most)"
    )
    parser.add_argument("--rounds-concurrency", type=int, help="max concurrent rounds per evaluation phase (default: --concurrency); the provider limit still caps actual requests")
    parser.add_argument("--runs-concurrency", type=int, help="max runs in flight at once (default: all of them)")
    parser.add_argument("--metric", type=str, default=MetricName.WIN_RATE.value, choices=[name.value for name in MetricName], help="metric to print the learning curve on")
    parser.add_argument("--notify", action="store_true", help="push a notification on each epoch end, a final summary, and on failure (needs NTFY_URL and NTFY_TOPIC)")
    parser.add_argument("--quiet", action="store_true", help="print only each phase's outcomes and the checkpoints, not the progress of every game")
    parser.add_argument(
        "--check", action="store_true", help="check the experiment and print what it would play, then exit: nothing is played and no model is called (exit 1 on a problem)"
    )
    parser.add_argument("--json", action="store_true", help="with --check: print the plan as JSON")
    parser.add_argument("--status", action="store_true", help="print the saved games and their metrics as JSON, then exit: nothing is played (for tools that poll a run)")
    parser.add_argument(
        "--summaries",
        choices=["check", "write"],
        help="compare each matchup's stored summary (and a training phase's place) in its metadata.json with its saved games, then exit (exit 1 on a "
        "difference); write: store them where they differ, changing nothing else",
    )
    add_limits_args(parser)
    args = parser.parse_args()

    # None leaves each provider at its own default (Metacentrum takes fewer parallel calls than the rest)
    concurrency = args.concurrency
    rounds_concurrency = args.rounds_concurrency if args.rounds_concurrency is not None else (concurrency if concurrency is not None else DEFAULT_CONCURRENCY)

    op = build_op(concurrency=concurrency, limits=args.limits)
    harness = training_from_args(op, args)

    if args.status:
        print(json.dumps(training_status(harness)))
        return

    if args.summaries:
        problems = check_summaries(harness, write=args.summaries == "write")
        print("\n".join(problems) if problems else "every summary matches its games")
        if args.summaries == "write":
            print(f"wrote {len(problems)} summaries")
        raise SystemExit(1 if problems and args.summaries == "check" else 0)

    if args.check:
        plan = training_plan(harness)
        print(json.dumps(plan.to_dict(), indent=2) if args.json else plan.summary())
        raise SystemExit(1 if plan.problems else 0)

    callbacks = console_training_callbacks(progress=not args.quiet)
    notify: AbstractContextManager[object] = nullcontext()
    if args.notify:
        warn_if_unconfigured()
        notification_callbacks, progress = notification_training_callbacks(harness.experiment)
        callbacks = TrainingCallbacks.combine(callbacks, notification_callbacks)
        # clankers reports the run's start, duration and outcome; the builders add how far it got, a crash included
        label = f"[{harness.experiment}] training"
        notify = clankers.Engage(
            label,
            success=lambda: f"{label} finished: {progress()}",
            failure=lambda error: f"{label} crashed at {progress()}: {describe(error)}",
        )

    with notify:
        results = asyncio.run(harness.run(sync=args.sync, concurrency=rounds_concurrency, runs_concurrency=args.runs_concurrency, training_callbacks=callbacks))

    metric = MetricName.from_value(args.metric)
    if metric is None:
        parser.error(f"Invalid metric: {args.metric}")

    report_curves(results, op.registry, harness.tester_configs, metric)
    report_learning_cost(harness, op.llm)
    print(f"\nresults under results/training/{harness.experiment}/")


if __name__ == "__main__":
    main()
