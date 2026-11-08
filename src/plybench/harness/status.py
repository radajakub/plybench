from __future__ import annotations

import math
from typing import Any

from plybench.analysis.extractors.suite import default_suite
from plybench.analysis.statistics.intervals import t_ci, wilson_ci
from plybench.common.enums import CIFamily
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.training.training import TrainingHarness
from plybench.trackers.result_tracker import ResultTracker

# How far an experiment has got and how it is doing, from its saved games only: per matchup, or per training run, epoch and phase, the games saved of
# those planned and the default metrics (outcomes, score, moves, tokens). For `--status` of plybench-run and plybench-train, which tools poll while a run
# plays, so it stays cheap: no optimality replay, no recognition from reasoning traces, and a t interval for means instead of the BCa bootstrap.

CONFIDENCE = 0.95


def _number(value: float) -> float | None:
    # JSON has no NaN or infinity (a mean over no games, or tokens a player never reports)
    return value if math.isfinite(value) else None


def _metrics(tracker: ResultTracker) -> dict[str, dict[str, Any]] | None:
    games = [game for game in tracker.games if game is not None]
    if not games:
        return None
    metrics: dict[str, dict[str, Any]] = {}
    for extractor in default_suite():
        distribution = extractor.extract(games, tracker.i)
        interval = wilson_ci(distribution, CONFIDENCE) if extractor.family == CIFamily.RATIO else t_ci(distribution, CONFIDENCE)
        metrics[extractor.name.value] = {"value": _number(interval.value), "n": distribution.n, "lower": _number(interval.lower), "upper": _number(interval.upper)}
    return metrics


def _phase(tracker: ResultTracker) -> dict[str, Any]:
    return {"planned_games": tracker.n, "saved_games": sum(1 for game in tracker.games if game is not None), "metrics": _metrics(tracker)}


def benchmark_status(benchmark: Benchmark) -> dict[str, Any]:
    results = benchmark.get_results(progress=False)
    matchups = [{"game": tracker.game.to_string(), "player": tracker.i.to_string(), "opponent": tracker.o.to_string(), **_phase(tracker)} for tracker in results.trackers]
    return {"experiment": benchmark.experiment, "kind": "benchmark", "confidence": CONFIDENCE, "matchups": matchups}


def training_status(harness: TrainingHarness) -> dict[str, Any]:
    results = harness.get_results(progress=False)
    # as the tests are keyed, so a tool can match them
    testers = [harness.op.registry.player_config(tester).to_string() for tester in harness.tester_configs]
    runs = []
    for run in results.runs:
        epochs = [
            {
                "epoch": epoch.epoch.index,
                "train": _phase(epoch.train) if epoch.train is not None else None,
                "tests": {tracker.o.to_string(): _phase(tracker) for tracker in epoch.tests},
            }
            for epoch in results.for_run(run)
        ]
        runs.append({**run.to_dict(), "epochs": epochs})
    return {"experiment": harness.experiment, "kind": "training", "confidence": CONFIDENCE, "testers": testers, "runs": runs}
