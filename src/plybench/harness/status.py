from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from plybench.analysis.summary import CONFIDENCE, Summary, matches, summarize
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.training.training import TrainingHarness
from plybench.trackers.result_tracker import ResultTracker

# How far an experiment has got and how it is doing: per matchup, or per training run, epoch and phase (training games, and the tests per tester), the
# games saved of those planned and the default metrics (outcomes, score, moves, tokens; Wilson interval for rates, t for means). For `--status` of
# plybench-run and plybench-train, which tools poll while a run plays. It reads each tracker's summary from its metadata.json (plybench.analysis.summary)
# and loads the games only of a tracker without one, or whose summary does not cover its completed games (metadata from before summaries).
# `--summaries check` compares every stored summary with one computed from the games (and a training phase's place with where it belongs);
# `--summaries write` stores them where they differ, changing only the `summary` and `place` keys of metadata.json.


def _summary(tracker: ResultTracker) -> Summary | None:
    # the stored summary when it covers every completed game, else one computed from the games
    if not tracker.completed:
        return None
    if tracker.summary is not None and tracker.summary["win_rate"]["n"] == len(tracker.completed):
        return tracker.summary
    tracker.load_if_exists()
    return summarize([game for game in tracker.games if game is not None], tracker.i)


def _phase(tracker: ResultTracker) -> dict[str, Any]:
    summary = _summary(tracker)
    metrics = {name: {key: entry[key] for key in ("value", "n", "lower", "upper")} for name, entry in summary.items()} if summary else None
    return {"planned_games": tracker.n, "saved_games": len(tracker.completed), "metrics": metrics}


def benchmark_status(benchmark: Benchmark) -> dict[str, Any]:
    results = benchmark.get_results(progress=False, games=False)
    matchups = [{"game": tracker.game.to_string(), "player": tracker.i.to_string(), "opponent": tracker.o.to_string(), **_phase(tracker)} for tracker in results.trackers]
    return {"experiment": benchmark.experiment, "kind": "benchmark", "confidence": CONFIDENCE, "matchups": matchups}


def training_status(harness: TrainingHarness) -> dict[str, Any]:
    results = harness.get_results(progress=False, games=False)
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


def _trackers(source: Benchmark | TrainingHarness) -> list[tuple[ResultTracker, dict[str, Any] | None]]:
    # every tracker with saved games, its games loaded, with the place it should carry (a training phase's; None for a benchmark's matchup)
    if isinstance(source, Benchmark):
        pairs: list[tuple[ResultTracker, dict[str, Any] | None]] = [(tracker, None) for tracker in source.get_results(progress=False).trackers]
    else:
        pairs = []
        for epoch in source.get_results(progress=False).epochs:
            if epoch.train is not None:
                pairs.append((epoch.train, epoch.epoch.train.place(source.experiment)))
            pairs.extend((tracker, epoch.epoch.test(tracker.o).place(source.experiment)) for tracker in epoch.tests)
    return [(tracker, place) for tracker, place in pairs if tracker.completed]


def _difference(stored: Summary | None, computed: Summary) -> str | None:
    # why a stored summary is not the games' one, or None when it is
    if matches(stored, computed):
        return None
    if stored is None:
        return "no summary"
    if stored["win_rate"]["n"] != computed["win_rate"]["n"]:
        return f"a summary of {stored['win_rate']['n']} games, {computed['win_rate']['n']} saved"
    return "the summary differs from the games"


def check_summaries(source: Benchmark | TrainingHarness, write: bool = False) -> list[str]:
    # the trackers whose stored summary differs from their games' (a missing one included), or a training phase's whose place is missing or wrong, as
    # "<metadata.json>: <why>"; with `write`, each one's summary and place are replaced in its metadata.json and nothing else in the file changes
    problems = []
    for tracker, place in _trackers(source):
        computed = summarize([game for game in tracker.games if game is not None], tracker.i)
        reasons = [why] if (why := _difference(tracker.summary, computed)) else []
        if place is not None and tracker.place != place:
            reasons.append("no place" if tracker.place is None else "a wrong place")
        if not reasons:
            continue
        problems.append(f"{tracker.metadata_path}: {', '.join(reasons)}")
        if write:
            _write(tracker.metadata_path, computed, place)
    return problems


def _write(path: Path, summary: Summary, place: dict[str, Any] | None) -> None:
    # the file as it is (its keys in their order), with the summary and the place replaced; written to a temporary file, then renamed over it
    data = json.loads(path.read_text())
    data["summary"] = summary
    if place is not None:
        data["place"] = place
    temporary = path.with_suffix(".summary.tmp")
    temporary.write_text(json.dumps(data, indent=None))
    os.replace(temporary, path)
