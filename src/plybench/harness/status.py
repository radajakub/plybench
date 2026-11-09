from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from plybench.analysis.summary import CONFIDENCE, Summary, matches, summarize
from plybench.configs.parser import ConfigParser
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.training.training import TrainingHarness
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.result_tracker import ResultTracker

# How far an experiment has got and how it is doing: per matchup, or per training run, epoch and phase (training games, and the tests per tester), the
# games saved of those planned and the default metrics (outcomes, score, moves, tokens; Wilson interval for rates, t for means). For `--status` of
# plybench-run and plybench-train, which tools poll while a run plays. It reads each tracker's summary from its metadata.json (plybench.analysis.summary)
# and loads the games only of a tracker without one, or whose summary does not cover its completed games (metadata from before summaries).
# `--summaries check` compares every stored summary with one computed from the games; `--summaries write` stores the computed one where they differ,
# changing only the `summary` key of metadata.json.


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


def _trackers(source: Benchmark | TrainingHarness) -> list[ResultTracker]:
    # every tracker with saved games, its games loaded
    if isinstance(source, Benchmark):
        trackers = source.get_results(progress=False).trackers
    else:
        trackers = [tracker for epoch in source.get_results(progress=False).epochs for tracker in (epoch.train, *epoch.tests) if tracker is not None]
    return [tracker for tracker in trackers if tracker.completed]


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
    # the trackers whose stored summary differs from their games' (a missing one included), as "<metadata.json>: <why>"; with `write`, each one's
    # summary is replaced in its metadata.json and nothing else in the file changes
    problems = []
    for tracker in _trackers(source):
        computed = summarize([game for game in tracker.games if game is not None], tracker.i)
        if (why := _difference(tracker.summary, computed)) is None:
            continue
        problems.append(f"{tracker.metadata_path}: {why}")
        if write:
            _write_summary(tracker.metadata_path, computed)
    return problems


def _write_summary(path: Path, summary: Summary) -> None:
    # the file as it is (its keys in their order), with the summary replaced; written to a temporary file, then renamed over it
    data = json.loads(path.read_text())
    data["summary"] = summary
    temporary = path.with_suffix(".summary.tmp")
    temporary.write_text(json.dumps(data, indent=None))
    os.replace(temporary, path)


def check_folder(root: Path, parser: ConfigParser, write: bool = False) -> tuple[int, list[str]]:
    # check_summaries for every metadata.json under `root`, whatever experiment, line or run it belongs to (an experiment file names only its enabled
    # lines): its games are the game_<round>.json beside it, its player the i_config. The number of files checked, and the problems; a file whose
    # player or games cannot be read is a problem and is never written
    problems, files = [], sorted(root.rglob("metadata.json"))
    for path in files:
        try:
            data = json.loads(path.read_text())
            games = [GameTracker.from_dict(json.loads((path.parent / f"game_{game_round}.json").read_text()), parser) for game_round in sorted(int(x) for x in data["completed"])]
            computed = summarize(games, parser.player_config(data["i_config"]))
        except Exception as error:
            # any error: the check reports every file it cannot read, instead of stopping at the first
            problems.append(f"{path}: cannot read ({type(error).__name__}: {error})")
            continue
        if not games or (why := _difference(data.get("summary"), computed)) is None:
            continue
        problems.append(f"{path}: {why}")
        if write:
            _write_summary(path, computed)
    return len(files), problems
