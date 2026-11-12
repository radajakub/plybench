from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from plybench.analysis.replay import ReplayerCache
from plybench.analysis.summary import CONFIDENCE
from plybench.callbacks.benchmark_callbacks import BenchmarkCallbacks
from plybench.callbacks.training_callbacks import TrainingCallbacks
from plybench.configs.names import short_names
from plybench.configs.training_run import Epoch, EpochPhase, TrainingRun
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.plan import optimal_outcomes_of
from plybench.harness.training.results import EpochResults
from plybench.harness.training.training import TrainingHarness
from plybench.report import Item, Report, ReportFile, estimate
from plybench.trackers.result_tracker import ResultTracker

# The report (plybench.report) of plybench-run and plybench-train with `--job-report`: the matchups or training runs this process plays (those with games
# left when it starts; what is saved already is not shown), rewritten while it plays from the trackers it holds (each one's running summary), so a tool
# reads one file. A benchmark: the outcomes per matchup and a table of win rate and tokens. A training: a grid of the epochs, the outcome measures of
# the tests per tester (with their interval) and of the training games (without: the learner changes during them), and the output tokens of the tests.
# Each point of a training's outcome measures carries the learner's moves in that phase (`moves`), counted over its decisions, with how many of them
# the solver rates optimal in a solvable game.

OUTCOMES = [
    {"key": "win", "label": "win", "role": "good"},
    {"key": "opponent_fail", "label": "opponent failed", "role": "good_other"},
    {"key": "draw", "label": "draw", "role": "neutral"},
    {"key": "loss", "label": "loss", "role": "bad"},
    {"key": "fail", "label": "failed", "role": "bad_other"},
]
# the outcome measures of a training, each with its value under the optimal policy's outcomes (a score counts a draw as half a win)
MEASURES = [
    ("win_rate", "win rate", lambda outcomes: outcomes["win"]),
    ("draw_rate", "draw rate", lambda outcomes: outcomes["draw"]),
    ("loss_rate", "loss rate", lambda outcomes: outcomes["loss"]),
    ("score", "score", lambda outcomes: outcomes["win"] + 0.5 * outcomes["draw"]),
]
INTERVAL_NOTE = f"{round(CONFIDENCE * 100)}% intervals: Wilson for rates, t for means"


def _metric(tracker: ResultTracker, name: str) -> dict[str, Any] | None:
    return tracker.summary.get(name) if tracker.summary is not None and tracker.completed else None


def _estimate(tracker: ResultTracker, name: str, interval: bool = True) -> dict[str, Any] | None:
    entry = _metric(tracker, name)
    if entry is None:
        return None
    return estimate(entry["value"], entry["lower"], entry["upper"]) if interval else estimate(entry["value"])


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


class BenchmarkReporter:
    def __init__(self, benchmark: Benchmark, path: Path) -> None:
        self.benchmark = benchmark
        # the matchups with games left, as the metadata says now; a matchup's own tracker replaces its entry once it starts
        loaded = benchmark.get_results(progress=False, games=False).trackers
        self.trackers: dict[Path, ResultTracker] = {tracker.metadata_path: tracker for tracker in loaded if not tracker.is_complete()}
        self.optimal = optimal_outcomes_of(benchmark.op, benchmark.game_configs, [])
        self.file = ReportFile(path, self.build)

    def callbacks(self) -> BenchmarkCallbacks:
        def started(tracker: ResultTracker, *_: object) -> None:
            if tracker.metadata_path in self.trackers:
                self.trackers[tracker.metadata_path] = tracker
            self.file.changed()

        return BenchmarkCallbacks(matchup_start_callback=started, round_complete_callback=lambda *_: self.file.changed())

    def _key(self, tracker: ResultTracker) -> str:
        return tracker.base_path.relative_to(self.benchmark.path_builder.results_dir / self.benchmark.experiment).as_posix()

    def build(self) -> Report:
        report = Report(labels=self.benchmark.labels, prefixes={"opponent": "vs "})
        trackers = list(self.trackers.values())
        for tracker in trackers:
            configs = {"game": tracker.game.to_string(), "player": tracker.i.to_string(), "opponent": tracker.o.to_string()}
            report.items.append(Item(self._key(tracker), configs, tracker.n, len(tracker.completed)))
        if not trackers:
            return report

        bars = []
        for tracker in trackers:
            values: dict[str, float] = {}
            if _metric(tracker, "win_rate") is not None:
                parts = {name: (_metric(tracker, f"{name}_rate") or {}).get("value") or 0.0 for name in ("win", "draw", "loss", "fail")}
                # the rest is the opponent's failures, which no rate names
                values = parts | {"opponent_fail": max(0.0, 1.0 - sum(parts.values()))}
            bars.append({"item": self._key(tracker), "values": values})
        report.add_bars("outcomes", OUTCOMES, bars)

        optimal = {game: outcomes for game, outcomes in self.optimal.items()}
        columns: list[dict[str, Any]] = [{"key": "item", "label": "matchup", "name": True}, {"key": "games", "label": "games"}]
        columns.append({"key": "win_rate", "label": f"win rate [{round(CONFIDENCE * 100)}%]", "rate": True})
        if optimal:
            columns.append({"key": "optimal", "label": "optimal win / draw / loss"})
        columns += [{"key": "input_tokens", "label": "input tokens/game", "digits": 0}, {"key": "output_tokens", "label": "output tokens/game", "digits": 0}]
        rows = []
        for tracker in trackers:
            row: dict[str, Any] = {"item": self._key(tracker), "games": f"{len(tracker.completed)}/{tracker.n}", "win_rate": _estimate(tracker, "win_rate")}
            if optimal:
                outcomes = optimal.get(tracker.game.to_string())
                row["optimal"] = " / ".join(_percent(outcomes[result]) for result in ("win", "draw", "loss")) if outcomes else None
            row["input_tokens"] = (_metric(tracker, "input_tokens_per_game") or {}).get("value")
            row["output_tokens"] = (_metric(tracker, "output_tokens_per_game") or {}).get("value")
            rows.append(row)
        report.add_table("matchups", columns, rows, note=INTERVAL_NOTE)
        return report


class TrainingReporter:
    def __init__(self, harness: TrainingHarness, path: Path) -> None:
        self.harness = harness
        self.testers = [harness.op.registry.player_config(tester).to_string() for tester in harness.tester_configs]
        # the runs with games left, each with its epochs as the metadata says now; a phase's own tracker replaces its entry once the phase starts
        self.runs: list[tuple[TrainingRun, list[EpochResults]]] = []
        for run in harness.runs:
            epochs = [harness.load_epoch(Epoch(run, index), games=False) for index in range(run.training_config.num_epochs + 1)]
            if any(not tracker.is_complete() for epoch in epochs for tracker in self._phases(epoch)):
                self.runs.append((run, epochs))
        self.live: dict[Path, ResultTracker] = {}
        # the optimum of every game an epoch plays, which is the run's game unless it changes at an epoch
        self.optimal = optimal_outcomes_of(harness.op, self._epoch_games([run for run, _ in self.runs]), [])
        self.replayers = ReplayerCache(harness.op.registry)
        # the learner's moves of each recorded game, as (move, optimal or None), by the phase's metadata and the round: a game is read and replayed once
        self.moves: dict[tuple[Path, int], list[tuple[str, bool | None]]] = {}
        self.file = ReportFile(path, self.build)

    @staticmethod
    def _epoch_games(runs: list[TrainingRun]) -> list[str]:
        return list(dict.fromkeys(Epoch(run, index).game.to_string() for run in runs for index in range(run.training_config.num_epochs + 1)))

    @staticmethod
    def _phases(epoch: EpochResults) -> list[ResultTracker]:
        return ([epoch.train] if epoch.train is not None else []) + epoch.tests

    def callbacks(self) -> TrainingCallbacks:
        def started(tracker: ResultTracker, phase: EpochPhase) -> None:
            self.live[tracker.metadata_path] = tracker
            self.file.changed()

        return TrainingCallbacks(phase_start_callback=started, round_complete_callback=lambda *_: self.file.changed())

    def _tracker(self, tracker: ResultTracker) -> ResultTracker:
        return self.live.get(tracker.metadata_path, tracker)

    def _key(self, run: TrainingRun) -> str:
        return self.harness.path_builder.run_dir(self.harness.experiment, run).relative_to(self.harness.path_builder.results_dir / self.harness.experiment).as_posix()

    def build(self) -> Report:
        report = Report(labels=self.harness.labels, prefixes={"opponent": "vs ", "replicate": "rep "})
        for run, epochs in self.runs:
            phases = [self._tracker(tracker) for epoch in epochs for tracker in self._phases(epoch)]
            configs = {
                "game": run.game.to_string(),
                "player": run.trainee.to_string(),
                "opponent": run.trainer.to_string(),
                "schedule": run.training_config.to_string(),
                "replicate": str(run.replicate),
            }
            key = self._key(run)
            report.items.append(Item(key, configs, sum(t.n for t in phases), sum(len(t.completed) for t in phases), group=key.rsplit("/", 1)[0]))
        if not self.runs:
            return report

        width = max(run.training_config.num_epochs for run, _ in self.runs) + 1
        grid = []
        for (run, epochs), item in zip(self.runs, report.items, strict=True):
            cells = []
            for epoch in epochs:
                phases = [self._tracker(tracker) for tracker in self._phases(epoch)]
                cells.append({"done": sum(len(t.completed) for t in phases), "total": sum(t.n for t in phases)})
            grid.append({"item": item.key, "cells": cells})
        report.add_grid("epochs", [str(index) for index in range(width)], grid, note="one cell per epoch (epoch 0 only tests): empty, partly played, done")

        tester_names, _ = short_names(self.testers, self.harness.labels)
        for tester in self.testers:
            group = f"tests vs {tester_names[tester]}"
            for metric, label, optimal in MEASURES:
                pick = self._test_of(tester)
                series = [self._series(item.key, epochs, pick, metric) for (_, epochs), item in zip(self.runs, report.items, strict=True)]
                report.add_line(f"{label}, {group}", "epoch", label, True, series, self._references(optimal), group=group, measure=label, note=INTERVAL_NOTE)
        group = "training games vs the trainer"
        for metric, label, optimal in MEASURES:
            series = [self._series(item.key, epochs, lambda epoch: epoch.train, metric, interval=False) for (_, epochs), item in zip(self.runs, report.items, strict=True)]
            report.add_line(
                f"{label}, {group}", "epoch", label, True, series, self._references(optimal), group=group, measure=label, note="no interval: the learner changes within the epoch"
            )
        series = [{"item": item.key, "points": self._pooled(epochs, "output_tokens_per_game")} for (_, epochs), item in zip(self.runs, report.items, strict=True)]
        report.add_line("output tokens per game, tests (all testers)", "epoch", "output tokens per game", False, series)
        return report

    @staticmethod
    def _test_of(tester: str) -> Callable[[EpochResults], ResultTracker | None]:
        return lambda epoch: next((tracker for tracker in epoch.tests if tracker.o.to_string() == tester), None)

    def _series(self, key: str, epochs: list[EpochResults], pick: Callable[[EpochResults], ResultTracker | None], metric: str, interval: bool = True) -> dict[str, Any]:
        points = []
        for epoch in epochs:
            tracker = pick(epoch)
            tracker = self._tracker(tracker) if tracker is not None else None
            value = _estimate(tracker, metric, interval) if tracker is not None else None
            if tracker is not None and value is not None and value["value"] is not None:
                points.append({"x": epoch.epoch.index, **{name: entry for name, entry in value.items() if entry is not None}, "moves": self._moves(tracker)})
        return {"item": key, "points": points}

    def _moves(self, tracker: ResultTracker) -> list[dict[str, Any]]:
        # the learner's moves in the phase's recorded games, most frequent first; `optimal` only where the solver rated every one of them
        counts: dict[str, list[int]] = {}
        rated = True
        for game_round in sorted(tracker.completed):
            key = (tracker.metadata_path, game_round)
            if key not in self.moves:
                game = tracker.games[game_round - 1] or tracker.load_game(game_round)
                if game is None:
                    continue
                optimal: dict[int, bool] = {}
                if self.harness.op.registry.solvable(tracker.game.key):
                    try:
                        optimal = {stats.seq: stats.is_optimal for stats in self.replayers(tracker.game).replay_stats(game, tracker.i)}
                    except Exception:
                        # any error: the moves are still counted, without the solver's rating
                        optimal = {}
                self.moves[key] = [(step.move, optimal.get(step.seq)) for step in game.steps if step.player_hash == tracker.i.hash]
            for move, is_optimal in self.moves[key]:
                entry = counts.setdefault(move, [0, 0])
                entry[0] += 1
                entry[1] += bool(is_optimal)
                rated = rated and is_optimal is not None
        ordered = sorted(counts.items(), key=lambda item: (-item[1][0], item[0]))
        return [{"move": move, "count": count, **({"optimal": optimal} if rated else {})} for move, (count, optimal) in ordered]

    def _pooled(self, epochs: list[EpochResults], metric: str) -> list[dict[str, Any]]:
        # the tests of an epoch as one mean: each tester's mean weighted by its count (exact for means)
        points = []
        for epoch in epochs:
            entries = [entry for tracker in epoch.tests if (entry := _metric(self._tracker(tracker), metric)) is not None and entry["value"] is not None and entry["n"]]
            n = sum(entry["n"] for entry in entries)
            if n:
                points.append({"x": epoch.epoch.index, "value": sum(entry["value"] * entry["n"] for entry in entries) / n})
        return points

    def _references(self, optimal: Callable[[dict[str, float]], float]) -> list[dict[str, Any]]:
        # the optimal policy's value of the reported runs' games in a measure, one line per distinct value, named by its game only when they differ
        games = self._epoch_games([run for run, _ in self.runs])
        by_value: dict[float, list[str]] = {}
        for game in games:
            if game in self.optimal:
                by_value.setdefault(optimal(self.optimal[game]), []).append(game)
        names, _ = short_names(games)
        lines = []
        for value, named in by_value.items():
            which = "" if len(by_value) == 1 else f" ({names[named[0]]})" if len(named) == 1 else f" ({len(named)} games)"
            lines.append({"label": f"optimal {_percent(value)}{which}", "value": value})
        return lines
