from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plybench.analysis.replay import ReplayerCache
from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.intervals import t_ci, wilson_ci
from plybench.analysis.summary import CONFIDENCE
from plybench.callbacks.benchmark_callbacks import BenchmarkCallbacks
from plybench.callbacks.training_callbacks import TrainingCallbacks
from plybench.configs.names import short_names
from plybench.configs.training_run import Epoch, EpochPhase, TrainingRun
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.plan import optimal_outcomes_of
from plybench.harness.training.results import EpochResults
from plybench.harness.training.training import TrainingHarness
from plybench.report import Item, Report, ReportFile, estimate, replicate_series
from plybench.trackers.result_tracker import ResultTracker

# The report (plybench.report) of plybench-run and plybench-train with `--job-report`: the matchups or training runs this process plays (those with games
# left when it starts; what is saved already is not shown), rewritten while it plays from the trackers it holds (each one's running summary), so a tool
# reads one file. A benchmark: the outcomes per matchup and a table of win rate and tokens. A training: a grid of the epochs, the outcome measures of
# the tests per tester (with their interval) and of the training games (without: the learner changes during them), and the output tokens of the tests.
# Each point of a training's outcome measures carries the learner's moves in that phase (`moves`), counted over its decisions, with how many of them
# the solver rates optimal in a solvable game. In a solvable game the same replay also gives the optimal play: per tester the optimal move rate and the
# regret per game of the tests, and for the training games the optimal move rate and the cumulative regret. Each line block names its `panel`.
# include_complete also includes finished work when rebuilding a report from saved results.

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


@dataclass(frozen=True)
class PhaseMoves:
    # the learner's moves in a phase's recorded games (most frequent first), the share the solver rates optimal with its Wilson interval (None unless
    # it rated every move) and each game's regret (None unless it rated every move)
    moves: list[dict[str, Any]]
    optimal: tuple[float, float, float] | None
    regrets: list[float] | None


class BenchmarkReporter:
    def __init__(self, benchmark: Benchmark, path: Path, *, include_complete: bool = False, item_keys: set[str] | None = None) -> None:
        self.benchmark = benchmark
        # the matchups with games left, as the metadata says now; a matchup's own tracker replaces its entry once it starts
        loaded = benchmark.get_results(progress=False, games=False).trackers
        self.trackers: dict[Path, ResultTracker] = {tracker.metadata_path: tracker for tracker in loaded if include_complete or not tracker.is_complete()}
        if item_keys is not None:
            self.trackers = {path: tracker for path, tracker in self.trackers.items() if self._key(tracker) in item_keys}
            missing = item_keys - {self._key(tracker) for tracker in self.trackers.values()}
            if missing:
                raise ValueError(f"report items are absent from the selection: {', '.join(sorted(missing))}")
        self.optimal = optimal_outcomes_of(benchmark.op, {game: game for game in benchmark.game_configs}, [])
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
    def __init__(self, harness: TrainingHarness, path: Path, *, include_complete: bool = False, item_keys: set[str] | None = None, include_moves: bool = True) -> None:
        self.harness = harness
        self.include_moves = include_moves
        self.testers = [harness.op.registry.player_config(tester).to_string() for tester in harness.tester_configs]
        # the runs with games left, each with its epochs as the metadata says now; a phase's own tracker replaces its entry once the phase starts
        self.runs: list[tuple[TrainingRun, list[EpochResults]]] = []
        for run in harness.runs:
            epochs = [harness.load_epoch(Epoch(run, index), games=False) for index in range(run.training_config.num_epochs + 1)]
            if include_complete or any(not tracker.is_complete() for epoch in epochs for tracker in self._phases(epoch)):
                self.runs.append((run, epochs))
        if item_keys is not None:
            self.runs = [(run, epochs) for run, epochs in self.runs if self._key(run) in item_keys]
            missing = item_keys - {self._key(run) for run, _ in self.runs}
            if missing:
                raise ValueError(f"report items are absent from the selection: {', '.join(sorted(missing))}")
        self.live: dict[Path, ResultTracker] = {}
        # the games the runs' epochs play, and per game the runs that play it; the runs are fixed here, so every build reuses them
        self.games = self._epoch_games([run for run, _ in self.runs])
        self.runs_of: dict[str, list[str]] = {}
        for run, _ in self.runs:
            for index in range(run.training_config.num_epochs + 1):
                played = self.runs_of.setdefault(Epoch(run, index).game.to_string(), [])
                if run.game.to_string() not in played:
                    played.append(run.game.to_string())
        # the optimum of every game an epoch plays, which is the run's game unless it changes at an epoch
        self.optimal = optimal_outcomes_of(harness.op, {game: game for game in self.games}, [])
        self.replayers = ReplayerCache(harness.op.registry)
        # the learner's moves of each recorded game, as (move, optimal, regret; both None when the solver did not rate it), by the phase's metadata and
        # the round: a game is read and replayed once
        self.moves: dict[tuple[Path, int], list[tuple[str, bool | None, float | None]]] = {}
        # a complete phase's moves, counted once by its metadata: it no longer changes, so a build reuses them, and its games leave self.moves
        self.finished: dict[Path, PhaseMoves] = {}
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

    def _settle(self) -> None:
        # a live phase that is complete is kept as a copy without its games: its moves are counted once (cached in self.moves), and the report keeps
        # no game in memory after its phase, which grew with every game played (about 9 KB a game: a 206k-game job ran out of memory at its end). The
        # harness's own tracker is not touched: the learner reads its games when the phase ends
        for path, tracker in list(self.live.items()):
            if not tracker.is_complete() or tracker.games.count(None) == len(tracker.games):
                continue
            if self.include_moves:
                self._phase(tracker)
            light = ResultTracker(
                tracker.experiment,
                tracker.i,
                tracker.o,
                tracker.game,
                tracker.n,
                set(tracker.completed),
                tracker.parser,
                tracker.path_builder,
                save_on_record=False,
                seed=tracker.seed,
                summary=tracker.summary,
                place=tracker.place,
            )
            self.live[path] = light

    def _key(self, run: TrainingRun) -> str:
        return self.harness.path_builder.run_dir(self.harness.experiment, run).relative_to(self.harness.path_builder.results_dir / self.harness.experiment).as_posix()

    def build(self) -> Report:
        self._settle()
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
        groups = {item.key: item.group for item in report.items}
        members: dict[str, list[dict[str, Any]]] = {}
        for row in grid:
            members.setdefault(groups[row["item"]] or row["item"], []).append(row)
        grouped_grid = []
        for key, rows in members.items():
            if len(rows) == 1:
                grouped_grid.append(rows[0])
                continue
            cells = [
                {field: sum(row["cells"][index][field] for row in rows if index < len(row["cells"])) for field in ("done", "total")}
                for index in range(max(len(row["cells"]) for row in rows))
            ]
            grouped_grid.append({"group": key, "cells": cells, "rows": rows})
        report.add_grid("epochs", [str(index) for index in range(width)], grouped_grid, note="one cell per epoch (epoch 0 only tests): empty, partly played, done")
        replicated = any(len(rows) > 1 for rows in members.values())
        replicate_note = f"{round(CONFIDENCE * 100)}% t intervals of the mean across available replicates; spread: 10–90%; no interval when n < 2"

        tester_names, _ = short_names(self.testers, self.harness.labels)
        for tester in self.testers:
            group = f"tests vs {tester_names[tester]}"
            for metric, label, optimal in MEASURES:
                pick = self._test_of(tester)
                series = [self._series(item.key, epochs, pick, metric) for (_, epochs), item in zip(self.runs, report.items, strict=True)]
                report.add_line(
                    f"{label}, {group}",
                    "epoch",
                    label,
                    True,
                    replicate_series(series, groups, True),
                    self._references(optimal),
                    group=group,
                    measure=label,
                    note=f"{replicate_note}; single runs: {INTERVAL_NOTE}" if replicated else INTERVAL_NOTE,
                    panel="tests",
                )
            if self.include_moves:
                pick = self._test_of(tester)
                rates = [self._optimal_series(item.key, epochs, pick, interval=True) for (_, epochs), item in zip(self.runs, report.items, strict=True)]
                regrets = [self._regret_series(item.key, epochs, pick) for (_, epochs), item in zip(self.runs, report.items, strict=True)]
                intervals = "Wilson for the rate (over moves), t for the regret (over games)"
                note = f"{replicate_note}; single runs: {intervals}" if replicated else f"{round(CONFIDENCE * 100)}% intervals: {intervals}"
                measures = [("optimal move rate", True, rates), ("regret per game", False, regrets)]
                self._add_optimal(report, f"optimal play, tests vs {tester_names[tester]}", measures, groups, "optimal", note)
        group = "training games vs the trainer"
        for metric, label, optimal in MEASURES:
            series = [self._series(item.key, epochs, lambda epoch: epoch.train, metric, interval=False) for (_, epochs), item in zip(self.runs, report.items, strict=True)]
            report.add_line(
                f"{label}, {group}",
                "epoch",
                label,
                True,
                replicate_series(series, groups, True),
                self._references(optimal),
                group=group,
                measure=label,
                note=f"{replicate_note}; single runs: no within-epoch interval" if replicated else "no interval: the learner changes within the epoch",
                panel="training",
            )
        if self.include_moves:
            rates = [self._optimal_series(item.key, epochs, lambda epoch: epoch.train, interval=False) for (_, epochs), item in zip(self.runs, report.items, strict=True)]
            regrets = [self._cumulative_regret(item.key, epochs) for (_, epochs), item in zip(self.runs, report.items, strict=True)]
            regret = "cumulative regret: the sum of V* - Q*(s, a) over the learner's moves in the training games up to the epoch"
            note = f"{replicate_note}; single runs: no interval; {regret}" if replicated else f"no interval; {regret}"
            self._add_optimal(report, group, [("optimal move rate", True, rates), ("cumulative regret", False, regrets)], groups, "training", note)
        series = [{"item": item.key, "points": self._pooled(epochs, "output_tokens_per_game")} for (_, epochs), item in zip(self.runs, report.items, strict=True)]
        report.add_line(
            "output tokens per game, tests (all testers)",
            "epoch",
            "output tokens per game",
            False,
            replicate_series(series, groups, False),
            note=replicate_note if replicated else None,
            panel="tokens",
        )
        return report

    @staticmethod
    def _add_optimal(report: Report, group: str, measures: list[tuple[str, bool, list[dict[str, Any]]]], groups: dict[str, str | None], panel: str, note: str) -> None:
        # the measures of the solver's rating as line blocks of one group; none when no run has a rated point (no solvable game, or nothing played yet)
        if not any(entry["points"] for _, _, series in measures for entry in series):
            return
        for label, rate, series in measures:
            # the optimal policy: every move optimal, no regret
            reference = {"label": f"optimal {_percent(1.0)}" if rate else "optimal 0", "value": 1.0 if rate else 0.0}
            report.add_line(f"{label}, {group}", "epoch", label, rate, replicate_series(series, groups, rate), [reference], group=group, measure=label, note=note, panel=panel)

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
                point = {"x": epoch.epoch.index, **{name: entry for name, entry in value.items() if entry is not None}}
                if self.include_moves:
                    point["moves"] = self._moves(tracker)
                points.append(point)
        return {"item": key, "points": points}

    def _rated(self, tracker: ResultTracker) -> list[list[tuple[str, bool | None, float | None]]]:
        # the learner's moves of each recorded game of the phase, as (move, optimal, regret), replayed once against the solver in a solvable game;
        # optimal and regret are None where the solver did not rate the game
        games = []
        for game_round in sorted(tracker.completed):
            key = (tracker.metadata_path, game_round)
            if key not in self.moves:
                game = tracker.games[game_round - 1] or tracker.load_game(game_round)
                if game is None:
                    continue
                rated: dict[int, tuple[bool, float]] = {}
                if self.harness.op.registry.solvable(tracker.game.key):
                    try:
                        rated = {stats.seq: (stats.is_optimal, stats.regret) for stats in self.replayers(tracker.game).replay_stats(game, tracker.i)}
                    except Exception:
                        # any error: the moves are still counted, without the solver's rating
                        rated = {}
                moves: list[tuple[str, bool | None, float | None]] = []
                for step in game.steps:
                    if step.player_hash == tracker.i.hash:
                        is_optimal, regret = rated.get(step.seq, (None, None))
                        moves.append((step.move, is_optimal, regret))
                self.moves[key] = moves
            games.append(self.moves[key])
        return games

    def _phase(self, tracker: ResultTracker) -> PhaseMoves:
        # a complete phase's moves from self.finished, counted there the first time; a live phase's counted from its games at every build
        finished = self.finished.get(tracker.metadata_path)
        if finished is not None:
            return finished
        counts: dict[str, list[int]] = {}
        rated = True
        flags: list[bool | None] = []
        regrets: list[float] | None = []
        for moves in self._rated(tracker):
            for move, is_optimal, _ in moves:
                entry = counts.setdefault(move, [0, 0])
                entry[0] += 1
                entry[1] += bool(is_optimal)
                rated = rated and is_optimal is not None
                flags.append(is_optimal)
            # a game's regret is the sum over the learner's moves
            if regrets is not None and any(regret is None for _, _, regret in moves):
                regrets = None
            elif regrets is not None:
                regrets.append(sum(regret for _, _, regret in moves if regret is not None))
        ordered = sorted(counts.items(), key=lambda item: (-item[1][0], item[0]))
        counted = [{"move": move, "count": count, **({"optimal": optimal} if rated else {})} for move, (count, optimal) in ordered]
        optimal_rate = wilson_ci(Distribution(float(bool(flag)) for flag in flags), CONFIDENCE).unwrap() if flags and rated else None
        phase = PhaseMoves(counted, optimal_rate, regrets)
        if tracker.is_complete():
            self.finished[tracker.metadata_path] = phase
            for game_round in tracker.completed:
                self.moves.pop((tracker.metadata_path, game_round), None)
        return phase

    def _moves(self, tracker: ResultTracker) -> list[dict[str, Any]]:
        # the learner's moves in the phase's recorded games, most frequent first; `optimal` only where the solver rated every one of them
        return self._phase(tracker).moves

    def _game_regrets(self, tracker: ResultTracker) -> list[float] | None:
        # each recorded game's regret (the sum over the learner's moves); None unless the solver rated every move of the phase
        return self._phase(tracker).regrets

    def _optimal_series(self, key: str, epochs: list[EpochResults], pick: Callable[[EpochResults], ResultTracker | None], interval: bool) -> dict[str, Any]:
        # per epoch the share of the learner's moves the solver rates optimal (Wilson over the moves), where it rated every move
        points = []
        for epoch in epochs:
            tracker = pick(epoch)
            if tracker is None:
                continue
            tracker = self._tracker(tracker)
            optimal_rate = self._phase(tracker).optimal
            if optimal_rate is None:
                continue
            value, lower, upper = optimal_rate
            points.append({"x": epoch.epoch.index, "value": value, **({"lower": lower, "upper": upper} if interval else {}), "moves": self._moves(tracker)})
        return {"item": key, "points": points}

    def _regret_series(self, key: str, epochs: list[EpochResults], pick: Callable[[EpochResults], ResultTracker | None]) -> dict[str, Any]:
        # per epoch the mean regret per game of the phase (t over the games), where the solver rated every move
        points = []
        for epoch in epochs:
            tracker = pick(epoch)
            if tracker is None:
                continue
            tracker = self._tracker(tracker)
            regrets = self._game_regrets(tracker)
            if not regrets:
                continue
            value, lower, upper = t_ci(Distribution(regrets), CONFIDENCE).unwrap()
            points.append({"x": epoch.epoch.index, "value": value, **({"lower": lower, "upper": upper} if len(regrets) > 1 else {}), "moves": self._moves(tracker)})
        return {"item": key, "points": points}

    def _cumulative_regret(self, key: str, epochs: list[EpochResults]) -> dict[str, Any]:
        # the regret of the training games summed from the first epoch to the end of each epoch (epoch 0 trains nothing: 0); stops at the first epoch
        # the solver did not rate, and at an epoch whose training has not started
        points = []
        total = 0.0
        for epoch in epochs:
            if epoch.train is None:
                points.append({"x": epoch.epoch.index, "value": total})
                continue
            tracker = self._tracker(epoch.train)
            regrets = self._game_regrets(tracker)
            if not regrets:
                break
            total += sum(regrets)
            points.append({"x": epoch.epoch.index, "value": total, "moves": self._moves(tracker)})
        # no rated training game yet (or no solver): no line, not a lone 0 at epoch 0
        return {"item": key, "points": points if any("moves" in point for point in points) else []}

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
        # the optimal policy's value of the reported runs' games in a measure, one line per distinct value, named by its game only when they differ;
        # `games`: the runs' games (Item.configs["game"]) it applies to, so a chart of some of the runs draws only theirs (a run that switches its
        # game at an epoch has the optimum of each game it plays)
        by_value: dict[float, list[str]] = {}
        for game in self.games:
            if game in self.optimal:
                by_value.setdefault(optimal(self.optimal[game]), []).append(game)
        names, _ = short_names(self.games)
        lines = []
        for value, named in by_value.items():
            which = "" if len(by_value) == 1 else f" ({names[named[0]]})" if len(named) == 1 else f" ({len(named)} games)"
            applies = list(dict.fromkeys(run for game in named for run in self.runs_of.get(game, [])))
            lines.append({"label": f"optimal {_percent(value)}{which}", "value": value, "games": applies})
        return lines
