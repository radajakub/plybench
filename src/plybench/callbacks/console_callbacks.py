from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from plybench.callbacks.benchmark_callbacks import BenchmarkCallbacks
from plybench.callbacks.training_callbacks import TrainingCallbacks
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.configs.training_run import Epoch, EpochPhase, TrainingRun, TrainingRunKey
from plybench.harness.training.results import TrainingResults
from plybench.trackers.game_tracker import GameStep
from plybench.trackers.result_tracker import ResultTracker
from plybench.utils.text import compress_ranges

# a matchup only reprints its status this often while moves stream in, so fast (bot) matchups
# and many concurrent matchups do not flood the console; round boundaries always print
MOVE_LOG_INTERVAL = 2.0

MatchupKey = tuple[str, str, str]
PhaseKey = tuple[TrainingRunKey, int, str]


@dataclass
class _MatchupProgress:
    preexisting: set[int]
    n: int
    played: set[int] = field(default_factory=set)
    active: set[int] = field(default_factory=set)
    moves: int = 0
    last_log: float = 0.0

    @property
    def done(self) -> set[int]:
        return self.preexisting | self.played

    @property
    def queued(self) -> set[int]:
        return set(range(1, self.n + 1)) - self.done - self.active

    def status(self) -> str:
        parts = [f"{len(self.done)}/{self.n} done"]
        if self.active:
            parts.append(f"playing {compress_ranges(self.active)}")
        if self.queued:
            parts.append(f"queued {compress_ranges(self.queued)}")
        parts.append(f"{self.moves} moves")
        return " · ".join(parts)


def console_benchmark_callbacks() -> BenchmarkCallbacks:
    # per-matchup state so every line shows a running done/total, which rounds are in flight vs waiting,
    # and how many moves this run has played; already-completed rounds are skipped on resume
    matchups: dict[MatchupKey, _MatchupProgress] = {}

    def key(game_config: GameConfig, i: PlayerConfig, o: PlayerConfig) -> MatchupKey:
        return (game_config.path, i.path, o.path)

    def label(game_config: GameConfig, i: PlayerConfig, o: PlayerConfig) -> str:
        return f"{game_config.path}: {i.path} vs {o.path}"

    def log(game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, progress: _MatchupProgress) -> None:
        progress.last_log = time.monotonic()
        print(f"[benchmark] {label(game_config, i, o)} — {progress.status()}")

    def on_matchup_start(result_tracker: ResultTracker, game_config: GameConfig, i: PlayerConfig, o: PlayerConfig) -> None:
        preexisting = set(result_tracker.get_completed_games())
        matchups[key(game_config, i, o)] = _MatchupProgress(preexisting, result_tracker.n)
        print(f"[benchmark] start {label(game_config, i, o)} ({len(preexisting)}/{result_tracker.n})")

    def on_round_start(game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, game_round: int) -> None:
        progress = matchups[key(game_config, i, o)]
        progress.active.add(game_round)
        log(game_config, i, o, progress)

    def on_round_complete(game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, game_round: int) -> None:
        progress = matchups[key(game_config, i, o)]
        if game_round in progress.preexisting:  # already done before this run — don't log skipped rounds
            return
        progress.active.discard(game_round)
        progress.played.add(game_round)
        log(game_config, i, o, progress)

    def on_move_complete(game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, game_round: int, _step: GameStep) -> None:
        progress = matchups[key(game_config, i, o)]
        progress.moves += 1
        if time.monotonic() - progress.last_log >= MOVE_LOG_INTERVAL:
            log(game_config, i, o, progress)

    def on_matchup_end(result_tracker: ResultTracker) -> None:
        progress = matchups.get((result_tracker.game.path, result_tracker.i.path, result_tracker.o.path))
        moves = f" ({progress.moves} moves played)" if progress is not None and progress.moves else ""
        print(f"[benchmark] done {result_tracker.game.path}: {result_tracker.i.path} vs {result_tracker.o.path}{moves}")

    return BenchmarkCallbacks(
        matchup_start_callback=on_matchup_start,
        round_start_callback=on_round_start,
        round_complete_callback=on_round_complete,
        move_complete_callback=on_move_complete,
        matchup_end_callback=on_matchup_end,
    )


def _outcomes(result_tracker: ResultTracker) -> str:
    counts: dict[str, int] = {}
    for game in result_tracker.games:
        if game is None or game.ending is None:
            continue
        name = game.get_result(result_tracker.i).name.lower()
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def console_training_callbacks() -> TrainingCallbacks:
    # the same running done/total the benchmark prints, keyed by phase instead of matchup: a phase is
    # one epoch's train or test-vs-one-tester, and its label carries the coordinates a matchup has no room for
    phases: dict[PhaseKey, _MatchupProgress] = {}

    def key(phase: EpochPhase) -> PhaseKey:
        return (phase.run.key, phase.epoch.index, phase.dir_name)

    def label(phase: EpochPhase) -> str:
        run = phase.run
        return f"{run.game.path}: {run.trainee.path} rep{run.replicate} · epoch {phase.epoch.index}/{run.training_config.num_epochs} · {phase.dir_name}"

    def log(phase: EpochPhase, progress: _MatchupProgress) -> None:
        progress.last_log = time.monotonic()
        print(f"[training] {label(phase)} — {progress.status()}")

    def on_training_start(game_configs: list[str], player_configs: list[str], trainer_configs: list[str], tester_configs: list[str]) -> None:
        matrix = f"{len(game_configs)} game(s) × {len(player_configs)} trainee(s) × {len(trainer_configs)} trainer(s)"
        print(f"[training] start {matrix}, evaluated against {len(tester_configs)} tester(s)")

    def on_run_start(run: TrainingRun) -> None:
        print(f"[training] run {run.game.path}: {run.trainee.path} vs {run.trainer.path} (replicate {run.replicate}) — {run.training_config}")

    def on_phase_start(result_tracker: ResultTracker, phase: EpochPhase) -> None:
        preexisting = set(result_tracker.get_completed_games())
        phases[key(phase)] = _MatchupProgress(preexisting, result_tracker.n)
        print(f"[training] {label(phase)} — {len(preexisting)}/{result_tracker.n} already done")

    def on_round_start(phase: EpochPhase, game_round: int) -> None:
        progress = phases[key(phase)]
        progress.active.add(game_round)
        log(phase, progress)

    def on_round_complete(phase: EpochPhase, game_round: int) -> None:
        progress = phases[key(phase)]
        if game_round in progress.preexisting:  # already done before this run — don't log skipped rounds
            return
        progress.active.discard(game_round)
        progress.played.add(game_round)
        log(phase, progress)

    def on_move_complete(phase: EpochPhase, game_round: int, _step: GameStep) -> None:
        progress = phases[key(phase)]
        progress.moves += 1
        if time.monotonic() - progress.last_log >= MOVE_LOG_INTERVAL:
            log(phase, progress)

    def on_phase_end(result_tracker: ResultTracker, phase: EpochPhase) -> None:
        print(f"[training] {label(phase)} — done ({_outcomes(result_tracker)})")

    def on_checkpoint(epoch: Epoch, checkpoint: Path) -> None:
        print(f"[training] epoch {epoch.index} checkpoint -> {checkpoint}")

    def on_run_end(run: TrainingRun) -> None:
        print(f"[training] run done {run.game.path}: {run.trainee.path} vs {run.trainer.path} (replicate {run.replicate})")

    def on_training_end(results: TrainingResults) -> None:
        print(f"[training] done: {len(results.epochs)} epoch(s) across {len(results.runs)} run(s)")

    return TrainingCallbacks(
        training_start_callback=on_training_start,
        training_end_callback=on_training_end,
        run_start_callback=on_run_start,
        run_end_callback=on_run_end,
        checkpoint_callback=on_checkpoint,
        phase_start_callback=on_phase_start,
        phase_end_callback=on_phase_end,
        round_start_callback=on_round_start,
        round_complete_callback=on_round_complete,
        move_complete_callback=on_move_complete,
    )
