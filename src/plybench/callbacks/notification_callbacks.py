from __future__ import annotations

import time
from dataclasses import dataclass, field

import clankers
from clankers.core.context import MessageBuilder
from clankers.core.models import format_duration

from plybench.callbacks.benchmark_callbacks import BenchmarkCallbacks
from plybench.callbacks.console_callbacks import PhaseKey
from plybench.callbacks.training_callbacks import TrainingCallbacks
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.configs.training_run import Epoch, EpochPhase, TrainingRun
from plybench.trackers.result_tracker import ResultTracker


@dataclass
class _NotificationProgress:
    total_matchups: int = 0
    done_matchups: int = 0
    # rounds actually played this run (preexisting/resumed rounds excluded) — the throughput signal for the ETA
    fresh_rounds_total: int = 0
    fresh_rounds_done: int = 0
    start: float = 0.0
    preexisting: dict[tuple[str, str, str], set[int]] = field(default_factory=dict)

    def eta(self, elapsed: float) -> float | None:
        rounds_left = self.fresh_rounds_total - self.fresh_rounds_done
        if rounds_left <= 0 or self.fresh_rounds_done <= 0 or elapsed <= 0:
            return None
        return rounds_left / (self.fresh_rounds_done / elapsed)

    def describe(self) -> str:
        parts = [f"{self.done_matchups}/{self.total_matchups} matchups"]
        if self.fresh_rounds_total:
            parts.append(f"{self.fresh_rounds_done}/{self.fresh_rounds_total} rounds")
        return ", ".join(parts)


def notification_benchmark_callbacks(experiment: str) -> tuple[BenchmarkCallbacks, MessageBuilder]:
    # push a neutral notification as each matchup finishes; the returned builder renders the progress so far, so the
    # `clankers.Engage` wrapper in run.py can report it on success *and* on a crash. matchups all share the per-provider
    # LLM-call semaphore and finish clustered near the end, so the ETA is derived from *round* throughput
    # (rounds drain through that pipe steadily) rather than from matchup completions.
    state = _NotificationProgress()

    def key(game_config: GameConfig, i: PlayerConfig, o: PlayerConfig) -> tuple[str, str, str]:
        return (game_config.path, i.path, o.path)

    def on_benchmark_start(game_configs: list[str], player_configs: list[str], opponent_configs: list[str]) -> None:
        state.total_matchups = len(game_configs) * len(player_configs) * len(opponent_configs)
        state.done_matchups = 0
        state.fresh_rounds_total = 0
        state.fresh_rounds_done = 0
        state.start = time.monotonic()
        state.preexisting.clear()

    def on_matchup_start(result_tracker: ResultTracker, game_config: GameConfig, i: PlayerConfig, o: PlayerConfig) -> None:
        preexisting = set(result_tracker.get_completed_games())
        state.preexisting[key(game_config, i, o)] = preexisting
        state.fresh_rounds_total += result_tracker.n - len(preexisting)

    def on_round_complete(game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, game_round: int) -> None:
        if game_round in state.preexisting.get(key(game_config, i, o), set()):
            return  # resumed round — completes instantly, must not inflate the throughput estimate
        state.fresh_rounds_done += 1

    def on_matchup_end(result_tracker: ResultTracker) -> None:
        state.done_matchups += 1
        elapsed = time.monotonic() - state.start
        label = f"{result_tracker.game.path}: {result_tracker.i.path} vs {result_tracker.o.path}"
        parts = [f"[{experiment}] matchup done ({state.describe()}): {label}"]
        eta = state.eta(elapsed)
        if eta is not None:
            parts.append(f"est. {format_duration(eta)} left")
        # clankers renders the elapsed time itself from `duration`
        clankers.blastthem(" | ".join(parts), elapsed)

    callbacks = BenchmarkCallbacks(
        benchmark_start_callback=on_benchmark_start,
        matchup_start_callback=on_matchup_start,
        round_complete_callback=on_round_complete,
        matchup_end_callback=on_matchup_end,
    )
    return callbacks, state.describe


@dataclass
class _TrainingProgress:
    num_testers: int = 0
    epochs_done: int = 0
    # every round the matrix will ever play, known exactly per run the moment the run starts
    total_rounds: int = 0
    # rounds already on disk, so the ETA paces on what this process actually plays
    resumed_rounds: int = 0
    fresh_rounds_done: int = 0
    start: float = 0.0
    preexisting: dict[PhaseKey, set[int]] = field(default_factory=dict)

    @property
    def playable(self) -> int:
        return self.total_rounds - self.resumed_rounds

    def eta(self, elapsed: float) -> float | None:
        rounds_left = self.playable - self.fresh_rounds_done
        if rounds_left <= 0 or self.fresh_rounds_done <= 0 or elapsed <= 0:
            return None
        return rounds_left / (self.fresh_rounds_done / elapsed)

    def describe(self) -> str:
        return f"{self.epochs_done} epoch(s), {self.fresh_rounds_done}/{self.playable} rounds"


def notification_training_callbacks(experiment: str) -> tuple[TrainingCallbacks, MessageBuilder]:
    # push a notification as each epoch finishes -- the unit a training run makes progress in, and the
    # point at which a checkpoint exists to resume from. Unlike a benchmark the denominator is exact:
    # a run's schedule says how many rounds it will ever play, so the ETA is honest from the first epoch
    state = _TrainingProgress()

    def key(phase: EpochPhase) -> PhaseKey:
        return (phase.run.key, phase.epoch.index, phase.dir_name)

    def on_training_start(game_configs: list[str], player_configs: list[str], trainer_configs: list[str], tester_configs: list[str]) -> None:
        del game_configs, player_configs, trainer_configs
        state.num_testers = len(tester_configs)
        state.epochs_done = 0
        state.total_rounds = 0
        state.resumed_rounds = 0
        state.fresh_rounds_done = 0
        state.start = time.monotonic()
        state.preexisting.clear()

    def on_run_start(run: TrainingRun) -> None:
        schedule = run.training_config
        # epoch 0 is evaluated but never trained, so there is one more evaluation round than training one
        training = schedule.num_epochs * schedule.num_training_games
        testing = (schedule.num_epochs + 1) * state.num_testers * schedule.num_test_games
        state.total_rounds += training + testing

    def on_phase_start(result_tracker: ResultTracker, phase: EpochPhase) -> None:
        preexisting = set(result_tracker.get_completed_games())
        state.preexisting[key(phase)] = preexisting
        state.resumed_rounds += len(preexisting)

    def on_round_complete(phase: EpochPhase, game_round: int) -> None:
        if game_round in state.preexisting.get(key(phase), set()):
            return  # resumed round — completes instantly, must not inflate the throughput estimate
        state.fresh_rounds_done += 1

    def on_epoch_end(epoch: Epoch) -> None:
        state.epochs_done += 1
        elapsed = time.monotonic() - state.start
        run = epoch.run
        label = f"{run.game.path}: {run.trainee.path} vs {run.trainer.path} rep{run.replicate} epoch {epoch.index}/{run.training_config.num_epochs}"
        parts = [f"[{experiment}] epoch done ({state.describe()}): {label}"]
        eta = state.eta(elapsed)
        if eta is not None:
            parts.append(f"est. {format_duration(eta)} left")
        # clankers renders the elapsed time itself from `duration`
        clankers.blastthem(" | ".join(parts), elapsed)

    callbacks = TrainingCallbacks(
        training_start_callback=on_training_start,
        run_start_callback=on_run_start,
        phase_start_callback=on_phase_start,
        round_complete_callback=on_round_complete,
        epoch_end_callback=on_epoch_end,
    )
    return callbacks, state.describe
