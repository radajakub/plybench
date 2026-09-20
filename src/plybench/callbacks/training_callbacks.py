from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

from plybench.callbacks.matchup_callbacks import MatchupCallbacks
from plybench.configs.training_run import Epoch, EpochPhase, TrainingRun
from plybench.harness.training.results import TrainingResults
from plybench.trackers.game_tracker import GameStep
from plybench.trackers.result_tracker import ResultTracker

# The training counterpart of `BenchmarkCallbacks`: the same matchup/round/move events, re-cut so every
# hook carries the run, epoch and phase it belongs to -- coordinates the benchmark bundle has no room
# for. `for_phase` binds this bundle to one phase, yielding the `MatchupCallbacks` the matchup layer
# takes, exactly as `MatchupCallbacks.for_round` binds itself to one round for the game loop.

# full training pipeline
TrainingStartCallback = Callable[[list[str], list[str], list[str], list[str]], None]
TrainingEndCallback = Callable[["TrainingResults"], None]

# a single run, i.e. (player, trainer, game) combination
RunStartCallback = Callable[["TrainingRun"], None]
RunEndCallback = Callable[["TrainingRun"], None]

# a single epoch
EpochStartCallback = Callable[["Epoch"], None]
EpochEndCallback = Callable[["Epoch"], None]
CheckpointCallback = Callable[["Epoch", Path], None]

# when a phase starts or ends
PhaseStartCallback = Callable[["ResultTracker", "EpochPhase"], None]
PhaseEndCallback = Callable[["ResultTracker", "EpochPhase"], None]

# when a round/matchup starts or ends
RoundStartCallback = Callable[["EpochPhase", int], None]
RoundCompleteCallback = Callable[["EpochPhase", int], None]

# callback on move
MoveCompleteCallback = Callable[["EpochPhase", int, "GameStep"], None]


@dataclass
class TrainingCallbacks:
    training_start_callback: TrainingStartCallback | None = None
    training_end_callback: TrainingEndCallback | None = None
    run_start_callback: RunStartCallback | None = None
    run_end_callback: RunEndCallback | None = None
    epoch_start_callback: EpochStartCallback | None = None
    epoch_end_callback: EpochEndCallback | None = None
    checkpoint_callback: CheckpointCallback | None = None
    phase_start_callback: PhaseStartCallback | None = None
    phase_end_callback: PhaseEndCallback | None = None
    round_start_callback: RoundStartCallback | None = None
    round_complete_callback: RoundCompleteCallback | None = None
    move_complete_callback: MoveCompleteCallback | None = None

    def on_training_start(self, game_configs: list[str], player_configs: list[str], trainer_configs: list[str], tester_configs: list[str]) -> None:
        if self.training_start_callback is not None:
            self.training_start_callback(game_configs, player_configs, trainer_configs, tester_configs)

    def on_training_end(self, results: TrainingResults) -> None:
        if self.training_end_callback is not None:
            self.training_end_callback(results)

    def on_run_start(self, run: TrainingRun) -> None:
        if self.run_start_callback is not None:
            self.run_start_callback(run)

    def on_run_end(self, run: TrainingRun) -> None:
        if self.run_end_callback is not None:
            self.run_end_callback(run)

    def on_epoch_start(self, epoch: Epoch) -> None:
        if self.epoch_start_callback is not None:
            self.epoch_start_callback(epoch)

    def on_epoch_end(self, epoch: Epoch) -> None:
        if self.epoch_end_callback is not None:
            self.epoch_end_callback(epoch)

    def on_checkpoint(self, epoch: Epoch, checkpoint: Path) -> None:
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(epoch, checkpoint)

    def on_phase_start(self, result_tracker: ResultTracker, phase: EpochPhase) -> None:
        if self.phase_start_callback is not None:
            self.phase_start_callback(result_tracker, phase)

    def on_phase_end(self, result_tracker: ResultTracker, phase: EpochPhase) -> None:
        if self.phase_end_callback is not None:
            self.phase_end_callback(result_tracker, phase)

    def on_round_start(self, phase: EpochPhase, game_round: int) -> None:
        if self.round_start_callback is not None:
            self.round_start_callback(phase, game_round)

    def on_round_complete(self, phase: EpochPhase, game_round: int) -> None:
        if self.round_complete_callback is not None:
            self.round_complete_callback(phase, game_round)

    def on_move_complete(self, phase: EpochPhase, game_round: int, step: GameStep) -> None:
        if self.move_complete_callback is not None:
            self.move_complete_callback(phase, game_round, step)

    def for_phase(self, phase: EpochPhase) -> MatchupCallbacks:
        # bind this bundle to one phase: the matchup layer's game and player arguments are dropped, because the phase already names them
        return MatchupCallbacks(
            matchup_start_callback=lambda tracker, game, i, o: self.on_phase_start(tracker, phase),
            matchup_end_callback=lambda tracker: self.on_phase_end(tracker, phase),
            round_start_callback=lambda game, i, o, game_round: self.on_round_start(phase, game_round),
            round_complete_callback=lambda game, i, o, game_round: self.on_round_complete(phase, game_round),
            move_complete_callback=lambda game, i, o, game_round, step: self.on_move_complete(phase, game_round, step),
        )

    @classmethod
    def combine(cls, *bundles: TrainingCallbacks | None) -> TrainingCallbacks:
        children = tuple(bundle for bundle in bundles if bundle is not None)

        def fan(method_name: str) -> Callable[..., None]:
            def call(*args: object) -> None:
                for child in children:
                    getattr(child, method_name)(*args)

            return call

        return cls(**{field.name: fan(f"on_{field.name.removesuffix('_callback')}") for field in fields(cls)})
