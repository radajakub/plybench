from __future__ import annotations

from dataclasses import dataclass

from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.configs.training_run import Epoch, TrainingRun, TrainingRunKey
from plybench.trackers.result_tracker import ResultTracker


@dataclass(frozen=True)
class EpochResults:
    epoch: Epoch
    train: ResultTracker | None
    tests: list[ResultTracker]

    @property
    def run(self) -> TrainingRun:
        return self.epoch.run

    def test(self, tester: PlayerConfig) -> ResultTracker:
        for tracker in self.tests:
            if tracker.o.hash == tester.hash:
                return tracker
        raise ValueError(f"No test result against {tester.to_string()} at epoch {self.epoch.index}")


class TrainingResults:
    def __init__(self, runs: list[TrainingRun], epochs: list[EpochResults]) -> None:
        self.runs = runs
        self.epochs = epochs
        self._index: dict[tuple[TrainingRunKey, int], EpochResults] = {(results.run.key, results.epoch.index): results for results in epochs}

    def find(self, run: TrainingRun, epoch: int) -> EpochResults:
        results = self._index.get((run.key, epoch))
        if results is None:
            raise ValueError(f"No results for epoch {epoch} of {run.to_dict()}")
        return results

    def for_run(self, run: TrainingRun) -> list[EpochResults]:
        return sorted((results for results in self.epochs if results.run.key == run.key), key=lambda results: results.epoch.index)

    def curve(self, run: TrainingRun, tester: PlayerConfig) -> list[ResultTracker]:
        # the learning curve: one frozen evaluation per epoch, epoch 0 being the untrained baseline
        return [results.test(tester) for results in self.for_run(run)]

    def for_game(self, game_config: GameConfig) -> list[EpochResults]:
        return [results for results in self.epochs if results.run.game.to_string() == game_config.to_string()]

    def for_trainee(self, trainee_config: PlayerConfig) -> list[EpochResults]:
        return [results for results in self.epochs if results.run.trainee.hash == trainee_config.hash]

    def for_trainer(self, trainer_config: PlayerConfig) -> list[EpochResults]:
        return [results for results in self.epochs if results.run.trainer.hash == trainer_config.hash]
