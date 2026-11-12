from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.configs.training_config import TrainingConfig

type TrainingRunKey = tuple[str, str, str, str, int]


@dataclass(frozen=True, eq=True)
class TrainingRun:
    game: GameConfig
    trainee: PlayerConfig
    trainer: PlayerConfig
    training_config: TrainingConfig
    replicate: int

    @property
    def key(self) -> TrainingRunKey:
        return (self.game.to_string(), self.trainee.hash, self.trainer.hash, self.training_config.to_string(), self.replicate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_config": self.game.to_string(),
            "trainee_config": self.trainee.to_string(),
            "trainer_config": self.trainer.to_string(),
            "training_config": self.training_config.to_string(),
            "replicate": self.replicate,
        }


@dataclass(frozen=True, eq=True)
class Epoch:
    run: TrainingRun
    index: int

    @property
    def game(self) -> GameConfig:
        # what this epoch trains and tests on: the run's game, or what it changes into at this epoch
        return self.run.game.at_epoch(self.index)

    @property
    def train(self) -> TrainPhase:
        return TrainPhase(self)

    def test(self, tester: PlayerConfig) -> TestPhase:
        return TestPhase(self, tester)


@dataclass(frozen=True, eq=True)
class EpochPhase(ABC):
    epoch: Epoch

    @property
    def run(self) -> TrainingRun:
        return self.epoch.run

    @property
    @abstractmethod
    def dir_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def opponent(self) -> PlayerConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def player_epoch(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_games(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def kind(self) -> str:
        raise NotImplementedError

    def place(self, experiment: str) -> dict[str, Any]:
        # where this phase's matchup belongs (experiment, run, epoch of how many, train or test), kept in its metadata.json so a tool reads it
        # without knowing the folder layout; the tester of a test is the matchup's opponent
        return {"experiment": experiment, "run": self.run.to_dict(), "epoch": self.epoch.index, "epochs": self.run.training_config.num_epochs, "phase": self.kind}


@dataclass(frozen=True, eq=True)
class TrainPhase(EpochPhase):
    @property
    def dir_name(self) -> str:
        return "train"

    @property
    def opponent(self) -> PlayerConfig:
        return self.run.trainer

    @property
    def player_epoch(self) -> int:
        # training plays the checkpoint the epoch starts from and leaves behind the one it produced
        return self.epoch.index - 1

    @property
    def num_games(self) -> int:
        return self.run.training_config.num_training_games

    @property
    def kind(self) -> str:
        return "train"


@dataclass(frozen=True, eq=True)
class TestPhase(EpochPhase):
    tester: PlayerConfig

    @property
    def dir_name(self) -> str:
        return f"test_vs_{self.tester.path}"

    @property
    def opponent(self) -> PlayerConfig:
        return self.tester

    @property
    def player_epoch(self) -> int:
        # testing plays what the epoch produced, frozen
        return self.epoch.index

    @property
    def num_games(self) -> int:
        return self.run.training_config.num_test_games

    @property
    def kind(self) -> str:
        return "test"
