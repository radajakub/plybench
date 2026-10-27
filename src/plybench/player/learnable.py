from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from plybench.common.enums import Mode
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import CheckpointedParams
from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier
from plybench.trackers.game_tracker import GameTracker


class LearnablePlayer(Player, ABC):
    def __init__(self, player_config: PlayerConfig, identifier: PlayerIdentifier) -> None:
        super().__init__(player_config, identifier)
        params = player_config.params
        if not isinstance(params, CheckpointedParams):
            raise TypeError(f"{player_config.to_string()} is learnable, so its params must be CheckpointedParams, got {type(params).__name__}")
        self.params = params
        self._mode = Mode.EVAL
        self.workdir: Path | None = None
        self._restored = False

    @property
    def checkpoint(self) -> Path | None:
        return self.params.checkpoint

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def is_training(self) -> bool:
        return self._mode is Mode.TRAIN

    @property
    def is_eval(self) -> bool:
        return self._mode is Mode.EVAL

    def set_mode(self, mode: Mode) -> None:
        self._mode = mode

    def begin_epoch(self, workdir: Path) -> None:
        # durable per-epoch scratch, handed over before the training phase starts
        workdir.mkdir(parents=True, exist_ok=True)
        self.workdir = workdir

    def restore(self) -> None:
        if self._restored or self.checkpoint is None:
            return
        self.load_checkpoint(self.checkpoint)
        self._restored = True

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        self._initialize_policy(game, prompt_adapter_template)
        # players the harness never touches -- an evaluation phase builds its own through the registry
        # -- restore on their first game; a learner the harness builds is restored before it plays
        self.restore()

    async def observe(self, game: GameTracker) -> None:
        self._require_training("observe")
        await self._observe(game)

    async def update(self, games: list[GameTracker]) -> None:
        self._require_training("update")
        await self._update(games)

    def _require_training(self, action: str) -> None:
        if not self.is_training:
            raise RuntimeError(f"{self} is frozen: {action}() is only legal inside a training phase")

    @abstractmethod
    def _initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _observe(self, game: GameTracker) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _update(self, games: list[GameTracker]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint: Path) -> None:
        raise NotImplementedError
