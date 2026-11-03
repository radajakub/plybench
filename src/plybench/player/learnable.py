from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path

from plybench.common.enums import Mode
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import CheckpointedParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.trackers.game_tracker import GameTracker


class LearnablePlayer[U](Player, ABC):
    # marks a cached proposal that was None: the game taught nothing, so there is nothing to commit
    _NO_UPDATE = "no_update"

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

    async def _observe(self, game: GameTracker) -> None:
        update = await self._cached(f"game_{game.game_round}", lambda: self.propose(game))
        if update is not None:
            self.commit(update)

    async def _update(self, games: list[GameTracker]) -> None:
        update = await self._cached("epoch", lambda: self.propose_epoch(games))
        if update is not None:
            self.commit_epoch(update)

    async def _cached(self, name: str, propose: Callable[[], Awaitable[U | None]]) -> U | None:
        # an update is proposed once per epoch and name; a replay loads it instead of proposing again
        if self.workdir is None:
            raise RuntimeError(f"{self} has no workdir: begin_epoch() must be called before it learns")
        directory = self.workdir / name
        if directory.exists():
            return None if (directory / LearnablePlayer._NO_UPDATE).exists() else self.load_update(directory)

        update = await propose()
        # the directory's existence marks the update done, so stage the write and rename it into place,
        # as the harness does for checkpoints: an interruption mid-write leaves no half update behind
        staged = directory.with_name(f"{name}.tmp")
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(parents=True)
        if update is None:
            (staged / LearnablePlayer._NO_UPDATE).touch()
        else:
            self.save_update(update, staged)
            # a resumed epoch commits what load_update reads, so it must be what was proposed; checked before it is cached
            if self.load_update(staged) != update:
                raise ValueError(f"{self}: load_update does not give back the update save_update stored; store all of it, as a type that compares by content (e.g. a dataclass)")
        os.replace(staged, directory)
        return update

    @abstractmethod
    def _initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        raise NotImplementedError

    # the move, written by the learner: read the learned state, never change it; what propose needs later goes into PlayerOutput.data, which the engine records
    @abstractmethod
    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        raise NotImplementedError

    # resume rule: a resumed epoch loads the cached updates and commits them again, so it lands on the same state only if everything sampled happens in propose,
    # commit depends only on the current state and the update, and nothing reads outside the records (wall-clock, process state, callbacks).
    # Randomness in propose (seeds of its LLM calls, sampling) derives from game.seed (derive_seed(game.seed, role, index)), not from self.rng,
    # which is seeded per game only when a game is played, not when a resumed epoch replays it
    @abstractmethod
    async def propose(self, game: GameTracker) -> U | None:
        raise NotImplementedError

    @abstractmethod
    def commit(self, update: U) -> None:
        raise NotImplementedError

    async def propose_epoch(self, games: list[GameTracker]) -> U | None:
        return None

    def commit_epoch(self, update: U) -> None:
        pass

    @abstractmethod
    def save_update(self, update: U, directory: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_update(self, directory: Path) -> U:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint: Path) -> None:
        raise NotImplementedError
