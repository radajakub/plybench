from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from plybench.common.enums import Mode
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import CheckpointedParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.llm_call import LLMCall

# the LLM calls of one proposal, next to its update in the update cache; a name save_update must not use
LEARNER_CALLS = "llm_calls.json"
# the learner's state after the update of a game (or of the epoch) was committed, as save_checkpoint writes it, next to the update
LEARNER_STATE = "state"


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
        # the calls of the proposal under way, reported with record_call; None outside propose / propose_epoch
        self._calls: list[LLMCall] | None = None

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

    async def observe_batch(self, games: list[GameTracker], index: int) -> None:
        # the index-th batch of this epoch's training games (TrainingConfig.update_every), after each of its games was observed
        self._require_training("observe_batch")
        name = f"batch_{index}"
        update = await self._cached(name, lambda: self.propose_batch(games))
        if update is not None:
            self.commit_batch(update)
        self._keep_state(name)

    def _require_training(self, action: str) -> None:
        if not self.is_training:
            raise RuntimeError(f"{self} is frozen: {action}() is only legal inside a training phase")

    async def _observe(self, game: GameTracker) -> None:
        name = f"game_{game.game_round}"
        update = await self._cached(name, lambda: self.propose(game))
        if update is not None:
            self.commit(update)
        self._keep_state(name)

    async def _update(self, games: list[GameTracker]) -> None:
        update = await self._cached("epoch", lambda: self.propose_epoch(games))
        if update is not None:
            self.commit_epoch(update)
        self._keep_state("epoch")

    def _keep_state(self, name: str) -> None:
        # the state after the commit, kept beside the update so the record shows what each game started from. A replay commits again and must
        # land on the state it kept (the resume rule): a difference is an error, never overwritten. A state lost between commit and write is written
        assert self.workdir is not None, "_cached checks the workdir before"
        target = self.workdir / name / LEARNER_STATE
        staged = self.workdir / f"{name}.{LEARNER_STATE}.tmp"
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(parents=True)
        self.save_checkpoint(staged)
        if not target.exists():
            os.replace(staged, target)
            return
        kept, now = _files(target), _files(staged)
        shutil.rmtree(staged)
        if kept != now:
            changed = sorted(path for path in kept.keys() | now.keys() if kept.get(path) != now.get(path))
            raise ValueError(
                f"{self}: the state replayed after {name} differs from the one kept in {target} ({', '.join(changed)}); commit must depend only on the state and the update"
            )

    def finish(self, game: GameTracker) -> dict[str, Any]:
        # what the game record keeps of the learner beyond its moves (GameEnding.players): the state it played from, the checkpoint it was built
        # with and, in training, the state the previous game of the epoch left (the checkpoint for the first). Paths as the checkpoint is given
        checkpoint = self.checkpoint.as_posix() if self.checkpoint is not None else None
        before = checkpoint
        if self.is_training and self.workdir is not None and game.game_round > 1:
            before = (self.workdir / f"game_{game.game_round - 1}" / LEARNER_STATE).as_posix()
        kept: dict[str, Any] = {"state": {"checkpoint": checkpoint, "before": before}}
        # in training, also where this game's update, calls and state go (the learner folder of the game, written once the game is learned from)
        if self.is_training and self.workdir is not None:
            kept["learned"] = (self.workdir / f"game_{game.game_round}").as_posix()
        return kept

    async def _cached(self, name: str, propose: Callable[[], Awaitable[U | None]]) -> U | None:
        # an update is proposed once per epoch and name; a replay loads it instead of proposing again
        if self.workdir is None:
            raise RuntimeError(f"{self} has no workdir: begin_epoch() must be called before it learns")
        directory = self.workdir / name
        if directory.exists():
            return None if (directory / LearnablePlayer._NO_UPDATE).exists() else self.load_update(directory)

        self._calls = []
        try:
            update = await propose()
            calls = self._calls
        finally:
            self._calls = None
        # the directory's existence marks the update done, so stage the write and rename it into place,
        # as the harness does for checkpoints: an interruption mid-write leaves no half update behind
        staged = directory.with_name(f"{name}.tmp")
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(parents=True)
        # kept for a proposal that taught nothing too: its calls were made and paid for
        (staged / LEARNER_CALLS).write_text(json.dumps([call.to_dict() for call in calls]))
        if update is None:
            (staged / LearnablePlayer._NO_UPDATE).touch()
        else:
            self.save_update(update, staged)
            # a resumed epoch commits what load_update reads, so it must be what was proposed; checked before it is cached
            if self.load_update(staged) != update:
                raise ValueError(f"{self}: load_update does not give back the update save_update stored; store all of it, as a type that compares by content (e.g. a dataclass)")
        os.replace(staged, directory)
        return update

    def record_call(self, call: LLMCall) -> None:
        # every LLM call of propose / propose_epoch is reported here and stored with its update, so the cost of learning shows in
        # the results; a call made in a move belongs in PlayerOutput (add_call) instead
        if self._calls is None:
            raise RuntimeError(f"{self}: record_call() is only legal inside propose() or propose_epoch(); a move records its calls in PlayerOutput")
        self._calls.append(call)

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

    # a batch of training games (TrainingConfig.update_every): proposed once, cached as batch_<index> and replayed on a resume as a game's update is
    async def propose_batch(self, games: list[GameTracker]) -> U | None:
        return None

    def commit_batch(self, update: U) -> None:
        pass

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


def _files(directory: Path) -> dict[str, bytes]:
    # every file under a directory by its relative path, with its content
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}
