from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import CheckpointedParams
from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier
from plybench.trackers.game_tracker import GameTracker


class LearnablePlayer(Player, ABC):
    """A player the training harness improves, with a PyTorch-style mode switch.

    In TRAIN mode everything is open -- memory, prompts, weights -- and the harness hands the learner
    each trajectory the moment its game ends, then the whole epoch's trajectories once the phase is
    over, so a learner picks its own granularity. Training games are therefore played one at a time,
    by this one instance, in round order. In EVAL mode nothing may change: `observe`/`update` refuse
    to run, the checkpoint is fixed, and the harness plays the evaluation games in parallel.

    A player is built in EVAL mode -- only a training phase turns learning on -- so every player the
    benchmark or an evaluation phase builds is frozen by construction.

    CHECKPOINTS travel in the params (hence `CheckpointedParams`), so a trained artifact is an
    ordinary player config that the benchmark can play afterwards and that every result records.
    `save_checkpoint` writes a directory and `load_checkpoint` reads one back; the config says which
    directory and this class decides when. `restore` is idempotent and applied once per instance, not
    once per game: a training phase plays all of its games on one player, which must keep what the
    epoch has learned so far rather than reverting to the checkpoint it started from.

    INVARIANT -- `_observe` and `_update` must be functions of (current state, the records they are
    handed, in order) and of nothing else. The harness resumes an interrupted epoch by replaying its
    already-recorded games through `_observe` before playing what is missing, so a resumed epoch must
    land on the checkpoint an uninterrupted one would. Two things follow: never mutate learned state
    from `__call__` (eval-mode play would drift, and the drift would not survive a replay), and never
    reach for anything outside the records -- wall-clock, process state, a `GameCallbacks` hook.

    A derivation that is itself sampled -- an LLM judging or summarising the trajectory, say -- breaks
    that equality on its own, because replaying a record re-rolls the sample. Such a learner must cache
    what it derived per game under `workdir`, keyed by the record's `game_round`, and reuse the cached
    value on a replay. `workdir` is this epoch's own directory and survives the interruption, which the
    epoch's checkpoint by definition does not; caching there also spares an interrupted epoch from
    paying for the derivation twice. Both hooks are async, so the derivation may call out to a model.

    `_initialize_policy` is called once per game, on this same instance; it must reset per-game
    scratch without touching what the epoch has learned.
    """

    def __init__(self, player_config: PlayerConfig, identifier: PlayerIdentifier) -> None:
        super().__init__(player_config, identifier)
        params = player_config.params
        if not isinstance(params, CheckpointedParams):
            raise TypeError(f"{player_config.to_string()} is learnable, so its params must be CheckpointedParams, got {type(params).__name__}")
        self.params = params
        self.training = False
        self.workdir: Path | None = None
        self._restored = False

    @property
    def checkpoint(self) -> Path | None:
        return self.params.checkpoint

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

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
        if not self.training:
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
