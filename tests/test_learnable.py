"""The LearnablePlayer contract: a learner is built frozen, and only a training phase lets it learn."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from plybench.common.enums import Mode
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import CheckpointedParams
from plybench.player.learnable import LearnablePlayer
from plybench.player.player import PlayerOutput


@dataclass(frozen=True, eq=True)
class _CounterParams(CheckpointedParams):
    path: Path | None = None

    @classmethod
    def from_string(cls, params_string: str) -> _CounterParams:
        return cls(Path(params_string) if params_string else None)

    def to_string(self) -> str:
        return str(self.path) if self.path else ""

    @property
    def path_suffix(self) -> str:
        return ""

    @property
    def checkpoint(self) -> Path | None:
        return self.path

    def with_checkpoint(self, checkpoint: Path) -> _CounterParams:
        return _CounterParams(checkpoint)


class _Counter(LearnablePlayer):
    # plays the first legal move and counts the games it learned from
    def __init__(self) -> None:
        super().__init__(PlayerConfig("counter", _CounterParams()), "i")
        self.games = 0

    def _initialize_policy(self, game, prompt_adapter_template) -> None:
        pass

    async def __call__(self, game, observation, legal_moves) -> PlayerOutput:
        return PlayerOutput(action=legal_moves[0])

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""

    async def _observe(self, game) -> None:
        self.games += 1

    async def _update(self, games) -> None:
        pass

    def load_checkpoint(self, checkpoint: Path) -> None:
        pass

    def save_checkpoint(self, checkpoint: Path) -> None:
        pass


def test_a_learner_is_built_frozen():
    learner = _Counter()

    assert learner.mode is Mode.EVAL and not learner.is_training
    with pytest.raises(RuntimeError, match="frozen"):
        asyncio.run(learner.observe(None))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="frozen"):
        asyncio.run(learner.update([]))


def test_only_train_mode_lets_a_learner_learn():
    learner = _Counter()

    learner.set_mode(Mode.TRAIN)
    assert learner.mode is Mode.TRAIN and learner.is_training
    asyncio.run(learner.observe(None))  # type: ignore[arg-type]
    assert learner.games == 1

    learner.set_mode(Mode.EVAL)
    with pytest.raises(RuntimeError, match="frozen"):
        asyncio.run(learner.observe(None))  # type: ignore[arg-type]
    assert learner.games == 1
