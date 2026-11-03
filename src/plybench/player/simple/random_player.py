from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.utils.text import extract_params


@dataclass(frozen=True, eq=True)
class RandomParams(PlayerParams):
    distribution: str = "uniform"

    @classmethod
    def from_string(cls, params_string: str) -> RandomParams:
        return cls(extract_params(params_string).get("distribution", "uniform"))

    def to_string(self) -> str:
        return f"distribution={self.distribution}"

    @property
    def path_suffix(self) -> str:
        return self.distribution


class RandomPlayer(Player):
    def __init__(self, player_config: PlayerConfig, identifier: PlayerIdentifier) -> None:
        super().__init__(player_config, identifier)
        params = cast(RandomParams, player_config.params)
        self._probs: Callable[[list[InterfaceAction]], np.ndarray]
        if params.distribution == "uniform":
            self._probs = self._uniform_probs
        elif params.distribution == "normal":
            self._probs = self._normal_probs
        else:
            raise ValueError(f"Invalid random distribution: {params.distribution}")

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        pass

    @staticmethod
    def _uniform_probs(legal_moves: list[InterfaceAction]) -> np.ndarray:
        return np.full(len(legal_moves), 1.0 / len(legal_moves))

    @staticmethod
    def _normal_probs(legal_moves: list[InterfaceAction]) -> np.ndarray:
        n = len(legal_moves)
        mu = (n - 1) / 2
        p = np.exp(-0.5 * ((np.arange(n) - mu) ** 2))
        return p / np.sum(p)

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        if len(legal_moves) == 0:
            return PlayerOutput(action=None)

        index = int(self.rng.choice(len(legal_moves), p=self._probs(legal_moves)))

        return PlayerOutput(action=legal_moves[index])

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""
