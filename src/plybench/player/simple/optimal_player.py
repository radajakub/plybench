from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import numpy as np

from plybench.common.paths import MinimaxPathBuilder
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.minimax import AVQ, AVQCache, solve_game
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.utils.text import extract_params, to_bool


@dataclass(frozen=True, eq=True)
class OptimalParams(PlayerParams):
    stochastic: bool = False
    eps: float = 0.0

    @classmethod
    def from_string(cls, params_string: str) -> OptimalParams:
        params = extract_params(params_string)
        return cls(stochastic=to_bool(params.get("stochastic", False)), eps=float(params.get("eps", 0)))

    def to_string(self) -> str:
        # eps is part of what this opponent IS, so it has to round-trip and land in every record; it is
        # appended only when set, which keeps every existing `stochastic=...` string and path unchanged
        base = f"stochastic={self.stochastic}"
        return f"{base},eps={self.eps}" if self.eps else base

    @property
    def path_suffix(self) -> str:
        base = "stochastic" if self.stochastic else "deterministic"
        return f"{base}_eps{self.eps}" if self.eps else base


class Judgeable(ABC):
    @abstractmethod
    def optimal(self, player: int, observation: InterfaceObservation) -> AVQ | None:
        raise NotImplementedError


class OptimalPlayer(Player, Judgeable):
    def __init__(self, game: TurnBasedGame, player_config: PlayerConfig, identifier: PlayerIdentifier) -> None:
        super().__init__(player_config, identifier)

        self._params = cast(OptimalParams, player_config.params)

        self._cache: AVQCache | None = None

        self._path_builder = MinimaxPathBuilder()

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        # load the solved value cache from disk, or solve the game and persist it
        path = self._path_builder.cache(game.game_name, game.params)
        if path.exists():
            self._cache = AVQCache.load(str(path))
        else:
            self._cache = solve_game(game)
            self._cache.save(str(path))

    def _verdict(self, player: int, os_state: str) -> AVQ:
        assert self._cache is not None, "Optimal policy not initialized"

        entry = self._cache[player, os_state]

        if entry is None:
            raise ValueError(f"Optimal action not found for state {os_state} and player {player}")

        return entry

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        optimal_numbers = self._verdict(game.get_player(), observation.os_observation.state).A()

        if np.random.rand() < self._params.eps:
            return PlayerOutput(action=legal_moves[int(np.random.choice(len(legal_moves)))])

        number = int(np.random.choice(optimal_numbers)) if self._params.stochastic else optimal_numbers[0]

        action = next((move for move in legal_moves if move.number == number), None)

        return PlayerOutput(action=action)

    def optimal(self, player: int, observation: InterfaceObservation) -> AVQ | None:
        return self._verdict(player, observation.os_observation.state)

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""
