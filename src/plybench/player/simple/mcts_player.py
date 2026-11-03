from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from open_spiel.python.algorithms import mcts

from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.utils.text import extract_params


@dataclass(frozen=True, eq=True)
class MctsParams(PlayerParams):
    max_simulations: int = 1000
    rollout_count: int = 1
    uct_c: float = 2.0

    @classmethod
    def from_string(cls, params_string: str) -> MctsParams:
        params = extract_params(params_string)
        return cls(
            max_simulations=int(params.get("max_simulations", 1000)),
            rollout_count=int(params.get("rollout_count", 1)),
            uct_c=float(params.get("uct_c", 2)),
        )

    def to_string(self) -> str:
        return f"max_simulations={self.max_simulations},rollout_count={self.rollout_count},uct_c={self.uct_c}"

    @property
    def path_suffix(self) -> str:
        return f"{self.max_simulations}_sims_{self.rollout_count}_rollouts_{self.uct_c}_uct"


class MCTSPlayer(Player):
    def __init__(self, game: TurnBasedGame, player_config: PlayerConfig, identifier: PlayerIdentifier) -> None:
        super().__init__(player_config, identifier)

        self._params = cast(MctsParams, player_config.params)
        self._rng = np.random.RandomState()

        self._bot: mcts.MCTSBot | None = None

    def seed(self, seed: int) -> None:
        super().seed(seed)
        # pyspiel's MCTS draws from a legacy RandomState, not a Generator; the bot is built after this, in initialize_policy
        self._rng = np.random.RandomState(seed)

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        self._bot = mcts.MCTSBot(
            game=game.game,
            uct_c=self._params.uct_c,
            max_simulations=self._params.max_simulations,
            evaluator=mcts.RandomRolloutEvaluator(n_rollouts=self._params.rollout_count, random_state=self._rng),
            random_state=self._rng,
            solve=True,
            verbose=False,
        )

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        if self._bot is None:
            raise ValueError("MCTS policy not initialized")

        action_number = self._bot.step(game.state.clone())
        action = next((move for move in legal_moves if move.number == action_number), None)

        return PlayerOutput(action=action)

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""
