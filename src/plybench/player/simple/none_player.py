from __future__ import annotations

from dataclasses import dataclass

from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerOutput

# the key of the player that fills the empty seat of a single-player game
NONE_PLAYER = "none"


@dataclass(frozen=True, eq=True)
class NoneParams(PlayerParams):
    @classmethod
    def from_string(cls, params_string: str) -> NoneParams:
        return cls()

    def to_string(self) -> str:
        return ""

    @property
    def path_suffix(self) -> str:
        return ""


class NonePlayer(Player):
    # the empty seat of a single-player game: the engine never asks it to move
    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        pass

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        raise RuntimeError(f"The {NONE_PLAYER} player fills the empty seat of a single-player game and is never asked to move")

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""
