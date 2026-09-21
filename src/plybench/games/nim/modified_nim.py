from __future__ import annotations

from typing import Any

from plybench.configs.game_config import GameConfig
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.games.nim.nim import NimAction, NimObservation
from plybench.utils.text import order_suffix
from plybench.visualization.piles import PilePrinter


class ModifiedNimGame(TurnBasedGame):
    def __init__(self) -> None:
        super().__init__(game_type="modified_nim", game_name="nim")


class ModifiedNimTransformer(InterfaceTransformer[NimAction, NimObservation, []]):
    printer = PilePrinter(column_label="lamp")

    def _inner_llm_action(self, action: NimAction) -> str:
        return f"lamp:{action.pile + 1}, decrease:{action.take}"

    def display_action(self, action: NimAction) -> str:
        return f"lamp:{action.pile + 1}, decrease:{action.take}"

    def _inner_llm_state(self, observation: NimObservation) -> str:
        return " ".join(str(pile) for pile in observation.state)

    def _inner_llm_partial_states(self, observation: NimObservation) -> list[str]:
        return [f"the {i}{order_suffix(i)} lamp has brightness level {pile}" for i, pile in enumerate(observation.state, start=1)]

    def _inner_llm_positions(self, observation: NimObservation) -> tuple[list[str], list[str]]:
        return [], []

    def _inner_display_state(self, observation: NimObservation) -> str:
        return ModifiedNimTransformer.printer(observation.state)

    def reset(self) -> None:
        pass

    def set_state(self) -> None:
        pass

    def get_other_params(self) -> dict[str, Any] | None:
        return None


MODIFIED_NIM_HEAD_PROMPT = """
You are playing a two-player strategic game with an array of 4 lamps with various brightness levels.
Initially, each lamp is turned on to some brightness level, specifically 1, 3, 5, and 7.
A player that turns off the last lamp loses the game.
During each turn, a player may decrease the brightness level of a single lamp by some number of levels, but must decrease by at least one level.
The objective is to force the opponent to turn off the last lamp, thereby winning the game.
The action is formatted as <lamp:x, decrease:y>, which means decrease the brightness level of the x-th lamp by y levels.
"""


class ModifiedNimPromptAdapter(PromptAdapter[[]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=MODIFIED_NIM_HEAD_PROMPT, use_partial_state=True, order_actions=False)
        self.head_prompt = self.head_prompt_template

    def action_format(self) -> str:
        return "<lamp:x, decrease:y>, e.g., <lamp:1, decrease:1>, <lamp:4, decrease:7>"

    def restart_prompt(self) -> None:
        pass


class ModifiedNimEngine(TurnBasedEngine[ModifiedNimTransformer, ModifiedNimPromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        super().__init__(game_config, ModifiedNimGame(), ModifiedNimTransformer(), ModifiedNimPromptAdapter(), NimAction, NimObservation)

    def reset(self) -> None:
        self.game.reset()
