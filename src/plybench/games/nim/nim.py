from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from plybench.configs.game_config import GameConfig
from plybench.configs.game_params import GameParams
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.games.generators.nim import NimGenerator, NimInstance
from plybench.utils.text import extract_params, order_suffix, to_bool
from plybench.visualization.piles import PilePrinter


@dataclass(frozen=True, eq=True)
class NimGameParams(GameParams):
    sample: bool = False
    num_piles: int = 4
    max_pile_size: int = 8
    pile_sum: int = 16
    nim_start: Literal["winning", "losing"] = "winning"

    @classmethod
    def from_string(cls, params_string: str) -> NimGameParams:
        params = extract_params(params_string)
        nim_start = params.get("nim_start", "winning")
        if nim_start not in ("winning", "losing"):
            raise ValueError(f"Invalid nim_start: {nim_start}")
        return cls(
            sample=to_bool(params.get("sample", False)),
            num_piles=int(params.get("num_piles", 4)),
            max_pile_size=int(params.get("max_pile_size", 8)),
            pile_sum=int(params.get("pile_sum", 16)),
            nim_start=nim_start,
        )

    def to_string(self) -> str:
        if not self.sample:
            return ""
        return f"sample=True,num_piles={self.num_piles},max_pile_size={self.max_pile_size},pile_sum={self.pile_sum},nim_start={self.nim_start}"

    @property
    def path_suffix(self) -> str:
        if not self.sample:
            return ""
        return f"sample_{self.num_piles}_piles_{self.max_pile_size}_max_size_{self.pile_sum}_sum_{self.nim_start}"


class NimGame(TurnBasedGame):
    @staticmethod
    def format_params(is_misere: bool, pile_sizes: list[int]) -> dict[str, Any]:
        return {"is_misere": is_misere, "pile_sizes": ";".join(str(pile) for pile in pile_sizes)}

    def __init__(self, is_misere: bool, pile_sizes: list[int]) -> None:
        super().__init__(game_type="nim", game_name="nim", params=NimGame.format_params(is_misere, pile_sizes))


class NimAction(InterfaceAction):
    @staticmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> NimAction:
        match = re.match(r"pile:(\d+), take:(\d+);", action.string)
        if not match:
            raise ValueError(f"Invalid action string format: {action.string}")
        pile = int(match.group(1)) - 1
        take = int(match.group(2))
        return NimAction(pile, take, action.number, interface_transformer)

    def __init__(self, pile: int, take: int, number: int, interface_transformer: InterfaceTransformer) -> None:
        super().__init__(number=number, interface_transformer=interface_transformer)
        self.pile = pile
        self.take = take

    def to_openspiel(self) -> OpenSpielAction:
        return OpenSpielAction(self.number, f"pile:{self.pile + 1}, take:{self.take};")


class NimObservation(InterfaceObservation):
    @staticmethod
    def _state_from_openspiel(state: str) -> list[int]:
        return [int(pile) for pile in state.split(" ")[1:]]

    @staticmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> NimObservation:
        state = NimObservation._state_from_openspiel(observation.state)
        i_actions = [NimAction.from_openspiel(action, interface_transformer) for action in observation.i_actions]
        o_actions = [NimAction.from_openspiel(action, interface_transformer) for action in observation.o_actions]
        return NimObservation(observation, state, i_actions, o_actions, interface_transformer)

    def __init__(
        self, os_observation: OpenSpielObservation, state: list[int], i_actions: list[NimAction], o_actions: list[NimAction], interface_transformer: InterfaceTransformer
    ) -> None:
        super().__init__(os_observation, i_actions, o_actions, interface_transformer)
        self.state = state


class NimTransformer(InterfaceTransformer[NimAction, NimObservation, [list[int], int]]):
    printer = PilePrinter(column_label="pile", fixed_height=8)

    def __init__(self, sample: bool = False, max_pile_size: int = 8, num_piles: int = 4, pile_sum: int = 16, nim_start: Literal["winning", "losing"] = "winning") -> None:
        self.nim_gen = NimGenerator(inverse=False, sample=sample, num_piles=num_piles, max_pile_size=max_pile_size, pile_sum=pile_sum, nim_start=nim_start, allow_zero=True)
        self.instance = self.nim_gen.new()

    def _inner_llm_action(self, action: NimAction) -> str:
        return f"pile:{action.pile + 1}, take:{action.take}"

    def display_action(self, action: NimAction) -> str:
        return f"pile:{action.pile + 1}, take:{action.take}"

    def _inner_llm_state(self, observation: NimObservation) -> str:
        return " ".join(str(pile) for pile in observation.state)

    def _inner_llm_partial_states(self, observation: NimObservation) -> list[str]:
        return [f"the {i}{order_suffix(i)} pile has {pile} matches" for i, pile in enumerate(observation.state, start=1)]

    def _inner_llm_positions(self, observation: NimObservation) -> tuple[list[str], list[str]]:
        return [], []

    def _inner_display_state(self, observation: NimObservation) -> str:
        return NimTransformer.printer(observation.state)

    def reset(self) -> None:
        self.instance = self.nim_gen.new()

    def set_state(self, pile_sizes: list[int], max_pile_size: int) -> None:
        self.instance = NimInstance(pile_sizes, max_pile_size)

    def get_other_params(self) -> dict[str, Any] | None:
        return None


NIM_HEAD_PROMPT = """
In Nim, a strategic game with a set of {num_piles} piles containing {pile_sizes} matches respectively, players aim to avoid taking the last match.
During each turn, a player may take any number of matches from a single pile, but must take at least one and cannot exceed the number remaining in that pile.
The objective is to force the opponent to pick up the final match, thereby winning the game.
The action is presented in <pile:x, take:y>, which means take y match(es) from the x-th pile.
"""


class NimPromptAdapter(PromptAdapter[[int, list[int]]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=NIM_HEAD_PROMPT, use_partial_state=True, position_name="", order_actions=False)

    def action_format(self) -> str:
        return "<pile:x, take:y>, e.g., <pile:1, take:1>, <pile:4, take:7>"

    def restart_prompt(self, num_piles: int, pile_sizes: list[int]) -> None:
        pile_sizes_string = ", ".join(str(pile) for pile in pile_sizes[:-1]) + ", and " + str(pile_sizes[-1])
        self.head_prompt = self.head_prompt_template.format(num_piles=num_piles, pile_sizes=pile_sizes_string)


class NimEngine(TurnBasedEngine[NimTransformer, NimPromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        params = cast(NimGameParams, game_config.params)
        transformer = NimTransformer(sample=params.sample, max_pile_size=params.max_pile_size, num_piles=params.num_piles, pile_sum=params.pile_sum, nim_start=params.nim_start)
        game = NimGame(is_misere=True, pile_sizes=transformer.instance.pile_sizes)
        super().__init__(game_config, game, transformer, NimPromptAdapter(), NimAction, NimObservation)

    def reset(self) -> None:
        self.interface_transformer.reset()
        pile_sizes = self.interface_transformer.instance.pile_sizes
        self.game = NimGame(is_misere=True, pile_sizes=pile_sizes)
        self.prompt_adapter.restart_prompt(len(pile_sizes), pile_sizes)
