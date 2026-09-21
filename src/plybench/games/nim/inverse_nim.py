from __future__ import annotations

from typing import Any, Literal, cast

from plybench.configs.game_config import GameConfig
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.games.generators.nim import NimGenerator, NimInstance
from plybench.games.nim.nim import NimAction, NimGameParams, NimObservation
from plybench.utils.text import order_suffix
from plybench.visualization.piles import PilePrinter


class InverseNimGame(TurnBasedGame):
    @staticmethod
    def format_params(is_misere: bool, pile_sizes: list[int]) -> dict[str, Any]:
        return {"is_misere": is_misere, "pile_sizes": ";".join(str(pile) for pile in pile_sizes)}

    def __init__(self, is_misere: bool, pile_sizes: list[int]) -> None:
        super().__init__(game_type="inverse_nim", game_name="nim", params=InverseNimGame.format_params(is_misere, pile_sizes))


class InverseNimTransformer(InterfaceTransformer[NimAction, NimObservation, [list[int], int]]):
    printer = PilePrinter(column_label="pile", fixed_height=8)

    def __init__(self, sample: bool = False, max_pile_size: int = 8, num_piles: int = 4, pile_sum: int = 16, nim_start: Literal["winning", "losing"] = "winning") -> None:
        self.max_pile_size = max_pile_size
        self.nim_gen = NimGenerator(inverse=True, sample=sample, num_piles=num_piles, max_pile_size=max_pile_size, pile_sum=pile_sum, nim_start=nim_start, allow_zero=True)
        self.instance = self.nim_gen.new()

    def _inner_llm_action(self, action: NimAction) -> str:
        return f"pile:{action.pile + 1}, add:{action.take}"

    def display_action(self, action: NimAction) -> str:
        return f"pile:{action.pile + 1}, add:{action.take}"

    def _inner_llm_state(self, observation: NimObservation) -> str:
        return " ".join(str(self.max_pile_size - pile) for pile in observation.state)

    def _inner_llm_partial_states(self, observation: NimObservation) -> list[str]:
        return [f"the {i}{order_suffix(i)} pile has {self.max_pile_size - pile} matches" for i, pile in enumerate(observation.state, start=1)]

    def _inner_llm_positions(self, observation: NimObservation) -> tuple[list[str], list[str]]:
        return [], []

    def _inner_display_state(self, observation: NimObservation) -> str:
        return InverseNimTransformer.printer([self.max_pile_size - pile for pile in observation.state])

    def get_original_pile_sizes(self) -> list[int]:
        return [self.max_pile_size - pile for pile in self.instance.pile_sizes]

    def get_pile_sizes(self) -> list[int]:
        return self.instance.pile_sizes

    def reset(self) -> None:
        self.instance = self.nim_gen.new()

    def set_state(self, pile_sizes: list[int], max_pile_size: int) -> None:
        self.instance = NimInstance(pile_sizes, max_pile_size)

    def get_other_params(self) -> dict[str, Any] | None:
        return None


INVERSE_NIM_HEAD_PROMPT = """
You are playing a two-player game with a set of {num_piles} piles containing {pile_sizes} matches respectively, players aim to avoid adding the last match.
During each turn, a player may add any number of matches to a single pile, but must add at least one and cannot exceed the maximum number of each pile, which is {max_pile_size}.
The objective is to force the opponent to add the last match, thereby winning the game.
The action is presented in <pile:x, add:y>, which means add y match(es) to the x-th pile.
"""


class InverseNimPromptAdapter(PromptAdapter[[int, list[int], int]]):
    def __init__(self, max_pile_size: int = 8) -> None:
        super().__init__(head_prompt_template=INVERSE_NIM_HEAD_PROMPT, use_partial_state=True, order_actions=False)
        self.max_pile_size = max_pile_size

    def action_format(self) -> str:
        return "<pile:x, add:y>, e.g., <pile:1, add:1>, <pile:4, add:7>"

    def restart_prompt(self, num_piles: int, pile_sizes: list[int], max_pile_size: int) -> None:
        pile_sizes_string = ", ".join(str(pile) for pile in pile_sizes[:-1]) + ", and " + str(pile_sizes[-1])
        self.head_prompt = self.head_prompt_template.format(num_piles=num_piles, pile_sizes=pile_sizes_string, max_pile_size=max_pile_size)


class InverseNimEngine(TurnBasedEngine[InverseNimTransformer, InverseNimPromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        self.nim_params = cast(NimGameParams, game_config.params)
        transformer = InverseNimTransformer(
            sample=self.nim_params.sample,
            num_piles=self.nim_params.num_piles,
            max_pile_size=self.nim_params.max_pile_size,
            pile_sum=self.nim_params.pile_sum,
            nim_start=self.nim_params.nim_start,
        )
        # the instance is held in inverse space; the underlying game needs the complemented piles, same as reset()
        game = InverseNimGame(is_misere=True, pile_sizes=transformer.get_original_pile_sizes())
        adapter = InverseNimPromptAdapter(max_pile_size=self.nim_params.max_pile_size)
        super().__init__(game_config, game, transformer, adapter, NimAction, NimObservation)

    def reset(self) -> None:
        self.interface_transformer.reset()
        game_pile_sizes = self.interface_transformer.get_original_pile_sizes()
        adapter_pile_sizes = self.interface_transformer.get_pile_sizes()
        self.game = InverseNimGame(is_misere=True, pile_sizes=game_pile_sizes)
        self.prompt_adapter.restart_prompt(len(game_pile_sizes), adapter_pile_sizes, self.nim_params.max_pile_size)
