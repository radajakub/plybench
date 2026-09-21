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
from plybench.visualization.characters import CharacterPrinter


class StoryNimGame(TurnBasedGame):
    @staticmethod
    def format_params(is_misere: bool, pile_sizes: list[int]) -> dict[str, Any]:
        return {"is_misere": is_misere, "pile_sizes": ";".join(str(pile) for pile in pile_sizes)}

    def __init__(self, is_misere: bool, pile_sizes: list[int]) -> None:
        super().__init__(game_type="story_nim", game_name="nim", params=StoryNimGame.format_params(is_misere, pile_sizes))


class StoryNimTransformer(InterfaceTransformer[NimAction, NimObservation, [list[int], int]]):
    names = ["alice", "bob", "charlie", "dave"]
    printer = CharacterPrinter(fixed_width=8)

    @staticmethod
    def idx_to_name(index: int) -> str:
        return StoryNimTransformer.names[index]

    def __init__(self, sample: bool = False, max_pile_size: int = 8, num_piles: int = 4, pile_sum: int = 16, nim_start: Literal["winning", "losing"] = "winning") -> None:
        self.max_pile_size = max_pile_size
        self.nim_gen = NimGenerator(inverse=True, sample=sample, num_piles=num_piles, max_pile_size=max_pile_size, pile_sum=pile_sum, nim_start=nim_start, allow_zero=False)
        self.instance = self.nim_gen.new()

    def _inner_llm_action(self, action: NimAction) -> str:
        return f"{StoryNimTransformer.idx_to_name(action.pile)}:{action.take}"

    def display_action(self, action: NimAction) -> str:
        return f"{StoryNimTransformer.idx_to_name(action.pile)}:{action.take}"

    def _inner_llm_state(self, observation: NimObservation) -> str:
        return " ".join(str(self.max_pile_size - pile) for pile in observation.state)

    def _handle_end(self, name: str, position: int) -> str:
        if position == self.max_pile_size:
            return f"{name} is at the end"
        return f"{name} is on the {position}{order_suffix(position)} square"

    def _inner_llm_partial_states(self, observation: NimObservation) -> list[str]:
        return [self._handle_end(StoryNimTransformer.idx_to_name(i), self.max_pile_size - position) for i, position in enumerate(observation.state)]

    def _inner_llm_positions(self, observation: NimObservation) -> tuple[list[str], list[str]]:
        return [], []

    def _inner_display_state(self, observation: NimObservation) -> str:
        return StoryNimTransformer.printer(names=StoryNimTransformer.names, positions=[self.max_pile_size - pile for pile in observation.state])

    def get_original_pile_sizes(self) -> list[int]:
        return [self.max_pile_size - pile for pile in self.instance.pile_sizes]

    def reset(self) -> None:
        self.instance = self.nim_gen.new()

    def set_state(self, pile_sizes: list[int], max_pile_size: int) -> None:
        self.instance = NimInstance(pile_sizes, max_pile_size)

    def get_other_params(self) -> dict[str, Any] | None:
        return None


STORY_NIM_HEAD_PROMPT = """
There are four characters with names {names}, each of them positioned on a line made of {num_squares} squares.
They can move only on their line to the right by any number of squares at a time until they reach the end of the line.
You and your opponent try to help them get to the end but you are also competing with each other and see which of you can help faster.
Therefore, your goal is to help the characters reach the end but you must make sure that you are not the one that helps the last character reach their end of the line.
If you are the last one, you lose and the opponent wins.
You are taking turns with the opponent and in each turn you can select any one of the characters and move it to the right by some number of squares.
The chosen character and the number of squares to move it are formatted as <character_name:number_of_squares_to_move_right>., e.g., <alice:3>.
"""


class StoryNimPromptAdapter(PromptAdapter[[list[str], int]]):
    def __init__(self, max_pile_size: int = 8) -> None:
        super().__init__(head_prompt_template=STORY_NIM_HEAD_PROMPT, use_partial_state=True, order_actions=False)
        self.max_pile_size = max_pile_size

    def action_format(self) -> str:
        return "<name:move_right>, e.g., <alice:3>"

    def restart_prompt(self, names: list[str], max_pile_size: int) -> None:
        self.head_prompt = self.head_prompt_template.format(names=", ".join(names), num_squares=max_pile_size)


class StoryNimEngine(TurnBasedEngine[StoryNimTransformer, StoryNimPromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        self.nim_params = cast(NimGameParams, game_config.params)
        transformer = StoryNimTransformer(
            sample=self.nim_params.sample,
            num_piles=self.nim_params.num_piles,
            max_pile_size=self.nim_params.max_pile_size,
            pile_sum=self.nim_params.pile_sum,
            nim_start=self.nim_params.nim_start,
        )
        # the instance is held in inverse space; the underlying game needs the complemented piles, same as reset()
        game = StoryNimGame(is_misere=True, pile_sizes=transformer.get_original_pile_sizes())
        adapter = StoryNimPromptAdapter(max_pile_size=self.nim_params.max_pile_size)
        super().__init__(game_config, game, transformer, adapter, NimAction, NimObservation)

    def reset(self) -> None:
        self.interface_transformer.reset()
        pile_sizes = self.interface_transformer.get_original_pile_sizes()
        self.game = StoryNimGame(is_misere=True, pile_sizes=pile_sizes)
        self.prompt_adapter.restart_prompt(StoryNimTransformer.names, self.nim_params.max_pile_size)
