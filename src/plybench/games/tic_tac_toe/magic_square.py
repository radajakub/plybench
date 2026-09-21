from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from plybench.configs.game_config import GameConfig
from plybench.configs.game_params import GameParams
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.games.generators.magic_squares import MagicSquare, MagicSquareGenerator
from plybench.games.tic_tac_toe.tic_tac_toe import TicTacToeAction, TicTacToeObservation
from plybench.utils.text import extract_params, to_bool
from plybench.visualization.grid import GridAxisLabel, GridPrinter, grid_to_positions


@dataclass(frozen=True, eq=True)
class MagicSquareGameParams(GameParams):
    sample: bool = False
    magic_constant_add: int = 0

    @classmethod
    def from_string(cls, params_string: str) -> MagicSquareGameParams:
        params = extract_params(params_string)
        return cls(sample=to_bool(params.get("sample", False)), magic_constant_add=int(params.get("magic_constant_add", 0)))

    def to_string(self) -> str:
        return f"sample={self.sample},magic_constant_add={self.magic_constant_add}"

    @property
    def path_suffix(self) -> str:
        return f"{'sample' if self.sample else 'normal'}_add_{self.magic_constant_add}"


class MagicSquareGame(TurnBasedGame):
    def __init__(self) -> None:
        super().__init__(game_type="magic_square", game_name="tic_tac_toe")


class MagicSquareTransformer(InterfaceTransformer[TicTacToeAction, TicTacToeObservation, [list[list[int]]]]):
    printer = GridPrinter(row_header=GridAxisLabel.NONE, col_header=GridAxisLabel.NONE)

    def __init__(self, sample: bool = False, magic_constant_add: int = 0) -> None:
        self.ms_gen = MagicSquareGenerator(sample=sample, magic_constant_add=magic_constant_add)
        self.square = self.ms_gen.new()

    def reset(self) -> None:
        self.square = self.ms_gen.new()

    def _inner_llm_action(self, action: TicTacToeAction) -> str:
        return f"N{self.square(action.row, action.col)}"

    def _inner_llm_partial_states(self, observation: TicTacToeObservation) -> list[str]:
        return []

    def _inner_llm_positions(self, observation: TicTacToeObservation) -> tuple[list[str], list[str]]:
        i_sign, o_sign = ("X", "O") if observation.player_order.is_first() else ("O", "X")
        return grid_to_positions(observation.state, i_sign, o_sign, lambda r, c: str(self.square(r - 1, c - 1)))

    def display_action(self, action: TicTacToeAction) -> str:
        return f"N{self.square(action.row, action.col)}"

    def _inner_llm_state(self, observation: TicTacToeObservation) -> str:
        unselected_numbers = []
        for row in range(len(observation.state)):
            for col in range(len(observation.state[row])):
                if observation.state[row][col] == ".":
                    unselected_numbers.append(self.square(row, col))
        return ", ".join(str(number) for number in sorted(unselected_numbers))

    def _inner_display_state(self, observation: TicTacToeObservation) -> str:
        state: list[list[str]] = []
        for row in range(len(observation.state)):
            state.append([])
            for col in range(len(observation.state[row])):
                if observation.state[row][col] == ".":
                    state[row].append(str(self.square(row, col)))
                else:
                    state[row].append(observation.state[row][col])
        return MagicSquareTransformer.printer(state)

    def set_state(self, layout: list[list[int]]) -> None:
        self.square = MagicSquare(np.array(layout))

    def get_other_params(self) -> dict[str, Any] | None:
        return {"layout": self.square.square.tolist()}


MAGIC_SQUARE_HEAD_PROMPT = """
This is a two-player game played with numbers.
Players take turns selecting numbers from a list of numbers from {start} to {end}.
When a player selects a number, it is removed from the list.
The goal is to have 3 numbers that sum to {magic_constant} among the selected numbers.
If all numbers are selected and no player has achieved the goal, the game is a draw.
Each move is represented by a string consisting of one part: the number (N).
For instance, N1 means selecting the number 1 from the list.
You are playing this game with the user (opponent).
"""


class MagicSquarePromptAdapter(PromptAdapter[[]]):
    def __init__(self, magic_constant_add: int = 0) -> None:
        super().__init__(head_prompt_template=MAGIC_SQUARE_HEAD_PROMPT, use_partial_state=False, position_name="numbers", order_actions=True)
        # base magic square is 1..9 summing to 15; every number shifts by the add
        start = 1 + magic_constant_add
        end = 9 + magic_constant_add
        magic_constant = 15 + magic_constant_add * 3
        self.head_prompt = self.head_prompt_template.format(start=start, end=end, magic_constant=magic_constant)

    def action_format(self) -> str:
        return "<Nx>, e.g., <N1>, <N7>"

    def restart_prompt(self) -> None:
        pass


class MagicSquareEngine(TurnBasedEngine[MagicSquareTransformer, MagicSquarePromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        params = cast(MagicSquareGameParams, game_config.params)
        transformer = MagicSquareTransformer(sample=params.sample, magic_constant_add=params.magic_constant_add)
        adapter = MagicSquarePromptAdapter(magic_constant_add=params.magic_constant_add)
        super().__init__(game_config, MagicSquareGame(), transformer, adapter, TicTacToeAction, TicTacToeObservation)

    def reset(self) -> None:
        self.interface_transformer.reset()
        self.game.reset()
