from __future__ import annotations

from typing import Any, cast

import numpy as np

from plybench.configs.game_config import GameConfig
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.games.generators.magic_squares import MagicSquare, MagicSquareGenerator
from plybench.games.tic_tac_toe.magic_square import MagicSquareGameParams
from plybench.games.tic_tac_toe.tic_tac_toe import TicTacToeAction, TicTacToeObservation
from plybench.visualization.grid import GridAxisLabel, GridPrinter, grid_to_positions


class StoryMagicSquare(TurnBasedGame):
    def __init__(self) -> None:
        super().__init__(game_type="story_magic_square", game_name="tic_tac_toe")


class StoryMagicSquareTransformer(InterfaceTransformer[TicTacToeAction, TicTacToeObservation, [list[list[int]]]]):
    printer = GridPrinter(row_header=GridAxisLabel.NONE, col_header=GridAxisLabel.NONE)

    def __init__(self, sample: bool = False) -> None:
        self.ms_gen = MagicSquareGenerator(sample=sample, magic_constant_add=0)
        self.square = self.ms_gen.new()

    def reset(self) -> None:
        self.square = self.ms_gen.new()

    def _inner_llm_action(self, action: TicTacToeAction) -> str:
        return f"jump:{self.square(action.row, action.col)}"

    def _inner_llm_partial_states(self, observation: TicTacToeObservation) -> list[str]:
        return []

    def _inner_llm_positions(self, observation: TicTacToeObservation) -> tuple[list[str], list[str]]:
        i_sign, o_sign = ("X", "O") if observation.player_order.is_first() else ("O", "X")
        return grid_to_positions(observation.state, i_sign, o_sign, lambda r, c: str(self.square(r - 1, c - 1)))

    def display_action(self, action: TicTacToeAction) -> str:
        return f"jump:{self.square(action.row, action.col)}"

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
        return StoryMagicSquareTransformer.printer(state)

    def get_other_params(self) -> dict[str, Any] | None:
        return {"layout": self.square.square.tolist()}

    def set_state(self, layout: list[list[int]]) -> None:
        self.square = MagicSquare(np.array(layout))


STORY_MAGIC_SQUARE_HEAD_PROMPT = """
You and your opponent are stuck on one bank of a river and need to get to the other side as fast as possible.
Inside the river there are some evenly spaced stones that you can jump on to cross the river.
In total there are 14 stones which makes the number of jumps needed to cross the river exactly 15.
Both of you are quite good at jumping so you can cover 1 to 9 distances in a single jump.
however, you are competitive with each other and really don't want to copy each other's jumps.
So once a certain number of distances is jumped, no one can jump the same number of distances ever again.
Also, you need to jump the exact 15 distances to cross the river, if you are short, you will be caught by the crocodiles waiting in the river, if you are long, you will fall off a cliff.
Therefore, your goal is to plan exactly three jumps of different lengths from 1 to 9 that will get you exactly to the other side of the river.
Morever, you must do the planning faster than your opponent, otherwise he can push you into the river.
Both of you alternate in selecting the jumps until one of you can get safely to the other side and can start jumping.
Each selected jump is represented by a string consisiting of the number of distances to jump.
For instance, jump:1 means jumping 1 distance, jump:7 means jumping 7 distances.
"""


class StoryMagicSquarePromptAdapter(PromptAdapter[[]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=STORY_MAGIC_SQUARE_HEAD_PROMPT, use_partial_state=False, position_name="jumps", order_actions=True)
        self.head_prompt = self.head_prompt_template

    def action_format(self) -> str:
        return "<jump:number_of_distances_to_jump>, e.g., <jump:1>, <jump:7>"

    def restart_prompt(self) -> None:
        pass


class StoryMagicSquareEngine(TurnBasedEngine[StoryMagicSquareTransformer, StoryMagicSquarePromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        params = cast(MagicSquareGameParams, game_config.params)
        transformer = StoryMagicSquareTransformer(sample=params.sample)
        super().__init__(game_config, StoryMagicSquare(), transformer, StoryMagicSquarePromptAdapter(), TicTacToeAction, TicTacToeObservation)

    def reset(self) -> None:
        self.interface_transformer.reset()
        self.game.reset()
