from __future__ import annotations

from typing import Any

from plybench.configs.game_config import GameConfig
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.games.tic_tac_toe.tic_tac_toe import TicTacToeAction, TicTacToeObservation
from plybench.visualization.grid import GridAxisLabel, GridPrinter, grid_to_positions


class ModifiedTicTacToeGame(TurnBasedGame):
    def __init__(self) -> None:
        super().__init__(game_type="modified_tic_tac_toe", game_name="tic_tac_toe")


class ModifiedTicTacToeTransformer(InterfaceTransformer[TicTacToeAction, TicTacToeObservation, []]):
    printer = GridPrinter(row_header=GridAxisLabel.NUMBERS, col_header=GridAxisLabel.NUMBERS)

    def symbol_forward(self, symbol: str) -> str:
        if symbol in ("X", "x"):
            return "B"
        if symbol in ("O", "o"):
            return "W"
        return symbol

    def _inner_llm_action(self, action: TicTacToeAction) -> str:
        return f"C{action.col + 1}R{action.row + 1}"

    def _inner_llm_partial_states(self, observation: TicTacToeObservation) -> list[str]:
        return []

    def _inner_llm_positions(self, observation: TicTacToeObservation) -> tuple[list[str], list[str]]:
        i_sign, o_sign = ("X", "O") if observation.player_order.is_first() else ("O", "X")
        return grid_to_positions(observation.state, i_sign, o_sign, lambda r, c: f"C{c}R{r}")

    def display_action(self, action: TicTacToeAction) -> str:
        return f"{self.symbol_forward(action.symbol)}[{action.row + 1}, {action.col + 1}]"

    def _inner_llm_state(self, observation: TicTacToeObservation) -> str:
        return "\n".join("".join(self.symbol_forward(cell) for cell in row) for row in observation.state)

    def _inner_display_state(self, observation: TicTacToeObservation) -> str:
        state = [[self.symbol_forward(cell) for cell in row] for row in observation.state]
        return ModifiedTicTacToeTransformer.printer(state)

    def reset(self) -> None:
        pass

    def set_state(self) -> None:
        pass

    def get_other_params(self) -> dict[str, Any] | None:
        return None


MODIFIED_TIC_TAC_TOE_HEAD_PROMPT = """
You are playing a two-player game on a 3x3 grid.
Players take turns filling the grid with their color, black ('B') or white ('W').
The goal is to get 3 of one's own colors in a row, either horizontally, vertically, or diagonally, before the opponent does.
If all nine squares are filled and no player has three in a row, the game is a draw.
Each move is represented by a string consisting of two parts: the column (C) and the row (R), in that order.
For instance, C2R1 means the placement of a color at the position of the first row and the second column of the grid.
You are playing this game with the user (opponent).
"""


class ModifiedTicTacToePromptAdapter(PromptAdapter[[]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=MODIFIED_TIC_TAC_TOE_HEAD_PROMPT, use_partial_state=False, position_name="positions", order_actions=True)
        self.head_prompt = self.head_prompt_template

    def action_format(self) -> str:
        return "<CxRy>, e.g., <C1R1>, <C3R3>"

    def restart_prompt(self) -> None:
        pass


class ModifiedTicTacToeEngine(TurnBasedEngine[ModifiedTicTacToeTransformer, ModifiedTicTacToePromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        super().__init__(game_config, ModifiedTicTacToeGame(), ModifiedTicTacToeTransformer(), ModifiedTicTacToePromptAdapter(), TicTacToeAction, TicTacToeObservation)

    def reset(self) -> None:
        self.game.reset()
