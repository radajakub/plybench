from __future__ import annotations

import re
from typing import Any

from plybench.configs.game_config import GameConfig
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.visualization.grid import GridAxisLabel, GridPrinter, grid_to_positions


class TicTacToeGame(TurnBasedGame):
    def __init__(self) -> None:
        super().__init__(game_type="tic_tac_toe", game_name="tic_tac_toe")


class TicTacToeAction(InterfaceAction):
    @staticmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> TicTacToeAction:
        match = re.match(r"([ox])\((\d+),\s*(\d+)\)", action.string)
        if not match:
            raise ValueError(f"Invalid action string format: {action.string}")
        symbol = match.group(1)
        row = int(match.group(2))
        col = int(match.group(3))
        return TicTacToeAction(symbol, row, col, action.number, interface_transformer)

    def __init__(self, symbol: str, row: int, col: int, number: int, interface_transformer: InterfaceTransformer) -> None:
        super().__init__(number=number, interface_transformer=interface_transformer)
        self.symbol = symbol
        self.row = row
        self.col = col

    def to_openspiel(self) -> OpenSpielAction:
        return OpenSpielAction(self.number, f"{self.symbol.lower()}({self.row},{self.col})")


class TicTacToeObservation(InterfaceObservation):
    @staticmethod
    def _format_cell(cell: str) -> str:
        if cell == "x":
            return "X"
        if cell == "o":
            return "O"
        return "."

    @staticmethod
    def _state_from_openspiel(state: str) -> list[list[str]]:
        rows = state.split("\n")
        return [[TicTacToeObservation._format_cell(cell) for cell in row] for row in rows]

    @staticmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> TicTacToeObservation:
        state = TicTacToeObservation._state_from_openspiel(observation.state)
        i_actions = [TicTacToeAction.from_openspiel(action, interface_transformer) for action in observation.i_actions]
        o_actions = [TicTacToeAction.from_openspiel(action, interface_transformer) for action in observation.o_actions]
        return TicTacToeObservation(observation, state, i_actions, o_actions, interface_transformer)

    def __init__(
        self,
        os_observation: OpenSpielObservation,
        state: list[list[str]],
        i_actions: list[TicTacToeAction],
        o_actions: list[TicTacToeAction],
        interface_transformer: InterfaceTransformer,
    ) -> None:
        super().__init__(os_observation, i_actions, o_actions, interface_transformer)
        self.state = state


class TicTacToeTransformer(InterfaceTransformer[TicTacToeAction, TicTacToeObservation, []]):
    printer = GridPrinter(row_header=GridAxisLabel.NUMBERS, col_header=GridAxisLabel.NUMBERS)

    def _inner_llm_action(self, action: TicTacToeAction) -> str:
        return f"C{action.col + 1}R{action.row + 1}"

    def display_action(self, action: TicTacToeAction) -> str:
        return f"{action.symbol}[{action.row + 1}, {action.col + 1}]"

    def _inner_llm_state(self, observation: TicTacToeObservation) -> str:
        return "\n".join("".join(row) for row in observation.state)

    def _inner_llm_partial_states(self, observation: TicTacToeObservation) -> list[str]:
        return []

    def _inner_llm_positions(self, observation: TicTacToeObservation) -> tuple[list[str], list[str]]:
        i_sign, o_sign = ("X", "O") if observation.player_order.is_first() else ("O", "X")
        return grid_to_positions(observation.state, i_sign, o_sign, lambda r, c: f"C{c}R{r}")

    def _inner_display_state(self, observation: TicTacToeObservation) -> str:
        return TicTacToeTransformer.printer(observation.state)

    def reset(self) -> None:
        pass

    def set_state(self) -> None:
        pass

    def get_other_params(self) -> dict[str, Any] | None:
        return None


TIC_TAC_TOE_HEAD_PROMPT = """
Tic Tac Toe is a two-player game played on a grid.
Players take turns marking a space with their respective symbols.
The goal is to get 3 of one's own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does.
If all nine squares are filled and no player has three in a row, the game is a draw.
The Tic Tac Toe game is played on a 3 by 3 grid, with the winning length as 3.
Each move is represented by a string consisting of two parts: the column (C) and the row (R), in that order.
For instance, C1R2 means the movement at the position of the first column and the second row of the grid.
You are playing this game with the user (opponent).
"""


class TicTacToePromptAdapter(PromptAdapter[[]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=TIC_TAC_TOE_HEAD_PROMPT, use_partial_state=False, position_name="positions", order_actions=True)
        self.head_prompt = self.head_prompt_template

    def action_format(self) -> str:
        return "<CxRy>, e.g., <C1R1>, <C3R3>"

    def restart_prompt(self) -> None:
        pass


class TicTacToeEngine(TurnBasedEngine[TicTacToeTransformer, TicTacToePromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        super().__init__(game_config, TicTacToeGame(), TicTacToeTransformer(), TicTacToePromptAdapter(), TicTacToeAction, TicTacToeObservation)

    def reset(self) -> None:
        self.game.reset()
