from __future__ import annotations

import re
from typing import Any

from plybench.configs.game_config import GameConfig
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.visualization.grid import GridAxisLabel, GridPrinter, grid_to_positions


class ConnectFourGame(TurnBasedGame):
    def __init__(self) -> None:
        super().__init__(game_type="connect_four", game_name="connect_four")


class ConnectFourAction(InterfaceAction):
    @staticmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> ConnectFourAction:
        match = re.match(r"([ox])(\d+)", action.string)
        if not match:
            raise ValueError(f"Invalid action string format: {action.string}")
        symbol = match.group(1)
        column = int(match.group(2))
        return ConnectFourAction(symbol, column, action.number, interface_transformer)

    def __init__(self, symbol: str, column: int, number: int, interface_transformer: InterfaceTransformer) -> None:
        super().__init__(number=number, interface_transformer=interface_transformer)
        self.symbol = symbol
        self.column = column

    def to_openspiel(self) -> OpenSpielAction:
        return OpenSpielAction(self.number, f"{self.symbol}{self.column}")


class ConnectFourObservation(InterfaceObservation):
    @staticmethod
    def _format_cell(cell: str) -> str:
        if cell == "x":
            return "X"
        if cell == "o":
            return "O"
        return "."

    @staticmethod
    def _state_from_openspiel(state: str) -> list[list[str]]:
        # OpenSpiel Connect Four adds an extra empty row; drop empty lines
        rows = [x for x in state.split("\n") if x.strip()]
        return [[ConnectFourObservation._format_cell(cell) for cell in row] for row in rows]

    @staticmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> ConnectFourObservation:
        state = ConnectFourObservation._state_from_openspiel(observation.state)
        i_actions = [ConnectFourAction.from_openspiel(action, interface_transformer) for action in observation.i_actions]
        o_actions = [ConnectFourAction.from_openspiel(action, interface_transformer) for action in observation.o_actions]
        return ConnectFourObservation(observation, state, i_actions, o_actions, interface_transformer)

    def __init__(
        self,
        os_observation: OpenSpielObservation,
        state: list[list[str]],
        i_actions: list[ConnectFourAction],
        o_actions: list[ConnectFourAction],
        interface_transformer: InterfaceTransformer,
    ) -> None:
        super().__init__(os_observation, i_actions, o_actions, interface_transformer)
        self.state = state


class ConnectFourTransformer(InterfaceTransformer[ConnectFourAction, ConnectFourObservation, []]):
    printer = GridPrinter(row_header=GridAxisLabel.NONE, col_header=GridAxisLabel.NUMBERS)

    def _inner_llm_action(self, action: ConnectFourAction) -> str:
        return f"C{action.column + 1}"

    def display_action(self, action: ConnectFourAction) -> str:
        return f"{action.symbol}{action.column + 1}"

    def _inner_llm_state(self, observation: ConnectFourObservation) -> str:
        return "\n".join("".join(row) for row in observation.state)

    def _inner_llm_partial_states(self, observation: ConnectFourObservation) -> list[str]:
        return []

    def _inner_llm_positions(self, observation: ConnectFourObservation) -> tuple[list[str], list[str]]:
        i_sign, o_sign = ("X", "O") if observation.player_order.is_first() else ("O", "X")
        return grid_to_positions(list(reversed(observation.state)), i_sign, o_sign, lambda r, c: f"C{c}R{r}")

    def _inner_display_state(self, observation: ConnectFourObservation) -> str:
        return ConnectFourTransformer.printer(observation.state)

    def reset(self) -> None:
        pass

    def set_state(self) -> None:
        pass

    def get_other_params(self) -> dict[str, Any] | None:
        return None


CONNECT_FOUR_HEAD_PROMPT = """
Connect 4 is a two-player connection board game, where the players choose a color and then take turns dropping colored discs into a vertically suspended grid.
The pieces fall straight down, occupying the next available space within the column.
The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs.
You are a gaming agent who aims to beat me in Connect 4 games.
Each move is represented by a string consisting of one part: the column (C).
For instance, C1 means the first column.
"""


class ConnectFourPromptAdapter(PromptAdapter[[]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=CONNECT_FOUR_HEAD_PROMPT, use_partial_state=False, order_actions=False)
        self.head_prompt = self.head_prompt_template

    def action_format(self) -> str:
        return "<Cx>, e.g., <C1>, <C3>"

    def restart_prompt(self) -> None:
        pass


class ConnectFourEngine(TurnBasedEngine):
    def __init__(self, game_config: GameConfig) -> None:
        super().__init__(game_config, ConnectFourGame(), ConnectFourTransformer(), ConnectFourPromptAdapter(), ConnectFourAction, ConnectFourObservation)

    def reset(self) -> None:
        self.game.reset()
