from __future__ import annotations

from typing import Any

from plybench.configs.game_config import GameConfig
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.utils.text import char_to_number_lower, number_to_char_lower
from plybench.visualization.grid import GridAxisDirection, GridAxisLabel, GridPrinter, grid_to_positions


class BreakthroughGame(TurnBasedGame):
    def __init__(self) -> None:
        super().__init__(game_type="breakthrough", game_name="breakthrough", params={"rows": 8, "columns": 3})


class BreakthroughAction(InterfaceAction):
    @staticmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> BreakthroughAction:
        if len(action.string) < 4 or len(action.string) > 5:
            raise ValueError(f"Invalid action string format: {action.string}")

        startcol, startrow, endcol, endrow = action.string[:4]
        taking = action.string[4] == "*" if len(action.string) == 5 else False

        return BreakthroughAction(
            action.number,
            char_to_number_lower(startcol),
            int(startrow) - 1,
            char_to_number_lower(endcol),
            int(endrow) - 1,
            taking,
            interface_transformer,
        )

    def __init__(self, number: int, start_col: int, start_row: int, end_col: int, end_row: int, taking: bool, interface_transformer: InterfaceTransformer) -> None:
        super().__init__(number=number, interface_transformer=interface_transformer)
        self.start_col = start_col
        self.start_row = start_row
        self.end_col = end_col
        self.end_row = end_row
        self.taking = taking

    def to_openspiel(self) -> OpenSpielAction:
        startcol, startrow = number_to_char_lower(self.start_col), str(self.start_row + 1)
        endcol, endrow = number_to_char_lower(self.end_col), str(self.end_row + 1)
        return OpenSpielAction(self.number, f"{startcol}{startrow}{endcol}{endrow}{'*' if self.taking else ''}")


class BreakthroughObservation(InterfaceObservation):
    @staticmethod
    def _format_cell(cell: str) -> str:
        if cell == "b":
            return "B"
        if cell == "w":
            return "W"
        return "."

    @staticmethod
    def _state_from_openspiel(state: str) -> list[list[str]]:
        rows = [row for row in state.split("\n") if row.strip() != ""]
        return [[BreakthroughObservation._format_cell(cell) for cell in row[1:]] for row in rows[:-1]][::-1]

    @staticmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> BreakthroughObservation:
        state = BreakthroughObservation._state_from_openspiel(observation.state)
        i_actions = [BreakthroughAction.from_openspiel(action, interface_transformer) for action in observation.i_actions]
        o_actions = [BreakthroughAction.from_openspiel(action, interface_transformer) for action in observation.o_actions]
        return BreakthroughObservation(observation, state, i_actions, o_actions, interface_transformer)

    def __init__(
        self,
        os_observation: OpenSpielObservation,
        state: list[list[str]],
        i_actions: list[BreakthroughAction],
        o_actions: list[BreakthroughAction],
        interface_transformer: InterfaceTransformer,
    ) -> None:
        super().__init__(os_observation, i_actions, o_actions, interface_transformer)
        self.state = state


class BreakthroughTransformer(InterfaceTransformer[BreakthroughAction, BreakthroughObservation, []]):
    printer = GridPrinter(
        row_header=GridAxisLabel.NUMBERS,
        row_direction=GridAxisDirection.REVERSED,
        col_header=GridAxisLabel.LETTERS,
    )

    def _inner_llm_action(self, action: BreakthroughAction) -> str:
        startcol, startrow = number_to_char_lower(action.start_col), str(action.start_row + 1)
        endcol, endrow = number_to_char_lower(action.end_col), str(action.end_row + 1)
        return f"{startcol}{startrow}->{endcol}{endrow}{'*' if action.taking else ''}"

    def display_action(self, action: BreakthroughAction) -> str:
        startcol, startrow = number_to_char_lower(action.start_col), str(action.start_row + 1)
        endcol, endrow = number_to_char_lower(action.end_col), str(action.end_row + 1)
        return f"{startcol}{startrow}->{endcol}{endrow}{'*' if action.taking else ''}"

    def _inner_llm_state(self, observation: BreakthroughObservation) -> str:
        r = len(observation.state)
        c = len(observation.state[0])
        state = [[str(r - i)] + row for i, row in enumerate(observation.state[::-1])] + [[" "] + [number_to_char_lower(i) for i in range(c)]]
        return "\n".join("".join(row) for row in state).replace("W", "w").replace("B", "b")

    def _inner_llm_partial_states(self, observation: BreakthroughObservation) -> list[str]:
        return []

    def _inner_llm_positions(self, observation: BreakthroughObservation) -> tuple[list[str], list[str]]:
        i_sign, o_sign = ("B", "W") if observation.player_order.is_first() else ("W", "B")
        return grid_to_positions(observation.state, i_sign, o_sign, lambda r, c: f"{number_to_char_lower(c - 1)}{r}")

    def _inner_display_state(self, observation: BreakthroughObservation) -> str:
        return BreakthroughTransformer.printer(observation.state[::-1])

    def reset(self) -> None:
        pass

    def set_state(self) -> None:
        pass

    def get_other_params(self) -> dict[str, Any] | None:
        return None


BREAKTHROUGH_HEAD_PROMPT = """
Breakthrough is a two-player game played on a rectangular board.
Players take turns moving their pieces, which can move one space straight or diagonally forward if the target square is empty.
A piece can also move diagonally forward to capture an opponent's piece.
Capturing is optional, and a player can only capture one piece per turn.
The goal is to be the first to reach the opponent's home row, the farthest row from the player.
If all of a player's pieces are captured, they lose.
The game does not allow draws, as pieces can only move forward or be captured.
The Breakthrough board is identified by columns labeled starting from A (from left to right) and rows numbered 1 to 8 (from bottom to top).
The intersection of a column and a row specifies a unique square on the board.
"""


class BreakthroughPromptAdapter(PromptAdapter[[]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=BREAKTHROUGH_HEAD_PROMPT, use_partial_state=False, position_name="pieces", order_actions=False)
        self.head_prompt = self.head_prompt_template

    def action_format(self) -> str:
        return "<[a-c][1-8]->[a-c][1-8]>, e.g., <a7->a6>"

    def restart_prompt(self) -> None:
        pass


class BreakthroughEngine(TurnBasedEngine):
    def __init__(self, game_config: GameConfig) -> None:
        super().__init__(game_config, BreakthroughGame(), BreakthroughTransformer(), BreakthroughPromptAdapter(), BreakthroughAction, BreakthroughObservation)

    def reset(self) -> None:
        self.game.reset()
