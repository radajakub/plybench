from __future__ import annotations

from plybench.utils.enums import ExtendedEnum

# how a failed move starts in the results; see MoveFailure
_FAIL_PREFIX = "FAIL: "


class PlayerOrder(ExtendedEnum):
    FIRST = "first"
    SECOND = "second"

    @staticmethod
    def from_int(player: int) -> PlayerOrder:
        if player == 0:
            return PlayerOrder.FIRST
        elif player == 1:
            return PlayerOrder.SECOND
        raise ValueError("Player has to be either 0 or 1")

    def is_first(self) -> bool:
        return self == PlayerOrder.FIRST


class ObservationType(ExtendedEnum):
    ACTIONS = "actions"
    STATE = "state"


class OutputStrategies(ExtendedEnum):
    TEXT = "text"
    STRUCTURED = "structured"


class MoveFailure(ExtendedEnum):
    # why a player gave no playable move. A failed move is recorded as "FAIL: <value> (<detail>)", the
    # same text results held before this enum, so older results parse too
    MALFORMED = "Wrong action format"
    ILLEGAL = "Illegal action selected"
    REFUSAL = "Refusal"
    PROMPT_BLOCKED = "Prompt blocked"
    # a player returned no action and no reason (a bot with no legal move to choose); recorded as
    # "FAIL: None" before this enum
    NO_ACTION = "No action"

    def to_move(self, detail: str) -> str:
        return f"{_FAIL_PREFIX}{self.value} ({detail})"

    @classmethod
    def from_move(cls, move: str) -> MoveFailure | None:
        # the failure a recorded move holds; None for a played move or a failure no member labels
        if not move.startswith(_FAIL_PREFIX):
            return None
        reason = move.removeprefix(_FAIL_PREFIX)
        return next((failure for failure in cls if reason.startswith(f"{failure.value} (")), None)


class GameResults(ExtendedEnum):
    WIN = 0
    LOSS = 1
    DRAW = 2
    MY_FAIL = 3
    OPPONENT_FAIL = 4

    def invert(self) -> GameResults:
        return {
            GameResults.WIN: GameResults.LOSS,
            GameResults.LOSS: GameResults.WIN,
            GameResults.DRAW: GameResults.DRAW,
            GameResults.MY_FAIL: GameResults.OPPONENT_FAIL,
            GameResults.OPPONENT_FAIL: GameResults.MY_FAIL,
        }[self]

    def to_reward(self) -> float:
        return {
            GameResults.WIN: 1.0,
            GameResults.DRAW: 0.5,
            GameResults.LOSS: 0.0,
            GameResults.MY_FAIL: -1.0,
            GameResults.OPPONENT_FAIL: 0.0,
        }[self]

    def to_prompt(self) -> str:
        return {
            GameResults.WIN: "WON",
            GameResults.LOSS: "LOST",
            GameResults.DRAW: "DRAW",
            GameResults.MY_FAIL: "LOST (failed to produce a valid move)",
            GameResults.OPPONENT_FAIL: "WON (opponent failed to produce a valid move)",
        }[self]


class StateClass(ExtendedEnum):
    DONT_CARE = "dont_care"  # all legal moves optimal AND state value > LOSS — forced non-losing
    LOST = "lost"  # all legal moves optimal AND state value == LOSS — already lost
    DECISION = "decision"  # optimal set is a strict subset of legal moves — a real choice exists

    @property
    def is_forced(self) -> bool:
        return self in (StateClass.DONT_CARE, StateClass.LOST)


class CIFamily(ExtendedEnum):
    RATIO = "ratio"  # a proportion in [0, 1] -> Wilson interval
    MEAN = "mean"  # a real-valued mean -> SEM / t / bootstrap intervals


class MetricName(ExtendedEnum):
    WIN_RATE = "win_rate"
    DRAW_RATE = "draw_rate"
    LOSS_RATE = "loss_rate"
    FAIL_RATE = "fail_rate"
    SCORE = "score"
    MOVES_PER_GAME = "moves_per_game"
    INPUT_TOKENS_PER_GAME = "input_tokens_per_game"
    OUTPUT_TOKENS_PER_GAME = "output_tokens_per_game"
    INPUT_TOKENS_PER_MOVE = "input_tokens_per_move"
    OUTPUT_TOKENS_PER_MOVE = "output_tokens_per_move"
    # produced only for solvable games via replay (Phase 6b)
    OPTIMALITY_RATE = "optimality_rate"
    OPTIMALITY_RATE_NON_TRIVIAL = "optimality_rate_non_trivial"
    REGRET = "regret"
    # produced only for recognisable games from reasoning traces (Phase 6c)
    RECOGNITION_RATE = "recognition_rate"
