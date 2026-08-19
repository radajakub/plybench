from __future__ import annotations

from dataclasses import dataclass

from plybench.analysis.recognition import recognizable, step_reasoning_trace, trace_mentions_original_game
from plybench.analysis.replay import ReplayedStep
from plybench.common.enums import StateClass


@dataclass(frozen=True)
class MoveRecord:
    state_class: StateClass
    is_optimal: bool
    regret: float
    input_tokens: int | None
    output_tokens: int | None
    recognized: bool | None
    n_legal: int = 0  # branching factor at this state
    n_optimal: int = 0  # size of the solver's optimal-action set at this state

    @staticmethod
    def from_replayed(replayed: ReplayedStep, game_key: str) -> MoveRecord:
        return MoveRecord(
            state_class=replayed.state_class,
            is_optimal=replayed.is_optimal,
            regret=replayed.regret,
            input_tokens=replayed.step.input_tokens,
            output_tokens=replayed.step.output_tokens,
            recognized=_recognized_from_trace(step_reasoning_trace(replayed.step), game_key),
            n_legal=replayed.n_legal,
            n_optimal=replayed.n_optimal,
        )


def _recognized_from_trace(trace: str | None, game_key: str) -> bool | None:
    if not recognizable(game_key) or trace is None:
        return None
    return trace_mentions_original_game(trace, game_key)
