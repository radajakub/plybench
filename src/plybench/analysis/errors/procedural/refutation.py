"""How hard a blunder was to see. `deep_error` says only that no shallow class fits it; these two numbers
say how deep "deep" was and how much of the tree agreed, which is what the residual was hiding.

Depth is measured against the taxonomy rather than instead of it: a `self_destruct` has to come out at one
ply and a `missed_block` at two, so the same number that grades the residual also checks the detectors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pyspiel as sp

from plybench.analysis.errors.procedural.position import MovePosition
from plybench.core.game import TurnBasedState
from plybench.core.minimax import DepthMemo, depth_limited_value


@dataclass(frozen=True, slots=True)
class Refutation:
    depth: int  # plies of lookahead the move survives, plus one: the search that first rejects it for good
    n_better: int  # legal alternatives the solver values above the move played
    n_refuting: int  # replies that hold the move to its punished value
    n_replies: int  # replies the opponent had at all; zero when the move ended the game itself

    @property
    def width(self) -> float | None:
        return self.n_refuting / self.n_replies if self.n_replies else None

    def to_dict(self) -> dict[str, Any]:
        return {"depth": self.depth, "n_better": self.n_better, "n_refuting": self.n_refuting, "n_replies": self.n_replies, "width": self.width}


class RefutationSolver:
    def __init__(self) -> None:
        self._memo: DepthMemo = {}

    def __call__(self, position: MovePosition) -> Refutation | None:
        if position.optimal:
            return None  # the solver agreed with the move: there is nothing to refute
        n_refuting, n_replies = self._replies(position)
        return Refutation(self._depth(position), self._n_better(position), n_refuting, n_replies)

    def _value(self, state: sp.State, depth: int, player: int) -> float:
        value = depth_limited_value(state, depth, self._memo)
        return value if player == 0 else -value

    def _horizon(self, position: MovePosition) -> int:
        return position.state.get_game().max_game_length()

    def _separated(self, position: MovePosition, depth: int) -> bool:
        played = self._value(position.after, depth - 1, position.player)
        best = max(self._value(position.child(action), depth - 1, position.player) for action in position.verdict.A())
        return played < best

    def _depth(self, position: MovePosition) -> int:
        # the deepest search that still fails to separate, plus one -- not the shallowest that succeeds.
        # Scoring cut-off leaves as 0 makes the ranking non-monotone in depth, so a search can reject the
        # move for the wrong reason and a deeper one take it back; this reports the depth from which the
        # rejection holds for good. The exact value always separates, so the sweep is bounded by the horizon
        horizon = self._horizon(position)
        unstable = max((depth for depth in range(1, horizon + 1) if not self._separated(position, depth)), default=0)
        return unstable + 1

    def _n_better(self, position: MovePosition) -> int:
        chosen = position.verdict.Q(position.chosen)
        return sum(position.verdict.Q(action) > chosen for action in position.state.legal_actions())

    def _replies(self, position: MovePosition) -> tuple[int, int]:
        after = position.after
        if TurnBasedState.is_terminal(after):
            return 0, 0
        # exact values, so `refuting` is every reply that holds the position below what it was worth before
        # the move -- not just the opponent's optimal set, which would count only the sharpest of them
        horizon, before = self._horizon(position), position.verdict.V()
        replies = after.legal_actions()
        refuting = sum(self._value(TurnBasedState.child(after, reply), horizon, position.player) < before for reply in replies)
        return refuting, len(replies)


@lru_cache(maxsize=None)  # one solver per game config, for the same reason ReplayerCache holds one replayer
def solver_for(game_config: str) -> RefutationSolver:
    return RefutationSolver()
