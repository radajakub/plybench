"""One analysed move as the solved tree sees it: the position it was played in, the action taken, and the
minimax verdict on both. Every procedural detector is a question asked of this object, so the counterfactual
("was the hazard avoidable?") and the value-to-outcome reading live here once rather than in each rule."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pyspiel as sp

from plybench.analysis.errors.moves import TracedMove
from plybench.analysis.replay import TurnBasedReplayer
from plybench.common.enums import GameResults
from plybench.core.game import TurnBasedState
from plybench.core.minimax import AVQ


@dataclass(frozen=True, slots=True)
class MovePosition:
    state: sp.State
    chosen: int  # the action number actually played
    verdict: AVQ
    win_value: float
    loss_value: float

    @classmethod
    def from_probe(cls, replayer: TurnBasedReplayer, move: TracedMove) -> MovePosition:
        state, chosen, verdict = replayer.probe(move.serialized_state, move.move)
        loss_value, win_value = replayer.reward_range
        return cls(state, chosen, verdict, win_value, loss_value)

    @property
    def player(self) -> int:
        return self.state.current_player()

    @property
    def opponent(self) -> int:
        return 1 - self.player

    @property
    def optimal(self) -> bool:
        return self.verdict.check_optimal(self.chosen)

    def child(self, action: int) -> sp.State:
        return TurnBasedState.child(self.state, action)

    @property
    def after(self) -> sp.State:
        return self.child(self.chosen)

    def alternatives(self) -> Iterator[sp.State]:
        return (self.child(action) for action in self.state.legal_actions() if action != self.chosen)

    def avoidable(self, hazard: Callable[[sp.State], bool]) -> bool:
        return hazard(self.after) and any(not hazard(state) for state in self.alternatives())

    def outcome(self, value: float) -> GameResults:
        # a solved value against the game's own reward range, not the scoring scale of GameResults.to_reward();
        # the fail members cannot occur here, since a value read off the tree is always one of the three
        if value == self.win_value:
            return GameResults.WIN
        return GameResults.LOSS if value == self.loss_value else GameResults.DRAW

    @property
    def outcome_before(self) -> GameResults:
        return self.outcome(self.verdict.V())

    @property
    def outcome_after(self) -> GameResults:
        return self.outcome(self.verdict.Q(self.chosen))

    def terminal_outcome(self, state: sp.State) -> GameResults | None:
        if not state.is_terminal():
            return None
        return self.outcome(TurnBasedState.get_rewards(state)[self.player])

    def wins_now(self, state: sp.State, player: int) -> bool:
        return any(self.terminal_outcome(successor) is self._target(player) for successor in TurnBasedState.children(state))

    def unstoppable(self, state: sp.State, player: int) -> bool:
        if state.is_terminal():
            return False
        return all(self.wins_now(successor, player) for successor in TurnBasedState.children(state))

    def _target(self, player: int) -> GameResults:
        # outcomes are always read from the analysed player's side, so the opponent winning is our loss
        return GameResults.WIN if player == self.player else GameResults.LOSS
