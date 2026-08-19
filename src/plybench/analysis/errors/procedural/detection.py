"""The procedural move labels and the rules that assign them. Every label here is decided by the solved
game tree, never by a model reading a trace: this is the ground truth the LLM passes are later measured
against, so it has to be reproducible and free.

Every move that put an action on the board is labelled, not just the blunders -- the labels cost a shallow
tree expansion each and are what a later stage correlates trace properties against, so a rate is only
readable next to the clean moves it is a rate over.

The universal detectors read the tree and travel to any solvable game; the per-family ones read the
position itself and buy interpretability with that knowledge: "ignored the nim-sum" names a rule a model
either has or has not learnt, where "suboptimal" only says it lost value."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache, reduce

import pyspiel as sp

from plybench.analysis.errors.procedural.position import MovePosition
from plybench.analysis.errors.procedural.refutation import Refutation
from plybench.analysis.recognition import original_game_name, recognizable
from plybench.common.enums import GameResults
from plybench.core.game import TurnBasedState
from plybench.utils.enums import ExtendedEnum


class MoveLabel(ExtendedEnum):
    MISSED_FORCED_WIN = "missed_forced_win"  # a move forced the win -- outright, or by leaving the opponent only losing replies
    MISSED_BLOCK = "missed_block"  # the opponent can now end the game; a move existed that stopped it
    SELF_DESTRUCT = "self_destruct"  # the move itself ended the game as a loss
    MISSED_FORK = "missed_fork"  # a forced win in three was available and not taken
    ALLOWED_FORK = "allowed_fork"  # the opponent can now force a win in three
    NIM_SUM_IGNORED = "nim_sum_ignored"  # a nim-sum-zeroing move existed and was not played
    MISERE_RULE_IGNORED = "misere_rule_ignored"  # the normal-play rule applied inside the misere endgame
    DEEP_ERROR = "deep_error"  # suboptimal with no diagnostic class: the refutation lies deeper
    THREW_WIN = "threw_win"  # severity: a won position is no longer won
    THREW_DRAW = "threw_draw"  # severity: a drawn position is now lost
    CLEAN = "clean"  # nothing fired: no shallow class, and no value lost either


Detector = tuple[MoveLabel, Callable[[MovePosition], bool]]


@dataclass(frozen=True, slots=True)
class MoveDiagnosis:
    """Everything stage 2 has to say about one move. The labels name what went wrong; the refutation says
    how hard it was to see, and is absent exactly when the solver agreed with the move."""

    labels: frozenset[MoveLabel]
    refutation: Refutation | None


# --- universal: the tree alone, so these travel to any solvable game --------------------------------
def _forces_win(position: MovePosition, action: int) -> bool:
    """The action ends the game as a win, or leaves the opponent nothing but replies that do. The second
    case is the only way a misere game is ever won -- there whoever ends the game is the one who loses --
    so without it the class is structurally dead in half the families here, and the misere blunders it
    would have named fall through to the residual instead."""
    successor = position.child(action)
    if successor.is_terminal():
        return position.terminal_outcome(successor) is GameResults.WIN
    return all(position.terminal_outcome(reply) is GameResults.WIN for reply in TurnBasedState.children(successor))


def _missed_forced_win(position: MovePosition) -> bool:
    if _forces_win(position, position.chosen):
        return False
    return any(_forces_win(position, action) for action in position.verdict.A())


def _missed_block(position: MovePosition) -> bool:
    return position.avoidable(lambda state: position.wins_now(state, position.opponent))


def _self_destruct(position: MovePosition) -> bool:
    return position.avoidable(lambda state: position.terminal_outcome(state) is GameResults.LOSS)


def _threw_win(position: MovePosition) -> bool:
    return position.outcome_before is GameResults.WIN and position.outcome_after is not GameResults.WIN


def _threw_draw(position: MovePosition) -> bool:
    return position.outcome_before is GameResults.DRAW and position.outcome_after is GameResults.LOSS


# --- line games (tic-tac-toe and its variants) ------------------------------------------------------
def _creates_forced_win(position: MovePosition, action: int) -> bool:
    successor = position.child(action)
    return position.terminal_outcome(successor) is GameResults.WIN or position.unstoppable(successor, position.player)


def _missed_fork(position: MovePosition) -> bool:
    if _creates_forced_win(position, position.chosen):
        return False
    return any(position.unstoppable(position.child(action), position.player) for action in position.verdict.A())


def _allowed_fork(position: MovePosition) -> bool:
    return position.avoidable(lambda state: any(position.unstoppable(reply, position.opponent) for reply in TurnBasedState.children(state)))


# --- nim games (all four variants here are misere: whoever takes the last match loses) ---------------
def piles(state: sp.State) -> list[int]:
    # pile sizes in game space, read off OpenSpiel's own observation ('(0): 1 3 5 7'): inverse_nim renders
    # the complement to the model, but the rule that decides whether the move was right is the tree's
    return [int(token) for token in state.observation_string(0).split(":")[1].split()]


def nim_sum(sizes: Sequence[int]) -> int:
    return reduce(operator.xor, sizes, 0)


def _endgame(sizes: Sequence[int]) -> bool:
    # with at most one pile still holding two or more matches the winning rule flips from "leave the
    # nim-sum at zero" to "leave an odd number of single piles"; above it, misere and normal play agree
    return sum(size >= 2 for size in sizes) <= 1


def _nim_sum_ignored(position: MovePosition) -> bool:
    before = piles(position.state)
    if _endgame(before) or nim_sum(before) == 0:
        return False
    return nim_sum(piles(position.after)) != 0


def _misere_rule_ignored(position: MovePosition) -> bool:
    return _endgame(piles(position.state)) and position.avoidable(lambda state: nim_sum(piles(state)) == 0)


# --- which detectors apply where --------------------------------------------------------------------
# Order is report order, sharpest class first: a one-move blunder says more than the value drop it caused.
UNIVERSAL: tuple[Detector, ...] = (
    (MoveLabel.MISSED_FORCED_WIN, _missed_forced_win),
    (MoveLabel.MISSED_BLOCK, _missed_block),
    (MoveLabel.SELF_DESTRUCT, _self_destruct),
)

SEVERITY: tuple[Detector, ...] = ((MoveLabel.THREW_WIN, _threw_win), (MoveLabel.THREW_DRAW, _threw_draw))
SEVERITY_LABELS: frozenset[MoveLabel] = frozenset(code for code, _ in SEVERITY)

LINE_DETECTORS: tuple[Detector, ...] = ((MoveLabel.MISSED_FORK, _missed_fork), (MoveLabel.ALLOWED_FORK, _allowed_fork))
NIM_DETECTORS: tuple[Detector, ...] = ((MoveLabel.NIM_SUM_IGNORED, _nim_sum_ignored), (MoveLabel.MISERE_RULE_IGNORED, _misere_rule_ignored))

# Keyed by the game a variant is really a presentation of, which recognition.py already maps out: the
# obfuscated variants exist to hide the same rules behind new framing, so "the same underlying game" and
# "the same detectors apply" are one statement and belong in one table. An unlisted family gets UNIVERSAL.
BY_FAMILY: dict[str, tuple[Detector, ...]] = {"tic tac toe": LINE_DETECTORS, "nim": NIM_DETECTORS}


@lru_cache(maxsize=None)  # resolved once per game rather than once per move: `detect` asks for every move
def diagnostics_for(game_key: str) -> tuple[Detector, ...]:
    family = original_game_name(game_key) if recognizable(game_key) else ""
    return (*UNIVERSAL, *BY_FAMILY.get(family, ()))


def label_kind(label: MoveLabel) -> str:
    if label in SEVERITY_LABELS:
        return "severity"
    return {MoveLabel.DEEP_ERROR: "residual", MoveLabel.CLEAN: "clean"}.get(label, "diagnostic")


def detect(position: MovePosition, game_key: str) -> frozenset[MoveLabel]:
    diagnostic = {code for code, detected in diagnostics_for(game_key) if detected(position)}
    severity = {code for code, detected in SEVERITY if detected(position)}
    if not diagnostic and not position.optimal:
        diagnostic = {MoveLabel.DEEP_ERROR}
    return frozenset((diagnostic | severity) or {MoveLabel.CLEAN})
