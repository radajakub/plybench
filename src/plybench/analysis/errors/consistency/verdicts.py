from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from plybench.analysis.errors.judge.store import VerdictStore
from plybench.analysis.errors.moves import TracedMove
from plybench.common.paths import ReasoningPathBuilder
from plybench.utils.enums import ExtendedEnum


class TraceConclusion(BaseModel):
    status: Literal["decided", "no_conclusion", "ambiguous"] = Field(description="whether the trace settles on exactly one legal move")
    move: str = Field(description="the chosen move, verbatim from the legal-moves list; empty unless status is 'decided'")
    evidence: str = Field(description="verbatim quote from the trace stating the decision; empty unless status is 'decided'")


class ConsistencyVerdict(ExtendedEnum):
    CONSISTENT = "consistent"  # the trace's conclusion is the move that was played
    INCONSISTENT = "inconsistent"  # the trace concluded one legal move and a different one was played
    NO_CONCLUSION = "no_conclusion"  # the trace never commits to a move -> nothing to compare against
    AMBIGUOUS = "ambiguous"  # the trace leaves several candidates live -> nothing to compare against
    UNMATCHED = "unmatched"  # the judge named something that is not a legal move -> a judge failure
    INVALID_EVIDENCE = "invalid_evidence"  # decided, but supplied no verbatim support from the trace


# the two verdicts that actually compare a conclusion with an action; the rest carry no evidence either way
GRADABLE_VERDICTS: tuple[ConsistencyVerdict, ...] = (ConsistencyVerdict.CONSISTENT, ConsistencyVerdict.INCONSISTENT)


class SlipKind(ExtendedEnum):
    """What an inconsistency cost. A trace that reasons its way to the right move and then plays another
    is a different failure from one that was going to be wrong anyway, and the two must not be pooled."""

    COSTLY = "costly"  # concluded an optimal move, played a non-optimal one
    LUCKY = "lucky"  # concluded a non-optimal move, played an optimal one
    HARMLESS = "harmless"  # both moves optimal (always the case where no decision was available)
    MOOT = "moot"  # neither move optimal


def _normalized(move: str) -> str:
    return move.strip().strip("<>").strip().casefold()


def match_legal_move(named: str, legal_moves: Sequence[str]) -> str | None:
    """Resolve what the judge wrote to one of the recorded legal-move strings. Matching on the normalised
    form keeps a judge that dropped the angle brackets or changed the case from being counted as a slip."""
    if not named.strip():
        return None
    return next((legal for legal in legal_moves if _normalized(legal) == _normalized(named)), None)


def verdict_for(move: TracedMove, conclusion: TraceConclusion) -> tuple[ConsistencyVerdict, str | None]:
    if conclusion.status == "no_conclusion":
        return ConsistencyVerdict.NO_CONCLUSION, None
    if conclusion.status == "ambiguous":
        return ConsistencyVerdict.AMBIGUOUS, None
    concluded = match_legal_move(conclusion.move, move.legal_moves)
    if concluded is None:
        return ConsistencyVerdict.UNMATCHED, None
    # Local import avoids making the annotation and consistency data models depend on each other at load.
    from plybench.analysis.errors.reasoning.annotation import quotes_trace

    # This pass asks for a verbatim decision quote. Fuzzy overlap is unsafe here: the shared prefix in
    # "I will play A1/A2" is enough to turn a hallucinated move into a costly slip.
    if not quotes_trace(move.trace, conclusion.evidence):
        return ConsistencyVerdict.INVALID_EVIDENCE, None
    verdict = ConsistencyVerdict.CONSISTENT if concluded == move.move else ConsistencyVerdict.INCONSISTENT
    return verdict, concluded


def slip_kind(move: TracedMove, concluded_move: str) -> SlipKind:
    concluded_optimal, played_optimal = concluded_move in move.optimal_moves, move.move in move.optimal_moves
    if concluded_optimal and played_optimal:
        return SlipKind.HARMLESS
    if concluded_optimal:
        return SlipKind.COSTLY
    return SlipKind.LUCKY if played_optimal else SlipKind.MOOT


@dataclass(frozen=True, slots=True)
class ConsistencyRecord:
    """One judge's reading of one trace. The concluded move is stored, not just the verdict, so the
    optimality of what the trace wanted can be re-derived later without re-running the judge."""

    move_uid: str
    annotator: str
    verdict: ConsistencyVerdict
    executed_move: str
    concluded_move: str | None = None
    evidence: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.move_uid, self.annotator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "move_uid": self.move_uid,
            "annotator": self.annotator,
            "verdict": self.verdict.value,
            "executed_move": self.executed_move,
            "concluded_move": self.concluded_move,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsistencyRecord:
        verdict = ConsistencyVerdict.from_value(data["verdict"])
        if not isinstance(verdict, ConsistencyVerdict):
            raise ValueError(f"Unknown consistency verdict: {data['verdict']}")
        return cls(
            move_uid=data["move_uid"],
            annotator=data["annotator"],
            verdict=verdict,
            executed_move=data["executed_move"],
            concluded_move=data.get("concluded_move"),
            evidence=data.get("evidence", ""),
        )


class ConsistencyStore(VerdictStore[ConsistencyRecord]):
    def __init__(self, experiment: str, path: Path | None = None) -> None:
        self.experiment = experiment
        super().__init__(path if path is not None else ReasoningPathBuilder().consistency(experiment), ConsistencyRecord.from_dict)
