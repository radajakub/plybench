"""One analysed move and the coordinates that address it. Every stage of the error analysis reads
`TracedMove`, so this is the root of `errors/`: the funnel produces them, and procedural detection,
consistency and the reasoning codebook all consume them without depending on each other."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from plybench.analysis.recognition import step_reasoning_trace
from plybench.analysis.replay import ReplayedStep
from plybench.analysis.stats.moves import MoveRecord
from plybench.trackers.result_tracker import ResultTracker
from plybench.utils.enums import ExtendedEnum


class FunnelStage(ExtendedEnum):
    NON_REASONING = "non_reasoning"  # no reasoning trace recorded -> nothing to analyse
    FAILED = "failed"  # illegal or malformed move -> stored, not analysed
    NON_DECISION = "non_decision"  # every legal move optimal -> no choice could shift the outcome
    OPTIMAL = "optimal"  # a real choice existed and the player took an optimal move
    SUBOPTIMAL = "suboptimal"  # a real choice existed and the player did not take an optimal move


class OutputFailure(ExtendedEnum):
    MALFORMED = "malformed_output"
    ILLEGAL = "illegal_action"
    EMPTY = "empty_output"
    PROVIDER = "provider_failure"
    UNKNOWN = "unknown_failure"


@dataclass(frozen=True, slots=True)
class MatchupId:
    experiment: str
    game: str  # GameConfig.to_string()
    player: str  # PlayerConfig.to_string()
    opponent: str  # PlayerConfig.to_string()

    @classmethod
    def from_tracker(cls, tracker: ResultTracker) -> MatchupId:
        return cls(tracker.experiment, tracker.game.to_string(), tracker.i.to_string(), tracker.o.to_string())


@dataclass(frozen=True, slots=True)
class TracedMove:
    matchup: MatchupId
    game_round: int
    seq: int
    record: MoveRecord
    trace: str | None
    observation: str  # the rendered position the model was shown
    move: str  # the recorded move string, or "FAIL: <reason>"
    legal_moves: tuple[str, ...]
    optimal_moves: tuple[str, ...]
    # the recorded position, kept so an analysis can rebuild the solved tree for this one move without
    # replaying the whole game again
    serialized_state: str = ""

    @property
    def uid(self) -> str:
        matchup = self.matchup
        parts = (matchup.experiment, matchup.game, matchup.player, matchup.opponent, str(self.game_round), str(self.seq))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    @property
    def game_uid(self) -> str:
        """The recorded game this move belongs to. The induction hold-out is game-level, not move-level:
        adjacent positions within one game are near-identical, so a move from a game induction has read is
        not an induction-naive move however unseen that particular position is. Deliberately not a prefix
        of `uid` -- `uid` is the key of every stored annotation and must never move."""
        matchup = self.matchup
        parts = (matchup.experiment, matchup.game, matchup.player, matchup.opponent, str(self.game_round))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    @property
    def failed(self) -> bool:
        return self.move.startswith("FAIL")

    @property
    def output_failure(self) -> OutputFailure | None:
        """A reproducible output-layer diagnosis, separate from strategic move quality."""
        if not self.failed:
            return None
        reason = self.move.casefold()
        if "wrong action format" in reason or "malformed" in reason or "parse" in reason:
            return OutputFailure.MALFORMED
        if "illegal action" in reason:
            return OutputFailure.ILLEGAL
        if "empty" in reason or "no output" in reason:
            return OutputFailure.EMPTY
        if "provider" in reason or "timeout" in reason or "api" in reason:
            return OutputFailure.PROVIDER
        return OutputFailure.UNKNOWN

    @property
    def has_trace(self) -> bool:
        return bool(self.trace)

    @property
    def decision(self) -> FunnelStage:
        # a failed move chose nothing, so the position it was played in cannot grade the choice
        if self.failed:
            return FunnelStage.FAILED
        if self.record.state_class.is_forced:
            return FunnelStage.NON_DECISION
        return FunnelStage.OPTIMAL if self.record.is_optimal else FunnelStage.SUBOPTIMAL

    @property
    def stage(self) -> FunnelStage:
        # no trace -> filtered out first, whatever position the move was played in
        return self.decision if self.has_trace else FunnelStage.NON_REASONING

    @staticmethod
    def from_replayed(replayed: ReplayedStep, matchup: MatchupId, game_key: str, game_round: int) -> TracedMove:
        return TracedMove(
            matchup=matchup,
            game_round=game_round,
            seq=replayed.step.seq,
            record=MoveRecord.from_replayed(replayed, game_key),
            trace=step_reasoning_trace(replayed.step),
            observation=replayed.step.observation,
            move=replayed.step.move,
            legal_moves=tuple(replayed.legal_moves),
            optimal_moves=tuple(replayed.optimal_moves),
            serialized_state=replayed.step.serialized_state,
        )


def by_stage(moves: Sequence[TracedMove]) -> dict[FunnelStage, list[TracedMove]]:
    return group_by(moves, lambda move: move.stage)


def group_by[T, K](items: Sequence[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    groups: dict[K, list[T]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups


def joined[T](moves: Sequence[TracedMove], records: Mapping[str, T]) -> list[tuple[TracedMove, T]]:
    return [(move, records[move.uid]) for move in moves if move.uid in records]
