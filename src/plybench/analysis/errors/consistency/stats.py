from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from plybench.analysis.errors.consistency.verdicts import GRADABLE_VERDICTS, ConsistencyRecord, ConsistencyVerdict, SlipKind, slip_kind
from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.moves import FunnelStage, TracedMove, group_by, joined
from plybench.analysis.errors.stores import AnalysisStores
from plybench.analysis.statistics.bundle import CIBundle, rate


def _counter(verdicts: list[ConsistencyVerdict]) -> dict[ConsistencyVerdict, int]:
    return {verdict: verdicts.count(verdict) for verdict in ConsistencyVerdict}


@dataclass(frozen=True)
class ConsistencyReport:
    """Trace-vs-action for one (game, player) cell, or for a pool of them."""

    scope: Scope
    annotator: str  # one report per judge: a second judge on the same moves is a reliability check, not more sample
    n_moves: int  # analysable moves in the cell -- what could have been judged
    n_scored: int  # of those, the ones carrying a verdict
    counts: dict[ConsistencyVerdict, int]
    inconsistency: CIBundle  # over gradable verdicts only: a trace that never decided cannot disagree
    by_outcome: dict[FunnelStage, dict[ConsistencyVerdict, int]]
    slips: dict[SlipKind, int]  # what the inconsistencies cost, by optimality of concluded vs played

    @property
    def coverage(self) -> float:
        """Share of analysable moves carrying a verdict. Under `--per-stratum` most of the shortfall is
        moves that were never sampled, not moves the judge failed on -- the store keeps no trace of the
        difference, so the per-run `RunStats.n_failed` is the only real failure count."""
        return self.n_scored / self.n_moves if self.n_moves else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope.to_dict(),
            "annotator": self.annotator,
            "n_moves": self.n_moves,
            "n_scored": self.n_scored,
            "coverage": self.coverage,
            "counts": {verdict.value: count for verdict, count in self.counts.items()},
            "inconsistency": self.inconsistency.to_dict(),
            "by_outcome": {outcome.value: {verdict.value: count for verdict, count in counts.items()} for outcome, counts in self.by_outcome.items()},
            "slips": {kind.value: count for kind, count in self.slips.items()},
        }


def _inconsistency(verdicts: list[ConsistencyVerdict], confidence: float) -> CIBundle:
    gradable = [verdict for verdict in verdicts if verdict in GRADABLE_VERDICTS]
    return rate([verdict == ConsistencyVerdict.INCONSISTENT for verdict in gradable], confidence)


def _slips(scored: list[tuple[TracedMove, ConsistencyRecord]]) -> dict[SlipKind, int]:
    kinds = [slip_kind(move, record.concluded_move) for move, record in scored if record.verdict == ConsistencyVerdict.INCONSISTENT and record.concluded_move is not None]
    return {kind: kinds.count(kind) for kind in SlipKind}


def consistency_report(scope: Scope, moves: Sequence[TracedMove], records: Mapping[str, ConsistencyRecord], annotator: str, confidence: float = 0.95) -> ConsistencyReport:
    """Takes the moves rather than the cell, so pooling several cells is the same call with their moves
    concatenated -- pooled over the moves, never over the cells' own rates."""
    scored = joined(moves, records)
    verdicts = [record.verdict for _, record in scored]

    # keyed in funnel order and only where the bucket is occupied: no cell has failed moves among the
    # analysable ones, so iterating the enum blindly would emit a row that can never be filled
    grouped = group_by(scored, lambda pair: pair[0].decision)
    by_outcome = {stage: _counter([record.verdict for _, record in group]) for stage in FunnelStage if (group := grouped.get(stage))}

    return ConsistencyReport(
        scope=scope,
        annotator=annotator,
        n_moves=len(moves),
        n_scored=len(scored),
        counts=_counter(verdicts),
        inconsistency=_inconsistency(verdicts, confidence),
        by_outcome=by_outcome,
        slips=_slips(scored),
    )


def consistency_reports(funnel: FunnelResult, stores: AnalysisStores) -> list[ConsistencyReport]:
    """One report per annotator that has verdicts for this cell. Annotators are kept apart rather than
    pooled: a second judge on the same moves is the reliability check, not extra sample size."""
    store, scope = stores.consistency(funnel.experiment), Scope.of(funnel)
    reports = [consistency_report(scope, funnel.analyzable, store.by_move(annotator), annotator) for annotator in store.annotators()]
    return [report for report in reports if report.n_scored]
