"""Bucket sizes and per-bucket move metrics for one cell. Same role as `procedural/stats.py` and
`consistency/stats.py`: turn one stage's records into a single report object that both the printed table
and the JSON row are read off."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.moves import FunnelStage, OutputFailure, TracedMove, by_stage
from plybench.analysis.statistics.bundle import CIBundle
from plybench.analysis.stats.move_metrics import DEFAULT_MOVE_METRICS, MoveMetric
from plybench.common.enums import MetricName, StateClass


def metrics(moves: Sequence[TracedMove], move_metrics: Sequence[MoveMetric] = DEFAULT_MOVE_METRICS, confidence: float = 0.95) -> dict[MetricName, CIBundle]:
    records = [move.record for move in moves]
    return {metric.name: metric.bundle(records, confidence) for metric in move_metrics}


@dataclass(frozen=True)
class StageSummary:
    n_moves: int
    share: float
    metrics: dict[MetricName, CIBundle]

    def to_dict(self) -> dict[str, Any]:
        return {"n_moves": self.n_moves, "share": self.share, "metrics": {name.value: bundle.to_dict() for name, bundle in self.metrics.items()}}


@dataclass(frozen=True)
class FunnelReport:
    scope: Scope
    n_moves: int
    stages: dict[FunnelStage, StageSummary]  # every leaf, empty ones included: the buckets partition the moves
    # already-lost positions are a consequence of the player's own earlier blunders, unlike genuinely forced
    # ones, so a weak player inflates its own non_decision bucket -- None when the bucket is empty, not zero
    lost_share: float | None
    output_failures: dict[OutputFailure, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope.to_dict(),
            "n_moves": self.n_moves,
            "stages": [{"stage": stage.value, **summary.to_dict()} for stage, summary in self.stages.items()],
            "lost_share": self.lost_share,
            "output_failures": {kind.value: count for kind, count in self.output_failures.items()},
        }


def _lost_share(moves: Sequence[TracedMove]) -> float | None:
    return sum(move.record.state_class == StateClass.LOST for move in moves) / len(moves) if moves else None


def _summary(moves: Sequence[TracedMove], total: int, confidence: float) -> StageSummary:
    return StageSummary(len(moves), len(moves) / total if total else 0.0, metrics(moves, confidence=confidence))


def funnel_report(scope: Scope, moves: Sequence[TracedMove], confidence: float = 0.95) -> FunnelReport:
    total, buckets = len(moves), by_stage(moves)
    failures = [move.output_failure for move in moves if move.output_failure is not None]
    return FunnelReport(
        scope=scope,
        n_moves=total,
        stages={stage: _summary(buckets.get(stage, []), total, confidence) for stage in FunnelStage},
        lost_share=_lost_share(buckets.get(FunnelStage.NON_DECISION, [])),
        output_failures={kind: failures.count(kind) for kind in OutputFailure},
    )
