"""One (game, player) cell with whatever has been computed about it. Stages attach one at a time, and
every report object is built once and reused, so the printed table and the JSON row are the same numbers
rather than two independent computations of them."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from plybench.analysis.errors.consistency import stats as consistency_stats
from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.funnel import stats as funnel_stats
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.moves import joined
from plybench.analysis.errors.procedural import stats as procedural_stats
from plybench.analysis.errors.procedural.detection import MoveDiagnosis
from plybench.analysis.errors.procedural.run import label_moves
from plybench.analysis.errors.reasoning import agreement, correlate
from plybench.analysis.errors.reasoning import stats as reasoning_stats
from plybench.analysis.errors.stores import AnalysisStores
from plybench.analysis.replay import ReplayerCache


@dataclass
class Analysis:
    funnel: FunnelResult
    stores: AnalysisStores
    labels: dict[str, MoveDiagnosis] = field(default_factory=dict)

    @cached_property
    def scope(self) -> Scope:
        return Scope.of(self.funnel)

    @cached_property
    def graded(self) -> procedural_stats.Graded:
        return joined(self.funnel.moves, self.labels)

    @cached_property
    def funnel_report(self) -> funnel_stats.FunnelReport:
        return funnel_stats.funnel_report(self.scope, self.funnel.moves)

    @cached_property
    def label_report(self) -> procedural_stats.LabelReport:
        return procedural_stats.label_report(self.scope, self.graded)

    @cached_property
    def consistency_reports(self) -> list[consistency_stats.ConsistencyReport]:
        return consistency_stats.consistency_reports(self.funnel, self.stores)

    @cached_property
    def prevalence_reports(self) -> list[reasoning_stats.PrevalenceReport]:
        return reasoning_stats.prevalence_reports(self.funnel, self.stores)

    @cached_property
    def reliability_report(self) -> agreement.ReliabilityReport | None:
        return agreement.reliability_report(self.funnel, self.stores)

    @cached_property
    def correlation_reports(self) -> list[correlate.CrosstabReport]:
        """Both joins, one pair per annotator: the codes against the solver's own labels, and against how
        long the trace was. These are the two design questions the separate stages cannot answer alone."""
        store, codebook = self.stores.annotations(self.funnel.experiment), self.stores.codebook(self.funnel.experiment)
        moves = self.funnel.analyzable
        reports = []
        for annotator in store.annotators():
            by_move = store.by_move(annotator)
            reports.append(correlate.code_by_label(self.scope, moves, by_move, self.labels, codebook, annotator))
            reports.append(correlate.code_by_length(self.scope, moves, by_move, codebook, annotator))
        return [report for report in reports if report.n_moves]

    def to_dict(self) -> dict[str, Any]:
        reliability = self.reliability_report
        return {
            **self.funnel_report.to_dict(),
            "labels": self.label_report.to_dict(),
            "consistency": [report.to_dict() for report in self.consistency_reports],
            "mistakes": [report.to_dict() for report in self.prevalence_reports],
            "reliability": reliability.to_dict() if reliability is not None else None,
            "correlations": [report.to_dict() for report in self.correlation_reports],
        }


def analyse(funnel: FunnelResult, stores: AnalysisStores, replayers: ReplayerCache, progress: bool | None = None) -> Analysis:
    return Analysis(funnel, stores, label_moves(funnel, replayers, progress))
