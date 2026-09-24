"""Pooling cells into one row along any combination of facets.

Every report up to here describes a single (game, player) cell, which cannot answer anything about a
*model*: on ttt the per-cell `explained` rates read 0.97-1.00 while the pooled one is 0.73, and both are
true of the same data. Reading a model off the per-cell tables is what that gap punishes.

Pooling is over the moves, never over the cells' own rates: averaging those would weight a cell of 30
moves like one of 1300. That is why the stage report functions take records rather than a cell -- a pooled
row is the same call with several cells' records concatenated, not a second implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from plybench.analysis.errors.analysis import Analysis
from plybench.analysis.errors.consistency.report import report_consistency
from plybench.analysis.errors.consistency.stats import ConsistencyReport, consistency_report
from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.funnel.report import report_funnel
from plybench.analysis.errors.funnel.stats import FunnelReport, funnel_report
from plybench.analysis.errors.moves import FunnelStage, TracedMove
from plybench.analysis.errors.procedural.report import report_labels
from plybench.analysis.errors.procedural.stats import LabelReport, label_report
from plybench.analysis.errors.reasoning.report import report_mistakes
from plybench.analysis.errors.reasoning.stats import PrevalenceReport, prevalence_report
from plybench.analysis.errors.stores import consistency_join
from plybench.analysis.statistics.standardization import Reference, reference_mix


@dataclass(frozen=True)
class PooledReport:
    scope: Scope
    facets: tuple[str, ...]  # what this row was grouped on; everything else was pooled over
    n_cells: int
    funnel: FunnelReport
    labels: LabelReport
    # one report per judge for both: annotators stay apart even pooled, since a second judge on the same
    # moves is the reliability check rather than extra sample
    consistency: list[ConsistencyReport]
    mistakes: list[PrevalenceReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope.to_dict(),
            "facets": list(self.facets),
            "n_cells": self.n_cells,
            "funnel": self.funnel.to_dict(),
            "labels": self.labels.to_dict(),
            "consistency": [report.to_dict() for report in self.consistency],
            "mistakes": [report.to_dict() for report in self.mistakes],
        }


def position_mix(analyses: Sequence[Analysis]) -> Reference[FunnelStage]:
    """The reference mix every standardised rate is reweighted to: how the whole corpus's analysable moves
    split across forced, optimal and suboptimal positions. Taken from the corpus rather than chosen, so a
    standardised rate reads as "what this group would show facing an average board"."""
    counts: dict[FunnelStage, int] = {}
    for analysis in analyses:
        for move in analysis.funnel.analyzable:
            counts[move.decision] = counts.get(move.decision, 0) + 1
    return reference_mix(counts)


def _analysable(members: Sequence[Analysis]) -> list[TracedMove]:
    return [move for analysis in members for move in analysis.funnel.analyzable]


def _consistency(scope: Scope, members: Sequence[Analysis], confidence: float) -> list[ConsistencyReport]:
    # every member of a pool shares an experiment -- `only` always keeps that facet -- so they share one
    # store, and restricting to this pool's own moves is what keeps the row about these cells
    store = members[0].stores.consistency(scope.experiment)
    moves = _analysable(members)
    reports = [consistency_report(scope, moves, store.by_move(annotator), annotator, confidence) for annotator in store.annotators()]
    return [report for report in reports if report.n_scored]


def _mistakes(scope: Scope, members: Sequence[Analysis], confidence: float, reference: Reference[FunnelStage] | None) -> list[PrevalenceReport]:
    stores = members[0].stores
    store, codebook = stores.annotations(scope.experiment), stores.codebook(scope.experiment)
    moves = _analysable(members)
    reports = []
    for annotator in store.annotators():
        consistency, _ = consistency_join(stores, scope.experiment, annotator)
        reports.append(prevalence_report(scope, moves, store.by_move(annotator), codebook, annotator, consistency, confidence, reference))
    return [report for report in reports if report.n_annotated]


def _pooled(scope: Scope, facets: tuple[str, ...], members: Sequence[Analysis], confidence: float, reference: Reference[FunnelStage] | None) -> PooledReport:
    moves = [move for analysis in members for move in analysis.funnel.moves]
    graded = [pair for analysis in members for pair in analysis.graded]
    return PooledReport(
        scope=scope,
        facets=facets,
        n_cells=len(members),
        funnel=funnel_report(scope, moves, confidence),
        labels=label_report(scope, graded, confidence),
        consistency=_consistency(scope, members, confidence),
        mistakes=_mistakes(scope, members, confidence, reference),
    )


def pooled_reports(
    analyses: Sequence[Analysis],
    facets: tuple[str, ...],
    confidence: float = 0.95,
    reference: Reference[FunnelStage] | None = None,
) -> list[PooledReport]:
    """One row per distinct combination of `facets`; cells differing only on facets not named here are
    pooled together. That is the whole point: `("model",)` is one row per model over every game it played,
    `("model", "presentation")` is that same model's behaviour under each obfuscation."""
    groups: dict[Scope, list[Analysis]] = {}
    for analysis in analyses:
        groups.setdefault(analysis.scope.only(*facets), []).append(analysis)
    ordered = sorted(groups.items(), key=lambda item: tuple(value for _, value in item[0].items))
    return [_pooled(scope, facets, members, confidence, reference) for scope, members in ordered]


def report_pooled(report: PooledReport) -> None:
    # the cell reporters, unchanged: a pooled row is the same report object over more moves, so printing it
    # through anything else would be two renderings of one table waiting to disagree
    report_funnel(report.funnel)
    print(f"  pooled over {report.n_cells} cell(s)")
    report_labels(report.labels)
    if report.consistency:
        report_consistency(report.scope, report.funnel.n_moves, report.consistency)
    if report.mistakes:
        report_mistakes(report.scope, report.funnel.n_moves, report.mistakes)
