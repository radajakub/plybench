"""Prevalence of the procedural move labels over one (game, player) cell. This is the ground truth the LLM
passes are later read against, and the one report that needs no judge and no money.

Rates are over every graded move, not over the blunders: a diagnostic class can fire on a move the solver
calls optimal -- a won position finished the slow way, a lost one given up on early -- so a denominator of
suboptimal moves would be a denominator the numerator does not sit inside. The one exception is
`explained`, which is a statement about the blunders and says so.

The refutation columns are the other way round: only a blunder has a refutation, so depth and width are
always over the moves the solver disagreed with, and a label's depth is over the blunders it fired on."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.moves import TracedMove
from plybench.analysis.errors.procedural.detection import MoveDiagnosis, MoveLabel, label_kind
from plybench.analysis.errors.procedural.refutation import Refutation
from plybench.analysis.statistics.bundle import CIBundle, mean_bundle, rate
from plybench.analysis.statistics.distribution import Distribution
from plybench.common.enums import StateClass

Graded = Sequence[tuple[TracedMove, MoveDiagnosis]]


def _mean(values: Iterable[float], confidence: float) -> CIBundle:
    return mean_bundle(Distribution.from_values(values), confidence)


@dataclass(frozen=True)
class LabelPrevalence:
    label: MoveLabel
    n: int
    rate: CIBundle
    by_class: dict[StateClass, int]
    # mean lookahead the blunders carrying this label needed. A detector names a shallow pattern, so this
    # is what says whether it really caught one: `self_destruct` at one ply, `missed_block` at two
    depth: CIBundle

    @property
    def kind(self) -> str:
        return label_kind(self.label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "n": self.n,
            "rate": self.rate.to_dict(),
            "kind": self.kind,
            "by_state_class": {state.value: count for state, count in self.by_class.items()},
            "depth": self.depth.to_dict(),
        }


@dataclass(frozen=True)
class LabelReport:
    scope: Scope
    n_moves: int  # graded moves: every move that put an action on the board, failed ones excluded
    n_suboptimal: int  # of those, the ones the solver disagreed with -- the denominator of `explained`
    prevalence: list[LabelPrevalence]  # only labels that occurred, most frequent first
    explained: CIBundle  # share of suboptimal moves a diagnostic class accounts for
    depth: CIBundle  # mean refutation depth over the blunders
    width: CIBundle  # mean share of opponent replies that punish; a narrow refutation is a harder one
    depths: dict[int, int]  # blunders per refutation depth, sparse: only the depths that occurred
    # the same histogram over the residual alone, which is what says whether it is deep or just
    # unwritten: a residual sitting at the shallow end is a detector nobody has added yet
    residual_depths: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope.to_dict(),
            "n_moves": self.n_moves,
            "n_suboptimal": self.n_suboptimal,
            "explained": self.explained.to_dict(),
            "depth": self.depth.to_dict(),
            "width": self.width.to_dict(),
            "depths": {str(depth): count for depth, count in sorted(self.depths.items())},
            "residual_depths": {str(depth): count for depth, count in sorted(self.residual_depths.items())},
            "prevalence": [row.to_dict() for row in self.prevalence],
        }


def _histogram(diagnoses: Iterable[MoveDiagnosis]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for diagnosis in diagnoses:
        if diagnosis.refutation is not None:
            counts[diagnosis.refutation.depth] = counts.get(diagnosis.refutation.depth, 0) + 1
    return counts


def _prevalence(label: MoveLabel, graded: Graded, confidence: float) -> LabelPrevalence:
    hits = [label in diagnosis.labels for _, diagnosis in graded]
    by_class = dict.fromkeys(StateClass, 0)  # every class keyed, empty ones included: the report is a fixed table
    depths: list[int] = []
    for move, diagnosis in graded:
        if label not in diagnosis.labels:
            continue
        by_class[move.record.state_class] += 1
        if diagnosis.refutation is not None:
            depths.append(diagnosis.refutation.depth)
    return LabelPrevalence(label, sum(hits), rate(hits, confidence), by_class, _mean(depths, confidence))


def label_report(scope: Scope, graded: Graded, confidence: float = 0.95) -> LabelReport:
    prevalences = [row for label in MoveLabel if (row := _prevalence(label, graded, confidence)).n]
    # a lost or forced position has no suboptimal move to offer, so this is the decision bucket by construction
    suboptimal = [diagnosis for move, diagnosis in graded if not move.record.is_optimal]
    refutations: list[Refutation] = [diagnosis.refutation for diagnosis in suboptimal if diagnosis.refutation is not None]
    # a move that ended the game itself left no reply to punish it, so it has no width to average
    widths = [width for refutation in refutations if (width := refutation.width) is not None]

    residual = [diagnosis for diagnosis in suboptimal if MoveLabel.DEEP_ERROR in diagnosis.labels]

    return LabelReport(
        scope=scope,
        n_moves=len(graded),
        n_suboptimal=len(suboptimal),
        prevalence=sorted(prevalences, key=lambda row: row.n, reverse=True),
        explained=rate([MoveLabel.DEEP_ERROR not in diagnosis.labels for diagnosis in suboptimal], confidence),
        depth=_mean([refutation.depth for refutation in refutations], confidence),
        width=_mean(widths, confidence),
        depths=_histogram(suboptimal),
        residual_depths=_histogram(residual),
    )
