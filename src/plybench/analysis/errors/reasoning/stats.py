from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from plybench.analysis.errors.consistency.verdicts import ConsistencyRecord, ConsistencyVerdict, SlipKind, slip_kind
from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.moves import FunnelStage, TracedMove, group_by, joined
from plybench.analysis.errors.reasoning.annotations import OTHER, Annotation
from plybench.analysis.errors.reasoning.codebook import Codebook
from plybench.analysis.errors.reasoning.scope import code_applies
from plybench.analysis.errors.stores import AnalysisStores, consistency_join
from plybench.analysis.statistics.bundle import CIBundle, rate
from plybench.analysis.statistics.standardization import Reference, standardized_rate
from plybench.utils.enums import ExtendedEnum

# The share of annotated moves carrying an error no code covered, above which the codebook may not be
# frozen and the OTHER descriptions go back into induction for another wave. Written down 2026-09-24,
# before the first wave was run: a threshold chosen after seeing the number is not a stopping rule.
FREEZE_THRESHOLD = 0.05


def coded(annotation: Annotation, codebook: Codebook, code_id: str, uncorrected_only: bool = True) -> bool:
    """Whether this annotation carries the given code. Labels are resolved through merges, so a label
    written before a consolidation still counts under the code that survived it. OTHER labels are skipped
    by the membership test: they name a failure the codebook has no code for."""
    labels = annotation.uncorrected if uncorrected_only else annotation.labels
    return any(codebook.resolve(label.code_id).id == code_id for label in labels if label.code_id in codebook.codes)


def uncovered(annotation: Annotation, uncorrected_only: bool = True) -> bool:
    """Whether this annotation found a load-bearing error the codebook has no code for. Measured on moves
    induction never saw, this is the completeness of the taxonomy; measured on the moves it was built
    from, it is nothing at all."""
    labels = annotation.uncorrected if uncorrected_only else annotation.labels
    return any(label.code_id == OTHER for label in labels)


@dataclass(frozen=True)
class CodePrevalence:
    code_id: str
    name: str
    n_moves: int  # denominator: annotated moves in this cell
    n_uncorrected: int
    n_self_corrected: int  # caught and repaired by the trace itself -- recorded, never in the rate
    rate: CIBundle
    by_outcome: dict[FunnelStage, CIBundle]
    # the rate this group would have shown facing the reference mix of positions; None when no reference
    # was asked for, or when the group entered none of its strata
    standardized: CIBundle | None = None

    @property
    def recovery(self) -> float | None:
        """Share of occurrences the trace caught itself. None when the code never occurred."""
        total = self.n_uncorrected + self.n_self_corrected
        return self.n_self_corrected / total if total else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "name": self.name,
            "n_moves": self.n_moves,
            "n_uncorrected": self.n_uncorrected,
            "n_self_corrected": self.n_self_corrected,
            "recovery": self.recovery,
            "rate": self.rate.to_dict(),
            "by_outcome": {stage.value: bundle.to_dict() for stage, bundle in self.by_outcome.items()},
        }


class Account(ExtendedEnum):
    """What explains a suboptimal move. Exhaustive and mutually exclusive by construction, which is the
    point: it turns two independent passes into one account of why the move was wrong."""

    SLIP = "slip"  # the trace concluded an optimal move and a different one was played
    REASONING_ERROR = "reasoning_error"  # a coded, uncorrected error survived into the decision
    BOTH = "both"  # a coded error and a slip
    UNEXPLAINED = "unexplained"  # neither: the trace reasoned cleanly to a losing conclusion
    UNJUDGED = "unjudged"  # not covered by both passes, so nothing can be attributed


@dataclass(frozen=True)
class PrevalenceReport:
    scope: Scope
    annotator: str
    codebook_version: str
    n_moves: int  # analysable moves in the cell
    n_annotated: int
    n_clean: int  # annotated moves carrying no uncorrected label at all
    codes: list[CodePrevalence]  # only codes that occurred, most prevalent first
    any_error: CIBundle
    uncovered: CIBundle  # moves whose error no code covered -- how incomplete the codebook still is
    # The same rate over moves from games induction never read. A codebook cannot be validated on the
    # traces it was built from, so this is the honest completeness figure and `uncovered` is the
    # descriptive one. `n_naive` is its denominator and is reported with it: without it there is no
    # telling whether 5% is one move in twenty or a hundred in two thousand.
    uncovered_naive: CIBundle
    n_naive: int
    n_uncovered_naive: int
    n_uncovered: int
    # what those errors were, to read before extending the codebook. Drawn from every annotated move,
    # including ones induction read, because more descriptions make a better codebook -- it is the rate,
    # not the descriptions, that has to come from the held-out games
    uncovered_descriptions: list[str]
    accounts: dict[Account, int]  # the suboptimal decomposition; empty when consistency verdicts are absent
    # False when the codebook predates game-level provenance: the hold-out then falls back to excluding
    # the induced moves themselves, which is weaker, and saying so beats quietly reporting the stronger one
    game_level_holdout: bool = True

    @property
    def coverage(self) -> float:
        return self.n_annotated / self.n_moves if self.n_moves else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope.to_dict(),
            "annotator": self.annotator,
            "codebook_version": self.codebook_version,
            "n_moves": self.n_moves,
            "n_annotated": self.n_annotated,
            "n_clean": self.n_clean,
            "coverage": self.coverage,
            "any_error": self.any_error.to_dict(),
            "uncovered": self.uncovered.to_dict(),
            "uncovered_naive": self.uncovered_naive.to_dict(),
            "n_naive": self.n_naive,
            "n_uncovered_naive": self.n_uncovered_naive,
            "game_level_holdout": self.game_level_holdout,
            "n_uncovered": self.n_uncovered,
            "uncovered_descriptions": self.uncovered_descriptions,
            "codes": [code.to_dict() for code in self.codes],
            "accounts": {account.value: count for account, count in self.accounts.items()},
        }


def _code_prevalence(scored: list[tuple[TracedMove, Annotation]], codebook: Codebook, code_id: str, confidence: float, reference: Reference[FunnelStage] | None) -> CodePrevalence:
    # A scoped code's denominator is only the positions where that category can logically occur.
    code = codebook.codes[code_id]
    scored = [(move, annotation) for move, annotation in scored if code_applies(code, move)]
    present = [coded(annotation, codebook, code_id) for _, annotation in scored]
    corrected = sum(coded(annotation, codebook, code_id, uncorrected_only=False) and not coded(annotation, codebook, code_id) for _, annotation in scored)

    grouped = group_by(scored, lambda pair: pair[0].decision)
    hits = {stage: [coded(annotation, codebook, code_id) for _, annotation in group] for stage, group in grouped.items()}
    by_outcome = {stage: rate(hits[stage], confidence) for stage in FunnelStage if stage in hits}
    counts = {stage: (sum(values), len(values)) for stage, values in hits.items()}

    return CodePrevalence(
        code_id=code_id,
        name=codebook.codes[code_id].name,
        n_moves=len(scored),
        n_uncorrected=sum(present),
        n_self_corrected=corrected,
        rate=rate(present, confidence),
        by_outcome=by_outcome,
        standardized=standardized_rate(counts, reference, confidence) if reference else None,
    )


def account_for(move: TracedMove, annotation: Annotation | None, record: ConsistencyRecord | None) -> Account:
    """Why one suboptimal move was suboptimal. A slip only counts when the trace concluded an optimal move
    and another was played -- an inconsistency that was going to be wrong either way explains nothing."""
    if annotation is None or record is None:
        return Account.UNJUDGED
    slipped = record.verdict == ConsistencyVerdict.INCONSISTENT and record.concluded_move is not None and slip_kind(move, record.concluded_move) == SlipKind.COSTLY
    errored = bool(annotation.uncorrected)
    if slipped and errored:
        return Account.BOTH
    if slipped:
        return Account.SLIP
    return Account.REASONING_ERROR if errored else Account.UNEXPLAINED


def accounts(
    moves: Sequence[TracedMove],
    annotations: Mapping[str, Annotation],
    consistency: Mapping[str, ConsistencyRecord],
) -> dict[Account, int]:
    """The decomposition of every suboptimal move in the cell. Empty when no consistency verdicts exist,
    because without them a slip cannot be distinguished from an unexplained loss."""
    if not consistency:
        return {}
    suboptimal = [move for move in moves if not move.record.is_optimal]
    found = [account_for(move, annotations.get(move.uid), consistency.get(move.uid)) for move in suboptimal]
    return {account: found.count(account) for account in Account}


def prevalence_report(
    scope: Scope,
    moves: Sequence[TracedMove],
    annotations: Mapping[str, Annotation],
    codebook: Codebook,
    annotator: str,
    consistency: Mapping[str, ConsistencyRecord] | None = None,
    confidence: float = 0.95,
    reference: Reference[FunnelStage] | None = None,
) -> PrevalenceReport:
    """Takes the moves rather than the cell, so pooling several cells is the same call with their moves
    concatenated -- pooled over the moves, never over the cells' own rates. That is also how the per-model
    figure is built: one call with every cell that model played."""
    moves = list(moves)
    scored = joined(moves, annotations)
    naive = [(move, annotation) for move, annotation in scored if codebook.naive(move)]

    prevalences = [_code_prevalence(scored, codebook, code.id, confidence, reference) for code in codebook.active()]
    occurring = sorted((code for code in prevalences if code.n_uncorrected or code.n_self_corrected), key=lambda code: code.n_uncorrected, reverse=True)
    escaped = [annotation for _, annotation in scored if uncovered(annotation)]

    return PrevalenceReport(
        scope=scope,
        annotator=annotator,
        codebook_version=codebook.version,
        n_moves=len(moves),
        n_annotated=len(scored),
        n_clean=sum(not annotation.uncorrected for _, annotation in scored),
        codes=occurring,
        any_error=rate([bool(annotation.uncorrected) for _, annotation in scored], confidence),
        uncovered=rate([uncovered(annotation) for _, annotation in scored], confidence),
        uncovered_naive=rate([uncovered(annotation) for _, annotation in naive], confidence),
        n_naive=len(naive),
        n_uncovered_naive=sum(uncovered(annotation) for _, annotation in naive),
        n_uncovered=len(escaped),
        uncovered_descriptions=[label.description for annotation in escaped for label in annotation.uncorrected if label.code_id == OTHER and label.description],
        accounts=accounts(moves, annotations, consistency or {}),
        game_level_holdout=codebook.induced_games_known,
    )


def prevalence_reports(funnel: FunnelResult, stores: AnalysisStores) -> list[PrevalenceReport]:
    """One report per annotator, each joined against the matching consistency judge, so the suboptimal
    decomposition appears when it can be trusted and is omitted when it cannot."""
    codebook, store = stores.codebook(funnel.experiment), stores.annotations(funnel.experiment)
    reports = []
    for annotator in store.annotators():
        consistency, _ = consistency_join(stores, funnel.experiment, annotator)
        reports.append(prevalence_report(Scope.of(funnel), funnel.analyzable, store.by_move(annotator), codebook, annotator, consistency))
    return [report for report in reports if report.n_annotated]
