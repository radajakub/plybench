from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.reasoning.annotations import AnnotationStore
from plybench.analysis.errors.reasoning.codebook import Codebook
from plybench.analysis.errors.reasoning.stats import coded
from plybench.analysis.errors.stores import AnalysisStores


@dataclass(frozen=True)
class Kappa:
    """Chance-corrected agreement on one binary decision ("does this code apply to this move?").

    `kappa` is None when it is undefined rather than zero: with both annotators calling every move the
    same way, expected agreement is 1 and the correction divides by zero. That happens constantly for a
    rare code, so `observed` and `n_positive` are reported alongside — a kappa of None on 40 moves both
    judges called clean is agreement, not a failure to measure it."""

    n: int
    observed: float
    kappa: float | None
    n_positive: int  # moves either annotator coded, the only ones that can disagree informatively

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "observed": self.observed, "kappa": self.kappa, "n_positive": self.n_positive}


def cohen_kappa(first: Sequence[bool], second: Sequence[bool]) -> Kappa:
    if len(first) != len(second):
        raise ValueError("Both annotators must have judged the same moves, in the same order")
    n = len(first)
    if n == 0:
        return Kappa(0, 0.0, None, 0)

    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    p_first, p_second = sum(first) / n, sum(second) / n
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)
    kappa = None if expected >= 1.0 else (observed - expected) / (1 - expected)
    return Kappa(n, observed, kappa, sum(a or b for a, b in zip(first, second, strict=True)))


def code_agreement(store: AnnotationStore, codebook: Codebook, first: str, second: str, code_id: str) -> Kappa:
    """Agreement between two annotators on one code, over the moves both of them judged."""
    shared = sorted(store.annotated_by(first) & store.annotated_by(second))
    by_first, by_second = store.by_move(first), store.by_move(second)
    return cohen_kappa(
        [coded(by_first[uid], codebook, code_id) for uid in shared],
        [coded(by_second[uid], codebook, code_id) for uid in shared],
    )


def any_error_agreement(store: AnnotationStore, first: str, second: str) -> Kappa:
    """Agreement on the coarsest question there is: did this trace contain an uncorrected error at all.
    Worth reporting separately, since two annotators can agree a trace is broken and still disagree about
    which code names the break."""
    shared = sorted(store.annotated_by(first) & store.annotated_by(second))
    by_first, by_second = store.by_move(first), store.by_move(second)
    return cohen_kappa([bool(by_first[uid].uncorrected) for uid in shared], [bool(by_second[uid].uncorrected) for uid in shared])


def reliability(store: AnnotationStore, codebook: Codebook, first: str, second: str) -> dict[str, Kappa]:
    """Per-code agreement plus the overall any-error figure, keyed by code id (`__any__` for the latter)."""
    report = {code.id: code_agreement(store, codebook, first, second, code.id) for code in codebook.active()}
    return {"__any__": any_error_agreement(store, first, second), **{code_id: kappa for code_id, kappa in report.items() if kappa.n_positive}}


@dataclass(frozen=True)
class ReliabilityReport:
    """Inter-judge agreement for one cell, so the kappas travel with the prevalences they qualify rather
    than living only in the terminal. This is agreement, not validity: two models can agree and both be
    wrong, which only a human-coded gold subsample can rule out."""

    scope: Scope
    first: str
    second: str
    n_shared: int  # moves both annotators judged -- the sample every kappa below is computed on
    # False when the two ran under different prompt revisions: then this measures the protocol change and
    # the judges at once, which is an experiment rather than a reliability figure
    same_protocol: bool
    codes: dict[str, Kappa]  # keyed by code id, with "__any__" for the overall any-error figure

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope.to_dict(),
            "first": self.first,
            "second": self.second,
            "n_shared": self.n_shared,
            "same_protocol": self.same_protocol,
            "codes": {code_id: kappa.to_dict() for code_id, kappa in self.codes.items()},
        }


def _revision(annotator: str) -> str:
    return annotator.split("|")[-1]


def best_pair(store: AnnotationStore) -> tuple[str, str, bool] | None:
    """The two annotators to compare, and whether they share a protocol. Most moves in common wins, with
    same-revision pairs preferred over cross-revision ones.

    Never alphabetical: that picked two abandoned 12-move columns over the 1200-move one that mattered.
    Same-revision first because two judges under one protocol is the reliability question. A cross-
    revision pair is still returned when it is all there is -- comparing the blind and informed
    annotations is a deliberate experiment -- but the flag says so, since that number measures the
    protocol change and the judges together and must not be read as reliability."""
    annotators = store.annotators()
    covered = {annotator: store.annotated_by(annotator) for annotator in annotators}
    pairs = [
        (_revision(first) == _revision(second), len(covered[first] & covered[second]), first, second)
        for index, first in enumerate(annotators)
        for second in annotators[index + 1 :]
    ]
    same, shared, first, second = max(pairs, default=(False, 0, "", ""))
    return (first, second, same) if shared else None


def reliability_report(funnel: FunnelResult, stores: AnalysisStores) -> ReliabilityReport | None:
    """Kappas between the two best-matched annotators, or None when fewer than two have judged a shared
    move. Two judges are what makes this measurable at all, so a missing second one is an absent report."""
    store, codebook = stores.annotations(funnel.experiment), stores.codebook(funnel.experiment)
    pair = best_pair(store)
    if pair is None:
        return None

    first, second, same_protocol = pair
    return ReliabilityReport(
        scope=Scope.of(funnel),
        first=first,
        second=second,
        same_protocol=same_protocol,
        n_shared=len(store.annotated_by(first) & store.annotated_by(second)),
        codes=reliability(store, codebook, first, second),
    )
