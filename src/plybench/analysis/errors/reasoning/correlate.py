"""The two classifications held against each other, and against how long the trace was.

Everything up to here measures one thing at a time: what the move was worth, and what the trace got
wrong. These are the joins, and they are what the experiment is actually about.

- `code_by_label` crosses each reasoning code with the tactical labels minimax put on the same move. A
  strong association makes the trace diagnostic of the decision; a flat one means the trace is narration
  written beside the decision rather than an account of it. Either answer is a result.
- `code_by_length` crosses each code with the length of the trace it was found in. Tokens per move are
  already known to rise with obfuscation while optimality stays flat; this asks whether the extra length
  buys anything, per error type.

Both are descriptive contingency tables with Wilson intervals, never a significance test: the cells are
not independent (one move carries several codes), so a chi-square over this table would be wrong.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.moves import TracedMove, joined
from plybench.analysis.errors.procedural.detection import MoveDiagnosis, MoveLabel
from plybench.analysis.errors.reasoning.annotations import Annotation
from plybench.analysis.errors.reasoning.codebook import Codebook
from plybench.analysis.errors.reasoning.stats import coded
from plybench.analysis.statistics.bundle import CIBundle, rate


@dataclass(frozen=True)
class CodeCrosstab:
    """One reasoning code against one set of buckets, as a rate per bucket.

    `baseline` is the code's rate over every annotated move in the same scope. A bucket rate far from it
    is the association; without it a high rate in a bucket says only that the code is common."""

    code_id: str
    name: str
    baseline: CIBundle
    by_bucket: dict[str, CIBundle]

    @property
    def lift(self) -> dict[str, float]:
        """Bucket rate over baseline rate. 1.0 is no association, 2.0 is twice as likely in that bucket."""
        return {bucket: bundle.value / self.baseline.value for bucket, bundle in self.by_bucket.items() if self.baseline.value}

    @property
    def strongest(self) -> tuple[str, float] | None:
        lift = {bucket: value for bucket, value in self.lift.items() if self.by_bucket[bucket].n}
        return max(lift.items(), key=lambda item: item[1]) if lift else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "name": self.name,
            "baseline": self.baseline.to_dict(),
            "by_bucket": {bucket: bundle.to_dict() for bucket, bundle in self.by_bucket.items()},
            "lift": self.lift,
        }


@dataclass(frozen=True)
class CrosstabReport:
    scope: Scope
    annotator: str
    axis: str  # what the buckets are: "procedural label" or "trace length"
    buckets: list[str]  # in the order they should be printed
    n_moves: int  # annotated moves the table was built over
    n_bucketed: dict[str, int]  # how many moves fell in each bucket, the denominators
    codes: list[CodeCrosstab]
    any_error: CodeCrosstab  # the same table for "carries any uncorrected error", as the reference row

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scope.to_dict(),
            "annotator": self.annotator,
            "axis": self.axis,
            "buckets": self.buckets,
            "n_moves": self.n_moves,
            "n_bucketed": self.n_bucketed,
            "any_error": self.any_error.to_dict(),
            "codes": [code.to_dict() for code in self.codes],
        }


def _crosstab(
    scope: Scope,
    axis: str,
    scored: Sequence[tuple[TracedMove, Annotation]],
    codebook: Codebook,
    annotator: str,
    buckets: Mapping[str, list[tuple[TracedMove, Annotation]]],
    order: Sequence[str],
    confidence: float,
) -> CrosstabReport:
    def row(code_id: str | None, name: str) -> CodeCrosstab:
        def hit(annotation: Annotation) -> bool:
            return bool(annotation.uncorrected) if code_id is None else coded(annotation, codebook, code_id)

        return CodeCrosstab(
            code_id=code_id or "any_error",
            name=name,
            baseline=rate([hit(annotation) for _, annotation in scored], confidence),
            by_bucket={bucket: rate([hit(annotation) for _, annotation in buckets.get(bucket, [])], confidence) for bucket in order},
        )

    codes = [row(code.id, code.name) for code in codebook.active()]
    return CrosstabReport(
        scope=scope,
        annotator=annotator,
        axis=axis,
        buckets=list(order),
        n_moves=len(scored),
        n_bucketed={bucket: len(buckets.get(bucket, [])) for bucket in order},
        codes=[code for code in codes if code.baseline.value],
        any_error=row(None, "any uncorrected error"),
    )


def code_by_label(
    scope: Scope,
    moves: Sequence[TracedMove],
    annotations: Mapping[str, Annotation],
    diagnoses: Mapping[str, MoveDiagnosis],
    codebook: Codebook,
    annotator: str,
    confidence: float = 0.95,
) -> CrosstabReport:
    """Design question 3: does a reasoning error predict a kind of bad move?

    A move carries several tactical labels at once, so the buckets overlap by construction and the rows do
    not sum to the total. That is correct for the question -- "of the moves that missed a block, how often
    was the trace blind to the threat" -- and it is why no independence test belongs here."""
    scored = joined(moves, annotations)
    buckets: dict[str, list[tuple[TracedMove, Annotation]]] = {}
    for move, annotation in scored:
        for label in diagnoses[move.uid].labels if move.uid in diagnoses else ():
            buckets.setdefault(label.value, []).append((move, annotation))
    order = [label.value for label in MoveLabel if label.value in buckets]
    return _crosstab(scope, "procedural label", scored, codebook, annotator, buckets, order, confidence)


def length_buckets(moves: Sequence[TracedMove], n_buckets: int = 4) -> list[int]:
    """Quantile cut points over the traces actually present, so the buckets hold equal numbers of moves.

    Fixed character thresholds would be useless here: the length distribution is heavily right-skewed --
    median 3 190 characters against a mean of 5 806 -- so any round number puts almost everything in one
    bucket."""
    lengths = sorted(len(move.trace) for move in moves if move.trace)
    if not lengths or n_buckets < 2:
        return []
    return [lengths[len(lengths) * index // n_buckets] for index in range(1, n_buckets)]


def _length_bucket(move: TracedMove, cuts: Sequence[int]) -> str:
    length = len(move.trace or "")
    index = sum(length >= cut for cut in cuts)
    lower = cuts[index - 1] if index else 0
    upper = cuts[index] if index < len(cuts) else None
    return f"{index + 1}. {lower}-{upper}" if upper is not None else f"{index + 1}. {lower}+"


def code_by_length(
    scope: Scope,
    moves: Sequence[TracedMove],
    annotations: Mapping[str, Annotation],
    codebook: Codebook,
    annotator: str,
    n_buckets: int = 4,
    confidence: float = 0.95,
) -> CrosstabReport:
    """Design question 4: are longer traces better traces, per error type?

    Length is measured in characters of stored trace, which is not comparable across providers -- a
    commercial model stores a summary, not its reasoning. Read this within a group, never across the two.
    """
    scored = joined(moves, annotations)
    cuts = length_buckets([move for move, _ in scored], n_buckets)
    buckets: dict[str, list[tuple[TracedMove, Annotation]]] = {}
    for move, annotation in scored:
        buckets.setdefault(_length_bucket(move, cuts), []).append((move, annotation))
    return _crosstab(scope, "trace length (characters)", scored, codebook, annotator, buckets, sorted(buckets), confidence)
