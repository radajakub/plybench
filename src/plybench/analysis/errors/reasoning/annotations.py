from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plybench.analysis.errors.judge.store import VerdictStore
from plybench.common.paths import ReasoningPathBuilder

# a load-bearing error the annotator found but no code in the codebook covers. Kept as a label rather than
# dropped, because "the codebook missed this" and "this trace was fine" are opposite findings, and an
# annotator with nowhere to put the former reports the latter.
OTHER = "__other__"


@dataclass(frozen=True, slots=True)
class MistakeLabel:
    """One coded mistake in one trace. `self_corrected` marks the case the coding protocol singles out:
    the trace made this error and then repaired it itself, with the final move following the repair. Such
    a label is kept but excluded from uncorrected-mistake rates, so recovery can be measured rather than
    silently discarded."""

    code_id: str
    evidence: str  # verbatim quote from the trace
    self_corrected: bool = False
    description: str = ""  # OTHER labels only: what the annotator says the uncovered failure was

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code_id": self.code_id, "evidence": self.evidence, "self_corrected": self.self_corrected}
        if self.description:
            data["description"] = self.description
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MistakeLabel:
        return cls(
            code_id=data["code_id"],
            evidence=data["evidence"],
            self_corrected=bool(data.get("self_corrected", False)),
            description=data.get("description", ""),
        )


@dataclass(frozen=True, slots=True)
class Annotation:
    """One annotator's verdict on one move. Keyed by (move_uid, annotator) so the same move can be coded
    twice by different annotators for inter-rater reliability without either overwriting the other."""

    move_uid: str
    annotator: str  # model id + prompt revision, so a protocol change is visible in the data
    codebook_version: str
    labels: tuple[MistakeLabel, ...] = ()
    notes: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.move_uid, self.annotator)

    @property
    def uncorrected(self) -> tuple[MistakeLabel, ...]:
        return tuple(label for label in self.labels if not label.self_corrected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "move_uid": self.move_uid,
            "annotator": self.annotator,
            "codebook_version": self.codebook_version,
            "labels": [label.to_dict() for label in self.labels],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        return cls(
            move_uid=data["move_uid"],
            annotator=data["annotator"],
            codebook_version=data["codebook_version"],
            labels=tuple(MistakeLabel.from_dict(label) for label in data.get("labels", [])),
            notes=data.get("notes"),
        )


class AnnotationStore(VerdictStore[Annotation]):
    def __init__(self, experiment: str, path: Path | None = None) -> None:
        self.experiment = experiment
        super().__init__(path if path is not None else ReasoningPathBuilder().annotations(experiment), Annotation.from_dict)

    def annotated_under(self, annotator: str, codebook_version: str) -> set[str]:
        """Moves this annotator judged against *this* version of the taxonomy -- what a resumed run may
        skip. A label made under an older codebook is stale rather than reusable: it was chosen from a
        different set of options, so counting it as coverage would silently mix two taxonomies in one
        column. Extending the codebook therefore re-annotates, which is the expensive but honest answer."""
        return {record.move_uid for record in self if record.annotator == annotator and record.codebook_version == codebook_version}
