from __future__ import annotations

from plybench.analysis.errors.consistency.verdicts import ConsistencyRecord, ConsistencyStore
from plybench.analysis.errors.reasoning.annotations import AnnotationStore
from plybench.analysis.errors.reasoning.codebook import Codebook


class AnalysisStores:
    """The stored artefacts of one run, opened once and shared. Sharing matters: the passes mutate the
    store objects they are given, so a report built from the same holder sees what the pass just wrote
    instead of re-reading the file and racing it."""

    def __init__(self, codebook_label: str = "") -> None:
        # only the codebook is forkable: the verdict stores are keyed by (move, annotator) already, and
        # splitting them by judge would mean re-merging to compute any agreement between two of them
        self.codebook_label = codebook_label
        self._consistency: dict[str, ConsistencyStore] = {}
        self._annotations: dict[str, AnnotationStore] = {}
        self._codebooks: dict[str, Codebook] = {}

    def consistency(self, experiment: str) -> ConsistencyStore:
        return self._consistency.setdefault(experiment, ConsistencyStore(experiment))

    def annotations(self, experiment: str) -> AnnotationStore:
        return self._annotations.setdefault(experiment, AnnotationStore(experiment))

    def codebook(self, experiment: str) -> Codebook:
        return self._codebooks.setdefault(experiment, Codebook.load(experiment, label=self.codebook_label))


def consistency_join(stores: AnalysisStores, experiment: str, annotator: str | None = None) -> tuple[dict[str, ConsistencyRecord], str | None]:
    """The consistency verdicts to decompose suboptimal moves against: the same model's if it graded that
    pass, otherwise the only one on file. With several judges and no match the choice is the caller's, so
    nothing is joined rather than joining an arbitrary one."""
    store = stores.consistency(experiment)
    available = store.annotators()
    # the annotator is "<provider>:<model>|<revision>", so the model survives a prompt revision bump
    model = annotator.split("|")[0] if annotator else None
    chosen = next((name for name in available if model is not None and name.split("|")[0] == model), None)
    if chosen is None:
        chosen = available[0] if len(available) == 1 else None
    return (store.by_move(chosen) if chosen is not None else {}), chosen
