from __future__ import annotations

from collections.abc import Sequence

from plybench.analysis.errors.consistency.prompt import consistency_prompt
from plybench.analysis.errors.consistency.verdicts import ConsistencyRecord, ConsistencyStore, TraceConclusion, verdict_for
from plybench.analysis.errors.judge.passes import MovePass
from plybench.analysis.errors.judge.runner import Judge, ResponseCache, RunStats
from plybench.analysis.errors.moves import TracedMove


def _record(move: TracedMove, annotator: str, conclusion: TraceConclusion) -> ConsistencyRecord:
    verdict, concluded = verdict_for(move, conclusion)
    return ConsistencyRecord(move.uid, annotator, verdict, move.move, concluded, conclusion.evidence)


CONSISTENCY_PASS: MovePass[TraceConclusion, ConsistencyRecord] = MovePass("Consistency", TraceConclusion, consistency_prompt, _record)


async def run_consistency(
    judge: Judge,
    moves: Sequence[TracedMove],
    store: ConsistencyStore,
    cache: ResponseCache | None = None,
    progress: bool | None = None,
) -> RunStats:
    return await CONSISTENCY_PASS.run(judge, moves, store, cache, progress)
