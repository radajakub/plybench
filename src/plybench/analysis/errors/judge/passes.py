"""One judge pass over a set of moves. Every per-move pass does the same five things -- skip what this
annotator already covered, render a prompt each, call, map the answers onto records, append -- and differs
only in the schema it asks for and the record it stores. Those two are the type parameters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from plybench.analysis.errors.judge.prompts import Prompt
from plybench.analysis.errors.judge.runner import Judge, ResponseCache, RunStats, run_prompts, run_stats
from plybench.analysis.errors.judge.store import MoveVerdict, VerdictStore
from plybench.analysis.errors.moves import TracedMove


@dataclass(frozen=True)
class MovePass[S: BaseModel, R: MoveVerdict]:
    """`S` is the shape the judge answers in, `R` the record that gets stored -- separate because the step
    between them is where a pass does its own work: matching a named move against the legal list, or
    checking a quote is really in the trace."""

    label: str
    schema: type[S]
    prompt: Callable[[TracedMove], Prompt]
    verdict: Callable[[TracedMove, str, S], R]
    # what this annotator has already covered. Defaults to every move it ever judged, which is right for a
    # pass whose answer depends only on the move; annotation overrides it, because a verdict made under an
    # older codebook is not a verdict under the current one
    covered: Callable[[VerdictStore[R], str], set[str]] | None = None

    async def run(
        self,
        judge: Judge,
        moves: Sequence[TracedMove],
        store: VerdictStore[R],
        cache: ResponseCache | None = None,
        progress: bool | None = None,
    ) -> RunStats:
        """Judge every move this annotator has not already covered and append the verdicts, so a re-run
        extends the store instead of paying for it twice. The stats count the calls made this time,
        including the ones that produced nothing."""
        done = self.covered(store, judge.annotator) if self.covered else store.annotated_by(judge.annotator)
        pending = [move for move in moves if move.uid not in done]
        completions = await run_prompts(judge, self.schema, [self.prompt(move) for move in pending], self.label, cache, progress)

        # a completion with nothing parsed is dropped rather than stored empty: an unjudged move is not a
        # clean one, and the stats below are where its absence is accounted for
        store.extend([self.verdict(move, judge.annotator, completion.parsed) for move, completion in zip(pending, completions, strict=True) if completion.parsed is not None])
        return run_stats(completions)
