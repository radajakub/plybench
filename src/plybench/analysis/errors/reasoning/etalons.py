"""Electing one reference instance per code, once the taxonomy has settled.

A definition is what the annotating judge is given; an etalon is what a reader is given. Induction
already records which moves a code came from, but in first-seen order -- the first trace to show a
mistake is rarely the clearest one. This asks, per code, which of its own examples shows it best, and
stores that quote verbatim. Nothing is written by the judge that a model did not already say."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from plybench.analysis.errors.judge.prompts import Prompt, render_batch
from plybench.analysis.errors.judge.runner import Judge, ResponseCache, RunStats, run_prompts, run_stats
from plybench.analysis.errors.moves import TracedMove
from plybench.analysis.errors.reasoning.annotation import verified_quote
from plybench.analysis.errors.reasoning.codebook import Code, Codebook, Etalon

ETALON_REVISION = "etalon:v1"

MAX_CANDIDATES = 6  # more than this and the prompt is mostly traces the judge will not read carefully

_ETALON_SYSTEM = """You are picking the single clearest example of one type of reasoning mistake, to be \
printed in a paper as the reference instance of that mistake.

The mistake:
{code}

You are given the moves this mistake was recorded in. Pick the one whose trace shows it most plainly, and \
quote the exact words from that trace that show it.

Rules:
1. move must be the number of one of the moves shown.
2. evidence must be copied verbatim from that move's trace. Do not paraphrase, tidy or shorten mid-sentence.
3. Prefer the instance where the mistake is unambiguous and self-contained over one that needs the rest of \
the trace explained. A short clear instance beats a long involved one.
4. reason: one sentence on why this instance over the others."""


class ChosenEtalon(BaseModel):
    move: int = Field(description="1-based number of the move whose trace shows the mistake most clearly")
    evidence: str = Field(description="verbatim quote from that move's trace showing the mistake")
    reason: str = Field(description="one sentence on why this instance was picked over the others")


@dataclass
class EtalonRun:
    """What the pass elected, and what it refused to. A quote that is not in the trace it was attributed
    to is the one failure mode that matters here, since the whole point is that the text is real."""

    chosen: list[str] = field(default_factory=list)
    skipped_no_examples: list[str] = field(default_factory=list)  # nothing was ever recorded under this code
    rejected_evidence: list[str] = field(default_factory=list)  # a quote not found in the trace it named
    rejected_move: list[str] = field(default_factory=list)  # a move number outside the candidates shown
    stats: RunStats | None = None


def etalon_prompt(code: Code, candidates: Sequence[TracedMove]) -> Prompt:
    described = f"{code.id} — {code.name}: {code.definition}"
    # the chosen move and the solver's verdict are shown: unlike annotation, this is not a measurement and
    # there is nothing left to bias -- the code is already assigned, and only the clearest wording is at stake
    return Prompt(_ETALON_SYSTEM.format(code=described), render_batch(candidates, reveal_choice=True, reveal_optimal=True))


def candidates_for(code: Code, moves: Mapping[str, TracedMove], limit: int = MAX_CANDIDATES) -> list[TracedMove]:
    """The code's own examples, in recorded order, capped. A uid with no move behind it is dropped: the
    codebook outlives any one analysis run, so it can name moves this run did not load."""
    found = [moves[uid] for uid in code.examples if uid in moves]
    return found[:limit]


def _elected(code: Code, candidates: Sequence[TracedMove], chosen: ChosenEtalon, run: EtalonRun) -> Etalon | None:
    if not 1 <= chosen.move <= len(candidates):
        run.rejected_move.append(f"{code.id}: move {chosen.move} of {len(candidates)}")
        return None
    move = candidates[chosen.move - 1]
    quote = verified_quote(move.trace, chosen.evidence)
    if quote is None:
        run.rejected_evidence.append(f"{code.id}: {chosen.evidence[:60]}")
        return None
    return Etalon(move.uid, quote, chosen.reason.strip())


async def elect_etalons(
    judge: Judge,
    codebook: Codebook,
    moves: Mapping[str, TracedMove],
    cache: ResponseCache | None = None,
    progress: bool | None = None,
) -> EtalonRun:
    """One call per code that has examples and no etalon yet, all in flight together -- unlike induction
    the codes are independent here, so nothing has to be sequential."""
    pending = [(code, candidates) for code in codebook.active() if code.etalon is None and (candidates := candidates_for(code, moves))]
    run = EtalonRun(skipped_no_examples=[code.id for code in codebook.active() if code.etalon is None and not candidates_for(code, moves)])
    if not pending:
        return run

    prompts = [etalon_prompt(code, candidates) for code, candidates in pending]
    completions = await run_prompts(judge, ChosenEtalon, prompts, "Electing etalons", cache, progress)
    run.stats = run_stats(completions)

    for (code, candidates), completion in zip(pending, completions, strict=True):
        if completion.parsed is None:
            continue
        etalon = _elected(code, candidates, completion.parsed, run)
        if etalon is not None:
            codebook.set_etalon(code.id, etalon)
            run.chosen.append(code.id)
    return run
