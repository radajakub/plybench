from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from plybench.analysis.errors.judge.prompts import Prompt, render_batch
from plybench.analysis.errors.judge.runner import Judge, ResponseCache, RunStats, run_prompts, run_stats
from plybench.analysis.errors.moves import FunnelStage, TracedMove
from plybench.analysis.errors.reasoning.annotation import verified_quote
from plybench.analysis.errors.reasoning.codebook import SCOPE_UNIVERSAL, Code, Codebook, game_scope, render_codes
from plybench.analysis.errors.reasoning.protocol import rules_block
from plybench.common.progress import track

INDUCTION_REVISION = "induction:v2"

_INDUCTION_SYSTEM = """You are building a taxonomy of reasoning mistakes made by language models playing a \
two-player game. For each move you get the position the player was shown, the legal moves, the move it \
chose, the moves a perfect solver considers optimal, and the player's reasoning trace.

Find the load-bearing reasoning errors and either assign each one to an existing code or propose a new one.

Coding rules:
{rules}

The codebook so far:
{codes}

For every error you find, report:
- move: the number of the move it occurs in
- code_id: the id of the existing code it is an instance of, or empty if no existing code fits
- name and definition: filled in only when you are proposing a new code (code_id empty), left empty otherwise
- evidence: a verbatim quote from that move's trace
- self_corrected: true when the trace caught and repaired this error and the final move follows the repair
- game_specific: true when the code only makes sense given this variant's mechanics, false when it names a \
reasoning step that would fail the same way in any game

Propose a new code only when no existing code covers the failure. Prefer assigning to an existing code, \
broadening its definition in your head, over creating a near-duplicate. A move with no load-bearing error \
contributes nothing — that is a normal outcome, not a gap to fill."""

_CONSOLIDATION_SYSTEM = """You are consolidating a taxonomy of reasoning mistakes that was grown \
incrementally, so it contains synonyms and codes at inconsistent levels of generality.

Return two things:
- merges: pairs where source_id and target_id name the same failure. The target survives; pick the one \
whose definition travels better across game variants. Never merge codes that describe different failures \
just because they co-occur.
- parents: pairs where code_id is a variant-specific special case of the more general parent_id. Only one \
level: a parent must not itself be a specialisation.

Leave a code alone if you are unsure. A taxonomy that keeps two distinct codes is recoverable; one that \
merged them is not."""


class InducedError(BaseModel):
    move: int = Field(description="1-based number of the move this error occurs in")
    code_id: str = Field(description="existing code id this is an instance of, or empty when proposing a new code")
    name: str = Field(description="short name of the proposed code; empty when code_id is given")
    definition: str = Field(description="which reasoning step failed, phrased to travel across variants; empty when code_id is given")
    evidence: str = Field(description="verbatim quote from that move's trace")
    self_corrected: bool = Field(description="the trace caught and repaired this error and the final move follows the repair")
    game_specific: bool = Field(description="the code only makes sense for this game variant's mechanics")


class InducedBatch(BaseModel):
    errors: list[InducedError]


class Merge(BaseModel):
    source_id: str
    target_id: str
    reason: str


class Parenting(BaseModel):
    code_id: str
    parent_id: str


class Consolidation(BaseModel):
    merges: list[Merge]
    parents: list[Parenting]


@dataclass
class InductionRun:
    """What a saturation loop did. `batches` and `new_codes` together are the saturation evidence: the
    curve of new codes per batch is what justifies stopping, so it is reported, not just the total."""

    batches: int = 0
    moves_seen: int = 0
    new_codes: list[str] = field(default_factory=list)
    new_per_batch: list[int] = field(default_factory=list)  # coded batches only, so the curve means what it looks like
    assignments: int = 0  # errors mapped onto a code that already existed
    instances: int = 0  # errors attributed at all, new code or existing: what the saturation claim rests on
    unknown_codes: list[str] = field(default_factory=list)  # ids the judge invented for codes it did not propose
    rejected_evidence: list[str] = field(default_factory=list)  # proposals not anchored in their trace
    rejected_scope: list[str] = field(default_factory=list)  # existing codes applied outside their domain
    failed_batches: int = 0  # produced no answer at all: not evidence of anything, least of all saturation
    unmixed_batches: int = 0  # held no suboptimal move, so finding nothing in them says nothing either
    silent_batches: int = 0  # mixed, but the judge attributed no error: also says nothing about completeness
    saturated: bool = False  # stopped because `patience` mixed batches in a row proposed nothing new
    stats: list[RunStats] = field(default_factory=list)


def slug(name: str, taken: set[str]) -> str:
    """A readable id from the proposed name. Ids are permanent, so they are derived once and then only
    ever suffixed — never renumbered, because annotations already reference them."""
    base = re.sub(r"[^a-z0-9]+", "_", name.strip().casefold()).strip("_") or "code"
    base = "_".join(base.split("_")[:4])
    if base not in taken:
        return base
    index = 2
    while f"{base}_{index}" in taken:
        index += 1
    return f"{base}_{index}"


def _game_key(move: TracedMove) -> str:
    return move.matchup.game.split(":")[0]


def _mixed(batch: Sequence[TracedMove]) -> bool:
    """Whether this batch held anything a mistake could be found in. A batch of moves the solver agrees
    with mostly contains no load-bearing error, so its silence is not evidence that the codebook is
    complete -- counting it toward `patience` is how induction stops before it has looked at anything."""
    return any(move.decision == FunnelStage.SUBOPTIMAL for move in batch)


def induction_prompt(moves: Sequence[TracedMove], codebook: Codebook) -> Prompt:
    # informed on purpose: discovery needs to see the chosen move and what the solver preferred, which is
    # exactly what annotation is denied later so that measurement stays independent of the ground truth
    system = _INDUCTION_SYSTEM.format(rules=rules_block(), codes=render_codes(codebook.active()))
    return Prompt(system, render_batch(moves, reveal_choice=True, reveal_optimal=True))


def _apply_batch(batch: InducedBatch, moves: Sequence[TracedMove], codebook: Codebook, run: InductionRun, inducer: str = "") -> tuple[int, int]:
    """Codes added, and errors attributed at all. The second number is what makes the first mean something:
    a batch that proposed nothing because everything fitted is evidence of saturation, and a batch that
    proposed nothing because the judge found nothing is not."""
    new_codes, coded = 0, 0
    for error in batch.errors:
        if not 1 <= error.move <= len(moves):
            continue  # a move number that does not exist cannot be attributed to a trace
        move = moves[error.move - 1]
        if verified_quote(move.trace, error.evidence) is None:
            run.rejected_evidence.append(f"move {error.move}: {error.evidence[:60]}")
            continue
        if error.code_id:
            if error.code_id in codebook.codes:
                resolved = codebook.resolve(error.code_id)
                if resolved not in codebook.applicable(move):
                    run.rejected_scope.append(f"move {error.move}: {error.code_id}")
                else:
                    codebook.add_example(error.code_id, move.uid)
                    run.assignments += 1
                    coded += 1
            else:
                run.unknown_codes.append(error.code_id)
            continue
        if not error.name or not error.definition:
            continue  # neither an assignment nor a usable proposal
        scope = game_scope(_game_key(move)) if error.game_specific else SCOPE_UNIVERSAL
        code = Code(id=slug(error.name, set(codebook.codes)), name=error.name, definition=error.definition, scope=scope, examples=(move.uid,), inducer=inducer)
        codebook.add(code)
        run.new_codes.append(code.id)
        new_codes += 1
        coded += 1
    run.instances += coded
    return new_codes, coded


async def induce(
    judge: Judge,
    moves: Sequence[TracedMove],
    codebook: Codebook,
    batch_size: int = 8,
    patience: int = 2,
    min_instances: int = 0,
    min_moves_before_saturation: int = 0,
    cache: ResponseCache | None = None,
    progress: bool | None = None,
) -> InductionRun:
    """Open coding to saturation: batches of traces are coded against the codebook as it stands, and the
    loop stops once `patience` consecutive batches attribute errors and need no new code for any of them.
    Batches are sequential by necessity — each one must see what the previous one added, or the same code
    is invented twice.

    Three kinds of batch cannot end the loop, because none of them is evidence the taxonomy is complete:
    one that got no answer, one holding no suboptimal move, and one where the judge attributed no error at
    all. `min_instances` is the fourth guard and the blunt one -- saturation is refused until that many
    errors have actually been coded, however quiet the run has been. ``min_moves_before_saturation``
    additionally forces a diversity prefix to be read before the consecutive quiet-batch test starts.

    Callers should pass the moves through `interleave` first: batches are contiguous slices, so an
    unordered list saturates on whatever came first rather than on the failure space."""
    batches = [list(moves[start : start + batch_size]) for start in range(0, len(moves), batch_size)]
    run = InductionRun()
    quiet = 0

    for batch in track(batches, "Inducing", len(batches), progress):
        completions = await run_prompts(judge, InducedBatch, [induction_prompt(batch, codebook)], "Induction batch", cache, progress=False)
        run.stats.append(run_stats(completions))
        run.batches += 1

        parsed = completions[0].parsed
        if parsed is None:  # a batch that never got an answer says nothing about saturation
            run.failed_batches += 1
            continue
        run.moves_seen += len(batch)
        codebook.record_induced(move.uid for move in batch)  # annotation holds these back to validate the book
        added, coded = _apply_batch(parsed, batch, codebook, run, judge.annotator)
        run.new_per_batch.append(added)

        if not _mixed(batch):
            run.unmixed_batches += 1
            continue
        if not coded:
            run.silent_batches += 1
            continue
        if run.moves_seen <= min_moves_before_saturation:
            quiet = 0
            continue
        quiet = quiet + 1 if added == 0 else 0
        if quiet >= patience and run.instances >= min_instances:
            run.saturated = True
            break
    return run


def consolidation_prompt(codebook: Codebook) -> Prompt:
    return Prompt(_CONSOLIDATION_SYSTEM, f"### The codebook\n{render_codes(codebook.active())}")


@dataclass
class ConsolidationRun:
    merged: list[tuple[str, str]] = field(default_factory=list)
    parented: list[tuple[str, str]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # proposals that would have broken the taxonomy
    stats: RunStats | None = None


async def consolidate(judge: Judge, codebook: Codebook, cache: ResponseCache | None = None, progress: bool | None = None) -> ConsolidationRun:
    """One pass over the whole codebook to merge the synonyms incremental growth produces. Every proposal
    goes through `Codebook.merge` / `set_parent`, so a suggestion that would orphan labels or build a
    chain is rejected here rather than corrupting the book."""
    completions = await run_prompts(judge, Consolidation, [consolidation_prompt(codebook)], "Consolidating", cache, progress)
    run = ConsolidationRun(stats=run_stats(completions))
    parsed = completions[0].parsed
    if parsed is None:
        return run

    for merge in parsed.merges:
        if merge.source_id not in codebook.codes or merge.target_id not in codebook.codes:
            run.rejected.append(f"merge {merge.source_id}->{merge.target_id}: unknown code")
            continue
        try:
            codebook.merge(merge.source_id, merge.target_id)
        except (ValueError, KeyError) as error:
            run.rejected.append(f"merge {merge.source_id}->{merge.target_id}: {error}")
            continue
        run.merged.append((merge.source_id, merge.target_id))

    for parenting in parsed.parents:
        try:
            codebook.set_parent(parenting.code_id, parenting.parent_id)
        except (ValueError, KeyError) as error:
            run.rejected.append(f"parent {parenting.code_id}<-{parenting.parent_id}: {error}")
            continue
        run.parented.append((parenting.code_id, parenting.parent_id))
    return run
