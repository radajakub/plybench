from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from plybench.analysis.errors.judge.prompts import Prompt, render_batch
from plybench.analysis.errors.judge.runner import Judge, ResponseCache, RunStats, run_prompts, run_stats
from plybench.analysis.errors.moves import FunnelStage, TracedMove
from plybench.analysis.errors.reasoning.annotation import verified_quote
from plybench.analysis.errors.reasoning.codebook import (
    LEVEL_FAMILY,
    LEVEL_PRESENTATION,
    LEVEL_UNIVERSAL,
    LEVELS,
    SCOPE_UNIVERSAL,
    Code,
    Codebook,
    family_scope,
    game_scope,
    render_codes,
)
from plybench.analysis.errors.reasoning.protocol import rules_block
from plybench.analysis.errors.reasoning.scope import code_applies, move_family
from plybench.analysis.recognition import original_game_name, recognizable
from plybench.common.progress import track

INDUCTION_REVISION = "induction:v3"

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
- level: which of three tiers the code belongs to.
  * "universal" -- a reasoning step that would fail the same way in any game, whatever it is about
  * "family" -- tied to the rules of the underlying game, so it would occur under any way of presenting \
those rules
  * "presentation" -- only possible given this particular formulation, and impossible once the same game \
is rendered differently
  Prefer the most general tier the failure really belongs to. "presentation" is the strong claim: it says \
the wording introduced the error, not the game

Propose a new code only when no existing code covers the failure. Prefer assigning to an existing code, \
broadening its definition in your head, over creating a near-duplicate. A move with no load-bearing error \
contributes nothing — that is a normal outcome, not a gap to fill."""

_CONSOLIDATION_SYSTEM = """You are consolidating a taxonomy of reasoning mistakes that was grown \
incrementally, so it contains synonyms and codes at inconsistent levels of generality.

Return three things:
- merges: pairs where source_id and target_id name the same failure. The target survives; pick the one \
whose definition travels better across game variants. Never merge codes that describe different failures \
just because they co-occur.
- parents: pairs where code_id is a special case of the more general parent_id. The hierarchy runs \
universal -> family -> presentation, so a chain is at most three deep.
- levels: codes whose tier was claimed wrongly during incremental coding, each with the level it should \
sit at. Each code is shown with the level it currently claims. Levels were assigned one batch at a time, \
with no view of the whole taxonomy, so this is where a code that was called presentation-specific but \
names a failure of the underlying game gets corrected. The levels are "universal", "family" and \
"presentation".

Leave a code alone if you are unsure. A taxonomy that keeps two distinct codes is recoverable; one that \
merged them is not."""


class InducedError(BaseModel):
    move: int = Field(description="1-based number of the move this error occurs in")
    code_id: str = Field(description="existing code id this is an instance of, or empty when proposing a new code")
    name: str = Field(description="short name of the proposed code; empty when code_id is given")
    definition: str = Field(description="which reasoning step failed, phrased to travel across variants; empty when code_id is given")
    evidence: str = Field(description="verbatim quote from that move's trace")
    self_corrected: bool = Field(description="the trace caught and repaired this error and the final move follows the repair")
    level: str = Field(description='one of "universal", "family", "presentation" -- the most general tier this failure really belongs to')


class InducedBatch(BaseModel):
    errors: list[InducedError]


class Merge(BaseModel):
    source_id: str
    target_id: str
    reason: str


class Parenting(BaseModel):
    code_id: str
    parent_id: str


class Levelling(BaseModel):
    code_id: str
    level: str = Field(description='"universal", "family" or "presentation"')
    reason: str


class Consolidation(BaseModel):
    merges: list[Merge]
    parents: list[Parenting]
    levels: list[Levelling] = Field(default_factory=list)


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
    # attributions per code id, the frequency spectrum the unseen-species estimators read. Counted here
    # rather than from `Code.examples`, which dedups by move and so loses a code seen twice in one trace
    per_code: dict[str, int] = field(default_factory=dict)
    unknown_codes: list[str] = field(default_factory=list)  # ids the judge invented for codes it did not propose
    rejected_evidence: list[str] = field(default_factory=list)  # proposals not anchored in their trace
    cross_level: list[str] = field(default_factory=list)  # codes used outside their declared level: evidence the level is wrong, not a rejection
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


def _scope_for(level: str, move: TracedMove) -> str:
    """The scope recording the level a proposal claimed. An unrecognised level falls back to universal:
    over-scoping a code is the recoverable mistake, since where it actually occurs is measured either way,
    whereas inventing a narrow scope from a malformed answer is not."""
    if level.strip().casefold() == LEVEL_PRESENTATION:
        return game_scope(_game_key(move))
    if level.strip().casefold() == LEVEL_FAMILY:
        return family_scope(move_family(move))
    return SCOPE_UNIVERSAL


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
                # kept, not rejected: a code turning up outside its declared level is the evidence that
                # the level claim is too narrow, and discarding it is how a taxonomy confirms itself
                if not code_applies(resolved, move):
                    run.cross_level.append(f"{resolved.id} on {_game_key(move)} (declared {resolved.level})")
                codebook.add_example(error.code_id, move.uid)
                run.per_code[resolved.id] = run.per_code.get(resolved.id, 0) + 1
                run.assignments += 1
                coded += 1
            else:
                run.unknown_codes.append(error.code_id)
            continue
        if not error.name or not error.definition:
            continue  # neither an assignment nor a usable proposal
        scope = _scope_for(error.level, move)
        code = Code(id=slug(error.name, set(codebook.codes)), name=error.name, definition=error.definition, scope=scope, examples=(move.uid,), inducer=inducer)
        codebook.add(code)
        run.per_code[code.id] = run.per_code.get(code.id, 0) + 1
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
        codebook.record_induced(batch)  # their games are held back, so completeness is measured on traces induction never read
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
    return Prompt(_CONSOLIDATION_SYSTEM, f"### The codebook\n{render_codes(codebook.active(), show_level=True)}")


@dataclass
class ConsolidationRun:
    merged: list[tuple[str, str]] = field(default_factory=list)
    parented: list[tuple[str, str]] = field(default_factory=list)
    relevelled: list[tuple[str, str]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # proposals that would have broken the taxonomy
    stats: RunStats | None = None


def _generalised_scope(code: Code, level: str) -> str | None:
    """The scope a code takes when moved to a more general level, or None when the move is a narrowing.

    Narrowing cannot be done here and is refused rather than guessed. Going from universal to family, or
    family to presentation, needs the name of the family or the presentation, and this call sees only the
    codebook -- the traces the code came from are not in front of it. Generalising throws that name away,
    which needs nothing. The asymmetry is fine: over-scoping is the recoverable mistake, because where a
    code actually occurs is measured either way."""
    target = level.strip().casefold()
    if target not in LEVELS or LEVELS.index(target) >= LEVELS.index(code.level):
        return None
    if target == LEVEL_UNIVERSAL:
        return SCOPE_UNIVERSAL
    return family_scope(original_game_name(key) if recognizable(key := code.scope.removeprefix("game:")) else key)


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

    for levelling in parsed.levels:
        if levelling.code_id not in codebook.codes:
            run.rejected.append(f"level {levelling.code_id}: unknown code")
            continue
        code = codebook.resolve(levelling.code_id)
        scope = _generalised_scope(code, levelling.level)
        if scope is None:
            run.rejected.append(f"level {code.id} {code.level}->{levelling.level}: only generalisation can be decided from the codebook alone")
            continue
        codebook.set_level(code.id, scope)
        run.relevelled.append((code.id, levelling.level))
    return run
