from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import partial

from pydantic import BaseModel, Field

from plybench.analysis.errors.judge.passes import MovePass
from plybench.analysis.errors.judge.prompts import Prompt, render_move
from plybench.analysis.errors.judge.runner import Judge, ResponseCache, RunStats
from plybench.analysis.errors.moves import TracedMove
from plybench.analysis.errors.reasoning.annotations import OTHER, Annotation, AnnotationStore, MistakeLabel
from plybench.analysis.errors.reasoning.codebook import Codebook, render_codes
from plybench.analysis.errors.reasoning.protocol import rules_block

ANNOTATION_REVISION = "annotation:v4"
# the same codebook and the same rules, told what the solver preferred. A separate revision rather than a
# setting, so the two columns sit side by side in one store and can be compared instead of argued about
INFORMED_ANNOTATION_REVISION = "annotation:v4-informed"

# shown to both variants: the chosen move, because rules 1, 3 and 4 are all defined against it
_CHOSEN = """You are told which move was finally played, because the coding rules are defined in terms of \
it: only reasoning the chosen move rests on counts, and a self-correction only counts when the chosen move \
follows the repair."""

_BLIND = """You are NOT told whether that move was any good, and no solver verdict is given. Do not try to \
work out whether it was the best move -- that is measured separately, and an error you only believe in \
because you suspect the move was bad is not an error. Judge the reasoning on its own terms."""

_INFORMED = """You are also told which moves a perfect solver considers optimal, so you can see where the \
reasoning diverged from what was actually correct.

Use it to locate errors, never to infer them. Rule 7 below is the one that matters here and it cuts both \
ways: a move the solver agrees with is often reached by faulty reasoning, and a move the solver rejects \
often rests on reasoning with no identifiable flaw. "The solver disagreed, so something must be wrong" is \
not a finding and must not produce a label; neither is "the solver agreed, so the reasoning is fine" a \
reason to stop reading."""

_ANNOTATION_SYSTEM = """You are applying a fixed codebook of reasoning mistakes to one move made by a \
language model playing a two-player game. You get the position the player was shown, the legal moves, and \
the player's reasoning trace.

{given}

Coding rules:
{rules}

The codebook — use these codes and no others:
{codes}

For every load-bearing error in this trace, report the code_id it falls under, a verbatim quote from the \
trace as evidence, and whether the trace itself caught and repaired it. Report no labels when the trace \
holds no load-bearing error: most traces are not mistaken, and inventing a label to fill the slot corrupts \
the measurement this feeds.

If the trace holds a load-bearing error that no code in the codebook covers, report it with an empty \
code_id and say what failed in `description`. Use this only for an error you would have coded had a code \
existed — it is how we measure where the codebook is incomplete, not somewhere to file a trace you are \
unsure about. Never invent a code id."""


class AppliedLabel(BaseModel):
    code_id: str = Field(description="id of a code from the codebook, exactly as listed; empty when no code covers this error")
    description: str = Field(description="which reasoning step failed; filled in only when code_id is empty, left empty otherwise")
    evidence: str = Field(description="verbatim quote from this trace showing the error")
    self_corrected: bool = Field(description="the trace caught and repaired this error and the final move follows the repair")


class MoveAnnotation(BaseModel):
    labels: list[AppliedLabel] = Field(description="one entry per load-bearing error; empty when no code applies")
    notes: str = Field(description="a short remark only if something about this trace needs one; empty otherwise")


def _squeezed(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _collapsed(text: str) -> str:
    return _squeezed(text).casefold()


MIN_ANCHOR = 40  # characters of verbatim trace text below which a quote is not evidence of anything


def quotes_trace(trace: str | None, evidence: str) -> bool:
    """Whether the evidence is a quote in full. Whitespace and case are forgiven — a judge that reflowed a
    line is still pointing at it — but invented text is not: a label whose quote is not in the trace is
    evidence of nothing, and counting it would inflate every rate downstream."""
    if not trace or not evidence.strip():
        return False
    return _collapsed(evidence) in _collapsed(trace)


def verified_quote(trace: str | None, evidence: str, min_anchor: int = MIN_ANCHOR) -> str | None:
    """The part of the evidence that is genuinely in the trace, or None when too little of it is.

    Judges routinely stitch: they copy a real passage and run on into paraphrase, so a whole-quote test
    throws away labels that do point at a real line. Trimming to the verbatim span keeps the label and
    makes the stored quote honest, while a label anchored in less than `min_anchor` characters of trace
    is discarded — at that length a match is a common phrase rather than a citation."""
    if not trace or not evidence.strip():
        return None
    if quotes_trace(trace, evidence):
        return evidence.strip()

    # matched case-insensitively but sliced out of the trace's own text, so the span keeps the wording the
    # model actually wrote; the final check makes any index skew from case folding harmless
    trace_text, evidence_text = _squeezed(trace), _squeezed(evidence)
    match = SequenceMatcher(None, trace_text.lower(), evidence_text.lower(), autojunk=False).find_longest_match(0, len(trace_text), 0, len(evidence_text))
    if match.size < min(min_anchor, len(evidence_text)):
        return None
    span = trace_text[match.a : match.a + match.size].strip()
    return span if quotes_trace(trace, span) else None


def annotation_prompt(move: TracedMove, codebook: Codebook, informed: bool = False) -> Prompt:
    """The chosen move is always shown; the solver's verdict only when `informed`.

    Rules 1, 3 and 4 are defined against "the final chosen move", so withholding it left the judge
    inferring the trace's own conclusion -- the consistency pass's job, done worse and as a side effect.

    Whether the verdict should be shown too is an open question, which is why both variants exist. Blind
    risks a judge that cannot verify anything calling plausible reasoning wrong; informed risks a judge
    that reasons backwards from the outcome, which would make the error rate a re-encoding of an
    optimality rate the solver already gives exactly. Neither bias is measured yet -- run both and compare.

    Never revealed either way: which model wrote the trace. That is the blinding the cross-model
    comparison actually rests on."""
    given = f"{_CHOSEN}\n\n{_INFORMED if informed else _BLIND}"
    system = _ANNOTATION_SYSTEM.format(given=given, rules=rules_block(), codes=render_codes(codebook.applicable(move)))
    return Prompt(system, render_move(move, reveal_choice=True, reveal_optimal=informed))


@dataclass
class AnnotationRun:
    """What an annotation pass produced, including what it threw away. The rejections are quality data:
    a judge quoting text that is not in the trace, or naming codes that do not exist, is a protocol
    failure that must be visible rather than absorbed."""

    stats: RunStats | None = None
    n_labels: int = 0
    n_self_corrected: int = 0
    n_other: int = 0  # errors no code covered: the codebook's incompleteness, measured rather than lost
    n_trimmed: int = 0  # labels kept, but with a stitched quote cut back to the part really in the trace
    rejected_unknown_code: list[str] = field(default_factory=list)
    rejected_evidence: list[str] = field(default_factory=list)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected_unknown_code) + len(self.rejected_evidence)


class LabelResolver:
    """The step between a judge's answer and a stored annotation. Stateful because what it rejects is the
    pass's own quality data: a rejection belongs to the run, not to the move it happened on."""

    def __init__(self, codebook: Codebook) -> None:
        self.codebook = codebook
        self.run = AnnotationRun()

    def __call__(self, move: TracedMove, annotator: str, parsed: MoveAnnotation) -> Annotation:
        labels = tuple(label for applied in parsed.labels if (label := self._label(move, applied)) is not None)
        return Annotation(move.uid, annotator, self.codebook.version, labels, parsed.notes.strip() or None)

    def _label(self, move: TracedMove, applied: AppliedLabel) -> MistakeLabel | None:
        code_id, description = applied.code_id.strip(), applied.description.strip()
        if not code_id and not description:
            self.run.rejected_unknown_code.append("(no code id and no description)")  # neither a label nor a usable escape
            return None
        if code_id and code_id not in self.codebook.codes:
            self.run.rejected_unknown_code.append(code_id)
            return None
        if code_id and self.codebook.resolve(code_id) not in self.codebook.applicable(move):
            self.run.rejected_unknown_code.append(f"{code_id} (out of scope)")
            return None
        quote = verified_quote(move.trace, applied.evidence)
        if quote is None:
            self.run.rejected_evidence.append(f"{code_id or OTHER}: {applied.evidence[:60]}")
            return None

        self.run.n_trimmed += not quotes_trace(move.trace, applied.evidence)
        self.run.n_labels += 1
        self.run.n_other += not code_id
        self.run.n_self_corrected += applied.self_corrected
        # resolve through merges so a label made against a retired id is stored under the surviving code
        resolved = self.codebook.resolve(code_id).id if code_id else OTHER
        return MistakeLabel(resolved, quote, applied.self_corrected, "" if code_id else description)

    def finished(self, stats: RunStats) -> AnnotationRun:
        self.run.stats = stats
        return self.run


async def run_annotation(
    judge: Judge,
    moves: Sequence[TracedMove],
    codebook: Codebook,
    store: AnnotationStore,
    cache: ResponseCache | None = None,
    progress: bool | None = None,
    informed: bool = False,
) -> AnnotationRun:
    """Apply the codebook to every given move and append the annotations. A codebook change makes the
    version recorded on the new annotations differ from the old ones, which is what keeps a mixed set
    readable."""
    if not codebook.active():
        raise ValueError("The codebook is empty -- run the induction pass before annotating")

    resolver = LabelResolver(codebook)
    annotating: MovePass[MoveAnnotation, Annotation] = MovePass(
        "Annotating (informed)" if informed else "Annotating",
        MoveAnnotation,
        partial(annotation_prompt, codebook=codebook, informed=informed),
        resolver,
        covered=lambda annotations, annotator: annotations.annotated_under(annotator, codebook.version),
    )
    return resolver.finished(await annotating.run(judge, moves, store, cache, progress))
