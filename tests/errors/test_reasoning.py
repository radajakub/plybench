"""Induction and annotation: how the codebook grows, how it is applied, and how prevalence is counted.
The judge is stubbed throughout -- what is under test is the machinery that decides what a judge's answer
is allowed to do to the taxonomy and to the rates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from pydantic import BaseModel

from plybench.analysis.errors.consistency.verdicts import ConsistencyRecord, ConsistencyVerdict
from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.judge.prompts import CHOICE_HEADER, LEGAL_HEADER, OPTIMAL_HEADER
from plybench.analysis.errors.judge.runner import Judge, ResponseCache
from plybench.analysis.errors.moves import FunnelStage, MatchupId, TracedMove
from plybench.analysis.errors.reasoning.agreement import best_pair, cohen_kappa, reliability
from plybench.analysis.errors.reasoning.annotation import (
    ANNOTATION_REVISION,
    INFORMED_ANNOTATION_REVISION,
    AppliedLabel,
    MoveAnnotation,
    annotation_prompt,
    quotes_trace,
    run_annotation,
    verified_quote,
)
from plybench.analysis.errors.reasoning.annotations import OTHER, Annotation, AnnotationStore, MistakeLabel
from plybench.analysis.errors.reasoning.codebook import Code, Codebook, game_scope
from plybench.analysis.errors.reasoning.etalons import ChosenEtalon, elect_etalons
from plybench.analysis.errors.reasoning.induction import Consolidation, InducedBatch, InducedError, Merge, Parenting, consolidate, induce, induction_prompt, slug
from plybench.analysis.errors.reasoning.protocol import rules_block
from plybench.analysis.errors.reasoning.stats import Account, account_for, prevalence_report
from plybench.analysis.stats.moves import MoveRecord
from plybench.app import PlyBench
from plybench.common.enums import StateClass
from plybench.llm import LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, LLMTokens, ModelConfig, OutputText, Provider

op = PlyBench(LLMConfig())

LEGAL = ("<A1>", "<A2>", "<A3>")
MATCHUP = MatchupId("exp", "story_magic_square:sample=False", "llm:player", "random:")
MODEL = ModelConfig(Provider.OPENAI, "judge-model", LLMCallOptions())
TRACE = "I could take A2 but then they win. Therefore the best move is A1."


def _traced(seq: int = 1, move: str = "<A1>", optimal: Sequence[str] | None = None, trace: str = TRACE, game_round: int = 1) -> TracedMove:
    optimal_moves = tuple(optimal) if optimal is not None else ("<A1>",)
    record = MoveRecord(StateClass.DECISION, move in optimal_moves, 0.0, None, len(LEGAL), None, len(LEGAL), len(optimal_moves))
    # the position carries the sequence number: two moves that render identically would be one prompt, and
    # the response cache would answer the second from the first
    return TracedMove(MATCHUP, game_round, seq, record, trace, f"board {seq}", move, LEGAL, optimal_moves)


def _funnel(moves: list[TracedMove]) -> FunnelResult:
    return FunnelResult("exp", op.registry.game_config("tic_tac_toe:"), op.registry.player_config("random:distribution=uniform"), moves)


class _Scripted:
    """Replies with a queued payload per call, so a multi-batch induction loop can be driven precisely."""

    def __init__(self, replies: list[BaseModel]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def generate(
        self,
        model_config: ModelConfig,
        system: LLMMessage,
        messages: list[LLMMessage],
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMResponse(Provider.OPENAI, "judge-model", LLMTokens(10, 0, 5), [OutputText([""])], reply.model_dump_json(), type(reply))


def _judge(replies: list[BaseModel], revision: str = "test:v1") -> tuple[Judge, _Scripted]:
    generator = _Scripted(replies)
    return Judge(generator, MODEL, revision), generator


def _error(name: str = "", code_id: str = "", move: int = 1, game_specific: bool = False, self_corrected: bool = False) -> InducedError:
    return InducedError(
        move=move,
        code_id=code_id,
        name=name,
        definition=f"definition of {name}" if name else "",
        evidence="Therefore the best move is A1.",
        self_corrected=self_corrected,
        game_specific=game_specific,
    )


def _book(*codes: Code) -> Codebook:
    book = Codebook("exp")
    for code in codes:
        book.add(code)
    return book


THREAT = Code("threat_blindness", "Missed an immediate threat", "Did not check what the opponent threatens next")


# --- induction -----------------------------------------------------------------------------------
def test_ids_are_derived_once_and_only_ever_suffixed():
    assert slug("Missed the opponent's threat!", set()) == "missed_the_opponent_s"
    assert slug("threat blindness", {"threat_blindness"}) == "threat_blindness_2"
    assert slug("!!!", set()) == "code"  # a name with nothing usable in it still yields a legal id


def _suboptimal(seq: int) -> TracedMove:
    return _traced(seq, move="<A2>", optimal=("<A1>",))


def test_induction_saturates_when_batches_keep_coding_and_stop_needing_new_codes(tmp_path):
    moves = [_suboptimal(seq) for seq in range(10)]
    # first batch proposes a code; the next two find errors and fit them to it -> patience 2 stops at batch 3
    fitted = InducedBatch(errors=[_error(code_id="threat_blindness")])
    judge, generator = _judge([InducedBatch(errors=[_error(name="Threat blindness")]), fitted])
    book = Codebook("exp")

    run = asyncio.run(induce(judge, moves, book, batch_size=2, patience=2, cache=ResponseCache(tmp_path / "c"), progress=False))

    assert run.new_codes == ["threat_blindness"] and run.new_per_batch == [1, 0, 0] and run.saturated
    assert run.batches == 3 and generator.calls == 3  # stopped early: 10 moves in batches of 2 would be 5
    assert run.instances == 3 and run.assignments == 2  # the coded errors the saturation claim rests on
    assert book.codes["threat_blindness"].examples == (moves[0].uid, moves[2].uid, moves[4].uid)


def test_a_batch_the_judge_found_nothing_in_cannot_declare_the_codebook_complete(tmp_path):
    """The counterpart to the unmixed-batch guard. "No new codes" means two opposite things: the judge
    coded errors and every one fitted, or the judge found no errors at all. Only the first is evidence of
    saturation -- counting the second is how a 120-move run produced six codes and then left 9.7% of the
    labels on a corpus of 1216 moves falling through to the uncovered escape."""
    moves = [_suboptimal(seq) for seq in range(10)]
    judge, _ = _judge([InducedBatch(errors=[_error(name="Threat blindness")]), InducedBatch(errors=[])])
    book = Codebook("exp")

    run = asyncio.run(induce(judge, moves, book, batch_size=2, patience=2, cache=ResponseCache(tmp_path / "c"), progress=False))

    assert run.batches == 5 and run.silent_batches == 4  # every batch after the first attributed nothing
    assert not run.saturated, "it ran out of moves, which is a different claim and must not read as saturation"


def test_saturation_is_refused_until_enough_errors_have_actually_been_coded(tmp_path):
    # the blunt guard: two quiet batches is a cheap way to stop after coding almost nothing, whatever the
    # judge did, so a floor on coded instances outranks patience
    moves = [_suboptimal(seq) for seq in range(10)]
    judge, _ = _judge([InducedBatch(errors=[_error(name="Threat blindness")]), InducedBatch(errors=[_error(code_id="threat_blindness")])])
    book = Codebook("exp")

    floored = asyncio.run(induce(judge, moves, book, batch_size=2, patience=2, min_instances=5, cache=ResponseCache(tmp_path / "c"), progress=False))

    judge, _ = _judge([InducedBatch(errors=[_error(name="Threat blindness")]), InducedBatch(errors=[_error(code_id="threat_blindness")])])
    unfloored = asyncio.run(induce(judge, moves, Codebook("exp"), batch_size=2, patience=2, cache=ResponseCache(tmp_path / "d"), progress=False))

    assert unfloored.batches == 3, "patience alone stops as soon as two batches need no new code"
    assert floored.batches == 5 and floored.instances == 5, "the floor keeps it reading until the evidence is there"


def test_saturation_countdown_starts_only_after_the_diversity_prefix(tmp_path):
    moves = [_suboptimal(seq) for seq in range(16)]
    first = InducedBatch(errors=[_error(name="Threat blindness")])
    fitted = InducedBatch(errors=[_error(code_id="threat_blindness")])
    judge, _ = _judge([first, fitted])

    run = asyncio.run(
        induce(
            judge,
            moves,
            Codebook("exp"),
            batch_size=2,
            patience=2,
            min_moves_before_saturation=8,
            cache=ResponseCache(tmp_path / "c"),
            progress=False,
        )
    )

    # Four batches read the required prefix. Only then do two fitted batches establish saturation.
    assert run.saturated and run.batches == 6 and run.moves_seen == 12


def test_batches_the_solver_agrees_with_cannot_end_the_loop(tmp_path):
    """A batch of moves the solver agrees with mostly holds no load-bearing error, so finding nothing in
    it is not evidence the codebook is complete. Left counting toward patience, induction stopped after
    two such batches -- which is how a live run produced a two-code codebook off five error instances."""
    moves = [_traced(seq) for seq in range(10)]  # every move optimal
    judge, _ = _judge([InducedBatch(errors=[_error(name="Threat blindness")]), InducedBatch(errors=[])])
    book = Codebook("exp")

    run = asyncio.run(induce(judge, moves, book, batch_size=2, patience=2, cache=ResponseCache(tmp_path / "c"), progress=False))

    assert run.batches == 5 and run.unmixed_batches == 5  # every batch coded, none of them able to stop the loop
    assert not run.saturated  # it ran out of moves; that is a different claim and must not read as saturation


def test_induction_records_every_move_it_was_shown_so_annotation_can_hold_them_back(tmp_path):
    moves = [_suboptimal(seq) for seq in range(4)]
    judge, _ = _judge([InducedBatch(errors=[])])
    book = Codebook("exp")

    asyncio.run(induce(judge, moves, book, batch_size=4, patience=5, cache=ResponseCache(tmp_path / "c"), progress=False))

    assert book.induced == {move.uid for move in moves}  # every move seen, not only the ones that yielded a code
    assert book.induced_games == {move.game_uid for move in moves}  # and the games, which is what is held back
    reloaded = Codebook.load("exp", book.save(tmp_path / "codebook.json"))
    assert (reloaded.induced, reloaded.induced_games) == (book.induced, book.induced_games)


def test_induction_counts_instances_per_code_so_the_spectrum_can_be_read(tmp_path):
    """`Code.examples` dedups by move, so a code the judge found twice in one trace looks like a singleton
    there. The unseen-species estimators read exactly that distinction, so the attributions are counted
    separately from the provenance."""
    move = _traced(1)
    errors = [_error(code_id="threat_blindness"), _error(code_id="threat_blindness")]
    judge, _ = _judge([InducedBatch(errors=errors)])
    book = _book(THREAT)

    run = asyncio.run(induce(judge, [move], book, batch_size=1, patience=5, cache=ResponseCache(tmp_path / "c"), progress=False))

    assert run.per_code == {"threat_blindness": 2}
    assert book.codes["threat_blindness"].examples == (move.uid,)  # one move, whatever it contributed


def test_an_assignment_to_an_existing_code_adds_provenance_instead_of_a_duplicate(tmp_path):
    move = _traced(1)
    judge, _ = _judge([InducedBatch(errors=[_error(code_id="threat_blindness"), _error(code_id="invented_by_the_judge")])])
    book = _book(THREAT)

    run = asyncio.run(induce(judge, [move], book, batch_size=1, patience=1, cache=ResponseCache(tmp_path / "c"), progress=False))

    assert run.assignments == 1 and run.new_codes == [] and len(book.active()) == 1
    assert book.codes["threat_blindness"].examples == (move.uid,)
    assert run.unknown_codes == ["invented_by_the_judge"]  # a hallucinated id is reported, never created


def test_a_game_specific_proposal_is_scoped_to_the_variant_it_came_from(tmp_path):
    judge, _ = _judge([InducedBatch(errors=[_error(name="Magic constant miscomputed", game_specific=True)])])
    book = Codebook("exp")

    asyncio.run(induce(judge, [_traced(1)], book, batch_size=1, patience=1, cache=ResponseCache(tmp_path / "c"), progress=False))

    assert book.active()[0].scope == game_scope("story_magic_square")  # the key, not the full config string


def test_annotation_sees_the_chosen_move_but_never_the_solver_verdict():
    """The split that keeps the measurement honest. The chosen move is shown because three of the seven
    coding rules are defined against it -- code only what the final move rests on, and count a repair only
    when the final move follows it. The solver's verdict is withheld because that is the half that would
    make the measurement circular, turning the error rate into a re-encoding of the optimality rate."""
    move = _traced(move="<A3>", optimal=("<A1>",))
    book = _book(THREAT)
    induction = induction_prompt([move], book)
    annotation = annotation_prompt(move, book)

    assert CHOICE_HEADER in induction.user and OPTIMAL_HEADER in induction.user  # discovery is fully informed
    assert CHOICE_HEADER in annotation.user and move.move in annotation.user
    assert OPTIMAL_HEADER not in annotation.user and "<A1>" not in annotation.user.split(CHOICE_HEADER)[1]
    assert LEGAL_HEADER in annotation.user and move.trace is not None and move.trace in annotation.user
    assert THREAT.definition in induction.system and THREAT.definition in annotation.system  # same codebook, same rules
    assert "self_corrected=true" in induction.system and "self_corrected=true" in annotation.system


def test_consolidation_applies_merges_and_parents_but_rejects_what_would_break_the_book(tmp_path):
    book = _book(
        THREAT,
        Code("xor_slip", "XOR slip", "Same failure, different words", game_scope("nim")),
        Code("nim_sum", "Nim-sum miscomputed", "Arithmetic on the xor of pile sizes is wrong", game_scope("nim")),
    )
    consolidation = Consolidation(
        merges=[Merge(source_id="xor_slip", target_id="nim_sum", reason="same failure"), Merge(source_id="ghost", target_id="nim_sum", reason="unknown")],
        parents=[Parenting(code_id="nim_sum", parent_id="threat_blindness"), Parenting(code_id="threat_blindness", parent_id="threat_blindness")],
    )
    judge, _ = _judge([consolidation])

    run = asyncio.run(consolidate(judge, book, ResponseCache(tmp_path / "c"), progress=False))

    assert run.merged == [("xor_slip", "nim_sum")] and run.parented == [("nim_sum", "threat_blindness")]
    assert len(run.rejected) == 2  # the unknown source and the self-parent
    assert book.resolve("xor_slip").id == "nim_sum" and book.codes["nim_sum"].parent_id == "threat_blindness"


# --- annotation ----------------------------------------------------------------------------------
def _applied(code_id: str, evidence: str, description: str = "", self_corrected: bool = False) -> AppliedLabel:
    return AppliedLabel(code_id=code_id, description=description, evidence=evidence, self_corrected=self_corrected)


def test_evidence_must_really_be_a_quote():
    assert quotes_trace(TRACE, "Therefore the best move is A1.")
    assert quotes_trace(TRACE, "therefore   the BEST move\nis A1.")  # reflowed and recased is still a quote
    assert not quotes_trace(TRACE, "I concluded that A3 wins")  # invented
    assert not quotes_trace(TRACE, "") and not quotes_trace(None, "anything")


def test_a_stitched_quote_is_trimmed_to_its_verbatim_span_rather_than_dropped():
    """Judges routinely copy a real passage and run on into paraphrase. Observed on the first live run:
    3 of 7 labels quoted 90-170 verbatim characters and then continued with text not in the trace."""
    stitched = "I could take A2 but then they win, so I rejected that line entirely as unsafe."
    anchored = verified_quote(TRACE, stitched, min_anchor=20)
    assert anchored == "I could take A2 but then they win" and anchored in TRACE  # the label survives, the quote is honest

    # a whole quote is returned untouched however short it is -- the anchor length only governs how much
    # of a *stitched* quote must be real before the trimmed remainder counts as a citation
    assert verified_quote(TRACE, "Therefore the best move is A1.") == "Therefore the best move is A1."
    assert verified_quote(TRACE, "take A2", min_anchor=40) == "take A2"
    assert verified_quote(TRACE, "the model considered every possible continuation carefully") is None
    assert verified_quote(TRACE, "I could take A2 but so it goes", min_anchor=40) is None  # 20 real chars, under the anchor


def test_a_label_survives_only_with_a_known_code_and_a_real_quote(tmp_path):
    reply = MoveAnnotation(
        labels=[
            _applied("threat_blindness", "Therefore the best move is A1."),
            _applied("threat_blindness", "I definitely checked every threat"),
            _applied("not_a_code", "Therefore the best move is A1."),
        ],
        notes="",
    )
    judge, _ = _judge([reply])
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    book = _book(THREAT)

    run = asyncio.run(run_annotation(judge, [_traced(1)], book, store, ResponseCache(tmp_path / "c"), progress=False))

    assert run.n_labels == 1 and run.n_rejected == 2
    assert len(run.rejected_evidence) == 1 and run.rejected_unknown_code == ["not_a_code"]
    stored = next(iter(store))
    assert [label.code_id for label in stored.labels] == ["threat_blindness"]
    assert stored.codebook_version == book.version  # which revision of the taxonomy this label was made under


def test_annotating_without_a_codebook_is_refused(tmp_path):
    judge, generator = _judge([MoveAnnotation(labels=[], notes="")])
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    try:
        asyncio.run(run_annotation(judge, [_traced(1)], Codebook("exp"), store, ResponseCache(tmp_path / "c"), progress=False))
    except ValueError as error:
        assert "empty" in str(error) and generator.calls == 0  # refused before spending anything
    else:
        raise AssertionError("an empty codebook must not be annotated against")


def test_an_error_no_code_covers_is_kept_as_evidence_the_codebook_is_incomplete(tmp_path):
    """Without somewhere to put it, an annotator that finds a real error the codebook does not name has
    only one move left -- report nothing -- and the trace is counted as clean. That turns a gap in the
    taxonomy into a low mistake rate, which is the opposite of what happened."""
    reply = MoveAnnotation(
        labels=[
            _applied("", "Therefore the best move is A1.", description="Counted the parity of the wrong pile"),
            _applied("", "I could take A2 but then they win", description=""),  # no code and nothing said: unusable
        ],
        notes="",
    )
    judge, _ = _judge([reply])
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")

    run = asyncio.run(run_annotation(judge, [_traced(1)], _book(THREAT), store, ResponseCache(tmp_path / "c"), progress=False))

    assert run.n_labels == 1 and run.n_other == 1 and run.n_rejected == 1
    stored = next(iter(store))
    assert [label.code_id for label in stored.labels] == [OTHER]
    # the description survives the file, since it is the whole point: it is what you read before extending
    assert next(iter(AnnotationStore("exp", store.path))).labels[0].description == "Counted the parity of the wrong pile"


def _prevalence(moves, *args, **kwargs):
    funnel = _funnel(moves if isinstance(moves, list) else list(moves))
    return prevalence_report(Scope.of(funnel), funnel.analyzable, *args, **kwargs)


def test_an_uncovered_error_counts_as_an_error_but_under_no_code():
    move = _traced(0)
    book = _book(THREAT)
    escape = Annotation(move.uid, "judge|v1", book.version, (MistakeLabel(OTHER, "Therefore the best move is A1.", False, "novel failure"),))
    report = _prevalence([move], {move.uid: escape}, book, "judge|v1")

    assert report.any_error.value == 1.0 and report.n_clean == 0  # it is a load-bearing error like any other
    assert report.uncovered.value == 1.0 and report.uncovered_descriptions == ["novel failure"]
    assert report.codes == []  # but it belongs to no code, so it inflates nobody's prevalence


def test_a_fresh_position_from_a_game_induction_read_does_not_measure_completeness(tmp_path):
    """The whole hold-out rests on this. Two moves of one game are two positions one ply apart, so a
    codebook asked about the second after being induced on the first is being asked about itself. The
    uncovered rate it produces there is a fit statistic, not a completeness figure."""
    read, same_game, other_game = _traced(1, game_round=7), _traced(2, game_round=7), _traced(3, game_round=8)
    book = _book(THREAT)
    book.record_induced([read])

    assert not book.naive(read) and not book.naive(same_game)  # unseen position, seen game
    assert book.naive(other_game)

    escape = MistakeLabel(OTHER, "Therefore the best move is A1.", False, "novel failure")
    annotations = {move.uid: Annotation(move.uid, "judge|v1", book.version, (escape,)) for move in (read, same_game, other_game)}
    report = _prevalence([read, same_game, other_game], annotations, book, "judge|v1")

    # every move looks uncovered, but only one of them is entitled to say so
    assert report.uncovered.value == 1.0 and report.n_uncovered == 3
    assert report.n_naive == 1 and report.n_uncovered_naive == 1 and report.uncovered_naive.value == 1.0


def test_the_completeness_rate_carries_the_denominator_it_was_measured_on(tmp_path):
    """A rate without its denominator cannot be acted on: 5% uncovered is a reason to run another wave if
    it is 100 moves in 2000 and no evidence of anything if it is one in twenty."""
    moves = [_traced(seq, game_round=seq) for seq in range(1, 5)]
    book = _book(THREAT)
    book.record_induced(moves[:2])
    escape = MistakeLabel(OTHER, "Therefore the best move is A1.", False, "novel failure")
    annotations = {move.uid: Annotation(move.uid, "judge|v1", book.version, (escape,) if move is moves[2] else ()) for move in moves}

    report = _prevalence(moves, annotations, book, "judge|v1")

    assert (report.n_uncovered_naive, report.n_naive) == (1, 2) and report.uncovered_naive.value == 0.5
    assert report.game_level_holdout


def test_a_codebook_with_no_game_provenance_reports_the_weaker_holdout_rather_than_claiming_the_stronger(tmp_path):
    """Codebooks written before games were recorded cannot be upgraded -- the uids are hashes. Falling
    back to move-level exclusion is defensible; quietly reporting it as the game-level figure is not."""
    read, same_game = _traced(1, game_round=7), _traced(2, game_round=7)
    book = _book(THREAT)
    book.induced.add(read.uid)  # as an old file loads: moves, no games

    assert not book.induced_games_known
    assert book.naive(same_game)  # the weaker exclusion is all that is available
    report = _prevalence([read, same_game], {}, book, "judge|v1")
    assert not report.game_level_holdout


def test_a_move_with_no_error_is_recorded_as_clean_rather_than_skipped(tmp_path):
    judge, _ = _judge([MoveAnnotation(labels=[], notes="clean line")])
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")

    run = asyncio.run(run_annotation(judge, [_traced(1)], _book(THREAT), store, ResponseCache(tmp_path / "c"), progress=False))

    assert run.n_labels == 0 and len(store) == 1
    stored = next(iter(store))
    assert stored.labels == () and stored.notes == "clean line"  # "no error found" is data, not an empty row


# --- prevalence ----------------------------------------------------------------------------------
def _annotation(uid: str, code_id: str | None = None, self_corrected: bool = False, annotator: str = "judge|v1") -> Annotation:
    labels = (MistakeLabel(code_id, "Therefore the best move is A1.", self_corrected),) if code_id else ()
    return Annotation(uid, annotator, "v", labels)


def test_prevalence_excludes_self_corrections_from_the_rate_but_keeps_them_on_the_record():
    moves = [_traced(0), _traced(1), _traced(2, move="<A2>")]
    book = _book(THREAT)
    annotations = {
        moves[0].uid: _annotation(moves[0].uid, "threat_blindness"),
        moves[1].uid: _annotation(moves[1].uid, "threat_blindness", self_corrected=True),
        moves[2].uid: _annotation(moves[2].uid),
    }
    report = _prevalence(moves, annotations, book, "judge|v1")

    code = report.codes[0]
    assert (code.n_uncorrected, code.n_self_corrected) == (1, 1)
    assert code.rate.value == 1 / 3 and code.recovery == 0.5  # caught half the time it happened
    assert report.n_clean == 2 and report.any_error.value == 1 / 3  # the repaired trace counts as clean
    # the split that says whether a code predicts a bad move rather than just a verbose trace
    assert code.by_outcome[FunnelStage.OPTIMAL].value == 0.5 and code.by_outcome[FunnelStage.SUBOPTIMAL].value == 0.0


def test_labels_written_against_a_retired_code_still_count_under_the_survivor():
    move = _traced(0)
    book = _book(THREAT, Code("xor_slip", "XOR slip", "same thing"))
    book.merge("xor_slip", "threat_blindness")
    report = _prevalence([move], {move.uid: _annotation(move.uid, "xor_slip")}, book, "judge|v1")

    assert [code.code_id for code in report.codes] == ["threat_blindness"] and report.codes[0].n_uncorrected == 1


def test_every_suboptimal_move_gets_exactly_one_account():
    move = _traced(0, move="<A2>", optimal=("<A1>",))  # suboptimal
    clean, errored = _annotation(move.uid), _annotation(move.uid, "threat_blindness")
    slip = ConsistencyRecord(move.uid, "j", ConsistencyVerdict.INCONSISTENT, "<A2>", "<A1>")  # concluded the optimal move
    doomed = ConsistencyRecord(move.uid, "j", ConsistencyVerdict.INCONSISTENT, "<A2>", "<A3>")  # concluded another loser
    consistent = ConsistencyRecord(move.uid, "j", ConsistencyVerdict.CONSISTENT, "<A2>", "<A2>")

    assert account_for(move, clean, slip) == Account.SLIP
    assert account_for(move, errored, slip) == Account.BOTH
    assert account_for(move, errored, consistent) == Account.REASONING_ERROR
    assert account_for(move, clean, consistent) == Account.UNEXPLAINED
    assert account_for(move, clean, doomed) == Account.UNEXPLAINED  # a slip that changed nothing explains nothing
    assert account_for(move, None, slip) == Account.UNJUDGED and account_for(move, clean, None) == Account.UNJUDGED


def test_the_decomposition_covers_the_suboptimal_moves_of_the_cell():
    optimal, suboptimal = _traced(0), _traced(1, move="<A2>", optimal=("<A1>",))
    annotations = {optimal.uid: _annotation(optimal.uid, "threat_blindness"), suboptimal.uid: _annotation(suboptimal.uid, "threat_blindness")}
    consistency = {suboptimal.uid: ConsistencyRecord(suboptimal.uid, "j", ConsistencyVerdict.CONSISTENT, "<A2>", "<A2>")}
    report = _prevalence([optimal, suboptimal], annotations, _book(THREAT), "judge|v1", consistency)

    assert report.accounts[Account.REASONING_ERROR] == 1 and sum(report.accounts.values()) == 1  # only the suboptimal move
    assert _prevalence([optimal, suboptimal], annotations, _book(THREAT), "judge|v1").accounts == {}  # no verdicts -> no claim


# --- agreement -----------------------------------------------------------------------------------
def test_kappa_corrects_for_chance_and_reports_when_it_cannot():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]).kappa == 1.0
    assert cohen_kappa([True, True, False, False], [False, False, True, True]).kappa == -1.0

    # both annotators called every move clean: expected agreement is 1, so kappa is undefined, not zero
    unanimous = cohen_kappa([False] * 5, [False] * 5)
    assert unanimous.kappa is None and unanimous.observed == 1.0 and unanimous.n_positive == 0

    partial = cohen_kappa([True, True, False, False], [True, False, False, False])
    assert partial.observed == 0.75 and partial.kappa is not None and 0 < partial.kappa < 1


def test_reliability_reports_per_code_and_overall_agreement(tmp_path):
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    book = _book(THREAT, Code("other", "Other", "another failure"))
    for uid, first, second in [("u1", "threat_blindness", "threat_blindness"), ("u2", "threat_blindness", None), ("u3", None, None)]:
        store.add(_annotation(uid, first, annotator="a|v1"))
        store.add(_annotation(uid, second, annotator="b|v1"))

    report = reliability(store, book, "a|v1", "b|v1")
    assert report["threat_blindness"].n == 3 and report["threat_blindness"].observed == 2 / 3
    assert "other" not in report  # a code neither annotator ever used carries no information
    assert report["__any__"].n == 3  # the coarse "is this trace broken at all" figure is always reported


def test_the_annotation_store_survives_a_codebook_version_change(tmp_path):
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    store.add(Annotation("u1", "judge|v1", "version-one", (MistakeLabel("threat_blindness", "quote"),)))
    lines = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert lines[0]["codebook_version"] == "version-one"

    reloaded = AnnotationStore("exp", store.path)
    assert next(iter(reloaded)).codebook_version == "version-one"  # the label stays readable under any later revision


def test_the_reliability_pair_is_chosen_by_revision_and_overlap_not_alphabetically(tmp_path):
    """The live ttt store holds two abandoned 12-move `annotation:v1` columns and one 1216-move `v2`.
    Taking annotators[0] and annotators[1] off a sorted list picked the two dead ones -- and would happily
    put two different prompt revisions side by side, which measures the prompt, not the judges."""
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    for uid in ("a", "b"):
        for annotator in ("openai:gpt-5-mini|annotation:v1", "openai:gpt-5.4|annotation:v1"):
            store.add(Annotation(uid, annotator, "v", ()))
    for uid in ("a", "b", "c", "d", "e"):
        for annotator in ("openai:gpt-5.4|annotation:v2", "zz:other|annotation:v2"):
            store.add(Annotation(uid, annotator, "v", ()))

    assert store.annotators()[:2] == ["openai:gpt-5-mini|annotation:v1", "openai:gpt-5.4|annotation:v1"]
    assert best_pair(store) == ("openai:gpt-5.4|annotation:v2", "zz:other|annotation:v2", True)


def test_a_same_protocol_pair_wins_even_when_a_cross_protocol_one_overlaps_more(tmp_path):
    # reliability is a question about judges, so two judges under one prompt beat one judge under two --
    # however many more moves the second pair happens to share
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    for uid in ("a", "b"):
        for annotator in ("openai:one|annotation:v3", "openai:two|annotation:v3"):
            store.add(Annotation(uid, annotator, "v", ()))
    for uid in ("a", "b", "c", "d", "e"):
        store.add(Annotation(uid, "openai:one|annotation:v3-informed", "v", ()))

    pair = best_pair(store)
    assert pair is not None  # two annotators share moves, so there is one; having none is its own test
    first, second, same = pair
    assert same and {first, second} == {"openai:one|annotation:v3", "openai:two|annotation:v3"}


def test_two_protocols_are_compared_when_asked_but_flagged_as_not_reliability(tmp_path):
    """Blind against informed is a deliberate experiment -- how much the blinding costs -- so it is
    returned rather than refused. The flag is what stops the number being read as inter-judge agreement:
    it mixes the judges with the prompt change and cannot separate them."""
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    for uid in ("a", "b", "c"):
        store.add(Annotation(uid, "metacentrum:gemma-4|annotation:v3", "v", ()))
        store.add(Annotation(uid, "metacentrum:gemma-4|annotation:v3-informed", "v", ()))

    pair = best_pair(store)
    assert pair is not None
    first, second, same = pair
    assert not same and {first, second} == {"metacentrum:gemma-4|annotation:v3", "metacentrum:gemma-4|annotation:v3-informed"}


def test_one_annotator_alone_is_still_no_pair(tmp_path):
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    store.add(Annotation("a", "openai:one|annotation:v3", "v", ()))
    assert best_pair(store) is None


# --- etalons -------------------------------------------------------------------------------------
def _with_examples(*uids: str) -> Codebook:
    return _book(Code("threat_blindness", THREAT.name, THREAT.definition, examples=uids))


def test_the_elected_etalon_is_a_real_quote_from_the_move_it_names(tmp_path):
    moves = [_traced(1), _traced(2)]
    book = _with_examples(*(move.uid for move in moves))
    judge, _ = _judge([ChosenEtalon(move=2, evidence="Therefore the best move is A1.", reason="shortest clear instance")])

    run = asyncio.run(elect_etalons(judge, book, {move.uid: move for move in moves}, ResponseCache(tmp_path / "c"), progress=False))

    etalon = book.codes["threat_blindness"].etalon
    assert run.chosen == ["threat_blindness"] and etalon is not None
    assert etalon.move_uid == moves[1].uid and etalon.evidence in (moves[1].trace or "")


def test_an_invented_quote_is_refused_rather_than_stored(tmp_path):
    # the entire point of an etalon is that a model really wrote it, so a quote absent from the trace is
    # worse than no etalon at all -- it would be printed in a paper as evidence
    move = _traced(1)
    book = _with_examples(move.uid)
    judge, _ = _judge([ChosenEtalon(move=1, evidence="A sentence the model never wrote.", reason="")])

    run = asyncio.run(elect_etalons(judge, book, {move.uid: move}, ResponseCache(tmp_path / "c"), progress=False))

    assert book.codes["threat_blindness"].etalon is None
    assert run.rejected_evidence and not run.chosen


def test_a_code_with_no_loadable_example_is_reported_rather_than_guessed(tmp_path):
    # the codebook outlives any one run, so it can name moves this invocation never loaded
    book = _with_examples("a-uid-from-another-run")
    judge, generator = _judge([ChosenEtalon(move=1, evidence="x", reason="")])

    run = asyncio.run(elect_etalons(judge, book, {}, ResponseCache(tmp_path / "c"), progress=False))

    assert run.skipped_no_examples == ["threat_blindness"] and generator.calls == 0


def test_the_informed_variant_adds_the_solver_verdict_and_nothing_else(tmp_path):
    """The two columns have to differ in exactly one thing or the comparison measures two changes at
    once: same codebook, same rules, same chosen move, same output schema -- only the solver's optimal
    set is added, under its own revision so both land in one store side by side."""
    move = _traced(move="<A3>", optimal=("<A1>",))
    book = _book(THREAT)
    blind, informed = annotation_prompt(move, book), annotation_prompt(move, book, informed=True)

    assert CHOICE_HEADER in blind.user and CHOICE_HEADER in informed.user
    assert OPTIMAL_HEADER not in blind.user and OPTIMAL_HEADER in informed.user
    assert rules_block() in blind.system and rules_block() in informed.system
    assert THREAT.definition in blind.system and THREAT.definition in informed.system
    assert ANNOTATION_REVISION != INFORMED_ANNOTATION_REVISION


def test_both_variants_write_their_own_column_of_the_same_store(tmp_path):
    move = _traced()
    book = _book(THREAT)
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    reply = MoveAnnotation(labels=[AppliedLabel(code_id="threat_blindness", description="", evidence=TRACE, self_corrected=False)], notes="")

    for revision, informed in ((ANNOTATION_REVISION, False), (INFORMED_ANNOTATION_REVISION, True)):
        judge, _ = _judge([reply], revision=revision)
        asyncio.run(run_annotation(judge, [move], book, store, ResponseCache(tmp_path / "c"), progress=False, informed=informed))

    # one move, two verdicts, neither overwriting the other -- which is what makes them comparable
    annotators = store.annotators()
    assert sorted(name.split("|")[-1] for name in annotators) == sorted([ANNOTATION_REVISION, INFORMED_ANNOTATION_REVISION])
    assert len(store) == 2
