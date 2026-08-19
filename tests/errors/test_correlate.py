"""The two joins the experiment is actually about: reasoning codes against what the solver said about the
same move, and against how long the trace was."""

from plybench.analysis.errors.facets import PROVIDER_SUMMARY, RAW_REASONING
from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.moves import MatchupId, TracedMove
from plybench.analysis.errors.procedural.detection import MoveDiagnosis, MoveLabel
from plybench.analysis.errors.reasoning.annotations import Annotation, MistakeLabel
from plybench.analysis.errors.reasoning.codebook import Code, Codebook
from plybench.analysis.errors.reasoning.correlate import code_by_label, code_by_length, length_buckets
from plybench.analysis.stats.moves import MoveRecord
from plybench.common.enums import StateClass

MATCHUP = MatchupId("exp", "tic_tac_toe:", "llm:player", "random:")
SCOPE = Scope((("experiment", "exp"),))


def _move(seq: int, trace: str = "reasoning") -> TracedMove:
    record = MoveRecord(StateClass.DECISION, False, 1.0, None, 9, None, 3, 1)
    return TracedMove(MATCHUP, 1, seq, record, trace, f"board {seq}", "<A1>", ("<A1>", "<A2>"), ("<A2>",))


def _book() -> Codebook:
    book = Codebook("exp")
    book.add(Code("threat_blindness", "Missed an immediate threat", "did not check what the opponent threatens"))
    book.add(Code("arithmetic_slip", "Arithmetic slip", "added the row wrong"))
    return book


def _annotation(move: TracedMove, *code_ids: str) -> Annotation:
    return Annotation(move.uid, "judge|v1", "v", tuple(MistakeLabel(code_id, "reasoning") for code_id in code_ids))


def test_a_code_that_tracks_a_tactical_label_shows_up_as_lift():
    """Design question 3. A reasoning code whose rate is much higher on the moves the solver says missed a
    block is diagnostic of that failure; a flat one is describing how the model writes."""
    blocked = [_move(seq) for seq in range(10)]  # every one missed a block
    forked = [_move(100 + seq) for seq in range(10)]  # every one allowed a fork
    diagnoses = {move.uid: MoveDiagnosis(frozenset({MoveLabel.MISSED_BLOCK}), None) for move in blocked}
    diagnoses |= {move.uid: MoveDiagnosis(frozenset({MoveLabel.ALLOWED_FORK}), None) for move in forked}
    # threat_blindness only ever occurs where a block was missed; arithmetic_slip is spread evenly
    annotations = {move.uid: _annotation(move, "threat_blindness", "arithmetic_slip") for move in blocked}
    annotations |= {move.uid: _annotation(move, "arithmetic_slip") for move in forked}

    report = code_by_label(SCOPE, blocked + forked, annotations, diagnoses, _book(), "judge|v1")

    threat = next(code for code in report.codes if code.code_id == "threat_blindness")
    assert threat.by_bucket["missed_block"].value == 1.0 and threat.by_bucket["allowed_fork"].value == 0.0
    assert threat.strongest == ("missed_block", 2.0)  # twice its overall rate of 0.5

    slip = next(code for code in report.codes if code.code_id == "arithmetic_slip")
    assert slip.lift == {"missed_block": 1.0, "allowed_fork": 1.0}  # no association at all, which is also a result


def test_a_move_carrying_two_labels_counts_under_both():
    # the buckets overlap on purpose: "of the moves that missed a block, how often was the trace blind"
    # is the question, and it does not need the columns to partition anything
    move = _move(1)
    diagnoses = {move.uid: MoveDiagnosis(frozenset({MoveLabel.MISSED_BLOCK, MoveLabel.THREW_WIN}), None)}
    report = code_by_label(SCOPE, [move], {move.uid: _annotation(move, "threat_blindness")}, diagnoses, _book(), "judge|v1")

    assert report.n_bucketed == {"missed_block": 1, "threw_win": 1}
    assert report.n_moves == 1  # one move, two buckets, so the columns do not sum to the total


def test_length_buckets_are_quantiles_because_the_distribution_is_skewed():
    # a handful of very long traces and many short ones: any fixed character threshold puts almost
    # everything in one bucket, so the cut points come from the data
    moves = [_move(seq, trace="x" * length) for seq, length in enumerate([10, 20, 30, 40, 5000, 9000])]
    assert length_buckets(moves, 2) == [40]  # three traces below, three at or above
    assert length_buckets(moves, 3) == [30, 5000]
    assert length_buckets([], 4) == []


def test_longer_traces_can_carry_a_different_error_profile():
    """Design question 4. Tokens per move are known to rise with obfuscation while optimality stays flat;
    this is what asks whether the extra length buys anything, per error type."""
    short = [_move(seq, trace="x" * 10) for seq in range(6)]
    long = [_move(100 + seq, trace="x" * 5000) for seq in range(6)]
    annotations = {move.uid: _annotation(move, "threat_blindness") for move in short}
    annotations |= {move.uid: _annotation(move) for move in long}

    report = code_by_length(SCOPE, short + long, annotations, _book(), "judge|v1", n_buckets=2)

    assert report.n_bucketed == {bucket: 6 for bucket in report.buckets}  # quantiles, so equal denominators
    threat = next(code for code in report.codes if code.code_id == "threat_blindness")
    assert threat.by_bucket[report.buckets[0]].value == 1.0  # every short trace
    assert threat.by_bucket[report.buckets[1]].value == 0.0  # none of the long ones


def test_a_self_hosted_model_returns_reasoning_and_a_commercial_one_returns_a_summary():
    """The one distinction that may never be pooled away: a rate over a provider's summary is a rate over
    what its summariser kept, and is not the same measurement as a rate over a chain of thought."""
    from plybench.analysis.errors.facets import _trace_kind  # noqa: PLC0415
    from plybench.app import PlyBench  # noqa: PLC0415
    from plybench.llm import LLMConfig  # noqa: PLC0415

    registry = PlyBench(LLMConfig()).registry
    assert _trace_kind(registry.player_config("llm:actions:text:metacentrum:gemma-4:thinking_enabled=True")) == RAW_REASONING
    assert _trace_kind(registry.player_config("llm:actions:text:openai:gpt-5-nano:thinking_enabled=True")) == PROVIDER_SUMMARY
    assert _trace_kind(registry.player_config("random:distribution=uniform")) == "-"  # a bot has no trace at all
