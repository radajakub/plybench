"""The consistency pass: did the model play the move its own trace concluded? The judge only extracts the
conclusion -- the comparison is done in code -- so these tests pin what a disagreement means, plus the
caching/resume behaviour that keeps a re-run from paying for verdicts it already has."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from pydantic import BaseModel

from plybench.analysis.errors.consistency.prompt import consistency_prompt
from plybench.analysis.errors.consistency.run import run_consistency
from plybench.analysis.errors.consistency.stats import ConsistencyReport, consistency_report
from plybench.analysis.errors.consistency.verdicts import ConsistencyRecord, ConsistencyStore, ConsistencyVerdict, SlipKind, TraceConclusion, slip_kind, verdict_for
from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.judge.runner import Judge, ResponseCache
from plybench.analysis.errors.moves import MatchupId, TracedMove
from plybench.analysis.stats.moves import MoveRecord
from plybench.app import PlyBench
from plybench.common.enums import StateClass
from plybench.llm import LLMCallOptions, LLMConfig, LLMMessage, LLMResponse, LLMTokens, ModelConfig, OutputText, Provider

op = PlyBench(LLMConfig())

LEGAL = ("<A1>", "<A2>", "<A3>")
MATCHUP = MatchupId("exp", "tic_tac_toe:", "llm:player", "random:")
MODEL = ModelConfig(Provider.OPENAI, "judge-model", LLMCallOptions())
ANNOTATOR = "openai:judge-model|consistency:test"


def _traced(
    seq: int = 1,
    move: str = "<A1>",
    optimal: Sequence[str] | None = None,
    state_class: StateClass = StateClass.DECISION,
    observation: str = "board",
    trace: str = "so I will play A1",
) -> TracedMove:
    optimal_moves = tuple(optimal) if optimal is not None else (LEGAL if state_class.is_forced else ("<A1>",))
    record = MoveRecord(state_class, move in optimal_moves, 0.0, None, 100, None, len(LEGAL), len(optimal_moves))
    return TracedMove(MATCHUP, 1, seq, record, trace, observation, move, LEGAL, optimal_moves)


def _funnel(moves: list[TracedMove]) -> FunnelResult:
    return FunnelResult("exp", op.registry.game_config("tic_tac_toe:"), op.registry.player_config("random:distribution=uniform"), moves)


def _decided(move: str) -> TraceConclusion:
    return TraceConclusion(status="decided", move=move, evidence=f"so I will play {move.strip('<>')}")


class _StubGenerator:
    """Stands in for the LLM: replies with scripted conclusions and counts the calls, which is what the
    cache and resume assertions are really about."""

    def __init__(self, reply: TraceConclusion) -> None:
        self.reply = reply
        self.calls: list[str] = []

    async def generate(
        self,
        model_config: ModelConfig,
        system: LLMMessage,
        messages: list[LLMMessage],
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages[0].content)
        return LLMResponse(Provider.OPENAI, "judge-model", LLMTokens(10, 0, 5), [OutputText([""])], self.reply.model_dump_json(), TraceConclusion)


def _judge(reply: TraceConclusion) -> tuple[Judge, _StubGenerator]:
    generator = _StubGenerator(reply)
    return Judge(generator, MODEL, "consistency:test"), generator


def _report(moves: list[TracedMove], records: list[ConsistencyRecord]) -> ConsistencyReport:
    funnel = _funnel(moves)
    return consistency_report(Scope.of(funnel), funnel.analyzable, {record.move_uid: record for record in records}, ANNOTATOR)


# --- the verdict, which is computed rather than judged -------------------------------------------
def test_the_conclusion_is_matched_against_the_played_move_mechanically():
    move = _traced(move="<A1>")
    assert verdict_for(move, _decided("<A1>")) == (ConsistencyVerdict.CONSISTENT, "<A1>")
    concludes_a2 = _traced(move="<A1>", trace="so I will play A2")
    assert verdict_for(concludes_a2, _decided("<A2>")) == (ConsistencyVerdict.INCONSISTENT, "<A2>")

    # how the judge wrote the move is not a disagreement about which move it is
    assert verdict_for(move, _decided("a1")) == (ConsistencyVerdict.CONSISTENT, "<A1>")

    # a trace that never decided carries no evidence either way, and a judge that named a non-move is a
    # judge failure -- both stay out of the rate instead of being scored as agreement or disagreement
    assert verdict_for(move, TraceConclusion(status="no_conclusion", move="", evidence=""))[0] == ConsistencyVerdict.NO_CONCLUSION
    assert verdict_for(move, TraceConclusion(status="ambiguous", move="", evidence=""))[0] == ConsistencyVerdict.AMBIGUOUS
    assert verdict_for(move, _decided("<Z9>"))[0] == ConsistencyVerdict.UNMATCHED
    assert verdict_for(move, TraceConclusion(status="decided", move="<A1>", evidence="invented quote"))[0] == ConsistencyVerdict.INVALID_EVIDENCE


def test_the_judge_is_shown_the_position_and_the_trace_but_not_the_move_played():
    move = _traced(move="<A3>", optimal=("<A1>",))
    prompt = consistency_prompt(move)
    assert move.observation in prompt.user and all(legal in prompt.user for legal in LEGAL)
    assert move.trace is not None and move.trace in prompt.user
    assert "actually played" not in prompt.user and "solver" not in prompt.user


def test_a_slip_is_graded_by_what_it_cost():
    assert slip_kind(_traced(move="<A2>", optimal=("<A1>",)), "<A1>") == SlipKind.COSTLY  # concluded right, played wrong
    assert slip_kind(_traced(move="<A1>", optimal=("<A1>",)), "<A2>") == SlipKind.LUCKY  # concluded wrong, played right
    assert slip_kind(_traced(move="<A1>", optimal=("<A1>", "<A2>")), "<A2>") == SlipKind.HARMLESS
    assert slip_kind(_traced(move="<A2>", optimal=("<A3>",)), "<A1>") == SlipKind.MOOT


# --- the pass ------------------------------------------------------------------------------------
def test_the_pass_stores_a_verdict_per_move_and_resumes_instead_of_re_paying(tmp_path):
    # distinct positions, so each move is a distinct prompt and therefore a distinct judge call
    moves = [_traced(1, move="<A1>", observation="board 1"), _traced(2, move="<A2>", observation="board 2")]
    judge, generator = _judge(_decided("<A1>"))
    store = ConsistencyStore("exp", tmp_path / "consistency.jsonl")
    cache = ResponseCache(tmp_path / "responses")

    stats = asyncio.run(run_consistency(judge, moves, store, cache, progress=False))
    assert (stats.n, stats.n_parsed, stats.n_failed) == (2, 2, 0) and len(generator.calls) == 2
    verdicts = {record.move_uid: record.verdict for record in store}
    assert verdicts[moves[0].uid] == ConsistencyVerdict.CONSISTENT and verdicts[moves[1].uid] == ConsistencyVerdict.INCONSISTENT

    # the same judge over the same moves: nothing is pending, so the model is not called again
    again = asyncio.run(run_consistency(judge, moves, store, cache, progress=False))
    assert again.n == 0 and len(generator.calls) == 2 and len(store) == 2

    # resume is per store, the response cache is per prompt: a fresh store still costs no calls
    fresh = ConsistencyStore("exp", tmp_path / "other.jsonl")
    cached = asyncio.run(run_consistency(judge, moves, fresh, cache, progress=False))
    assert cached.n_cached == 2 and len(generator.calls) == 2 and len(fresh) == 2


def test_a_failing_judge_costs_one_move_not_the_pass(tmp_path):
    class _Broken(_StubGenerator):
        async def generate(self, model_config, system, messages, output_schema=None):  # type: ignore[no-untyped-def]
            if not self.calls:
                self.calls.append(messages[0].content)
                raise RuntimeError("provider exploded")
            return await super().generate(model_config, system, messages, output_schema)

    generator = _Broken(_decided("<A1>"))
    judge = Judge(generator, MODEL, "consistency:test")
    store = ConsistencyStore("exp", tmp_path / "consistency.jsonl")

    stats = asyncio.run(run_consistency(judge, [_traced(1), _traced(2)], store, ResponseCache(tmp_path / "responses"), progress=False))
    assert stats.n == 2 and stats.n_failed == 1 and stats.n_parsed == 1
    assert len(store) == 1 and "RuntimeError" in stats.errors[0]  # the failed move simply has no verdict


def test_the_store_round_trips_and_keeps_both_judges_of_one_move(tmp_path):
    store = ConsistencyStore("exp", tmp_path / "consistency.jsonl")
    store.add(ConsistencyRecord("uid-1", "a|v1", ConsistencyVerdict.INCONSISTENT, "<A2>", "<A1>", "play A1"))
    store.add(ConsistencyRecord("uid-1", "b|v1", ConsistencyVerdict.CONSISTENT, "<A2>", "<A2>", "play A2"))

    assert store.double_annotated() == ["uid-1"] and store.annotators() == ["a|v1", "b|v1"]
    assert store.by_move("a|v1")["uid-1"].concluded_move == "<A1>"
    reloaded = ConsistencyStore("exp", store.path)
    assert {record.key for record in reloaded} == {record.key for record in store}
    assert reloaded.by_move("a|v1")["uid-1"].verdict == ConsistencyVerdict.INCONSISTENT


# --- the report ----------------------------------------------------------------------------------
def test_only_decided_traces_enter_the_inconsistency_rate():
    moves = [_traced(seq) for seq in range(4)]
    records = [
        ConsistencyRecord(moves[0].uid, ANNOTATOR, ConsistencyVerdict.CONSISTENT, "<A1>", "<A1>"),
        ConsistencyRecord(moves[1].uid, ANNOTATOR, ConsistencyVerdict.INCONSISTENT, "<A1>", "<A2>"),
        ConsistencyRecord(moves[2].uid, ANNOTATOR, ConsistencyVerdict.NO_CONCLUSION, "<A1>"),
        ConsistencyRecord(moves[3].uid, ANNOTATOR, ConsistencyVerdict.UNMATCHED, "<A1>"),
    ]
    report = _report(moves, records)

    assert (report.n_moves, report.n_scored, report.coverage) == (4, 4, 1.0)
    # the denominator is the two gradable verdicts: a trace that never decided cannot disagree with itself
    assert report.inconsistency.n == 2 and report.inconsistency.value == 0.5
    assert report.counts[ConsistencyVerdict.UNMATCHED] == 1  # judge failures stay visible, outside the rate
    assert report.slips[SlipKind.LUCKY] == 1  # concluded <A2>, played the optimal <A1>


def test_forced_positions_are_scored_and_kept_apart_from_real_decisions():
    decision, forced = _traced(1, move="<A2>"), _traced(2, move="<A2>", state_class=StateClass.DONT_CARE)
    records = [
        ConsistencyRecord(decision.uid, ANNOTATOR, ConsistencyVerdict.INCONSISTENT, "<A2>", "<A1>"),
        ConsistencyRecord(forced.uid, ANNOTATOR, ConsistencyVerdict.INCONSISTENT, "<A2>", "<A1>"),
    ]
    report = _report([decision, forced], records)

    outcomes = {outcome.value: counts[ConsistencyVerdict.INCONSISTENT] for outcome, counts in report.by_outcome.items()}
    assert outcomes == {"suboptimal": 1, "non_decision": 1}
    # where every legal move is optimal, fumbling the conclusion cannot cost anything -- that is the baseline
    assert report.slips[SlipKind.COSTLY] == 1 and report.slips[SlipKind.HARMLESS] == 1


def test_unjudged_moves_lower_coverage_rather_than_the_rate():
    moves = [_traced(seq) for seq in range(3)]
    report = _report(moves, [ConsistencyRecord(moves[0].uid, ANNOTATOR, ConsistencyVerdict.CONSISTENT, "<A1>", "<A1>")])
    assert (report.n_moves, report.n_scored) == (3, 1) and report.coverage == 1 / 3
    assert report.inconsistency.value == 0.0 and report.inconsistency.n == 1
