"""What a run spent. The ledger is deliberately a record of calls *made*, not of artefacts on disk: a
resumed pass serves most moves from cache and must report that it paid nothing for them."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from plybench.analysis.errors.consistency.verdicts import TraceConclusion
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.judge.cost import CostLedger
from plybench.analysis.errors.judge.runner import ResponseCache, RunStats
from plybench.analysis.errors.moves import MatchupId, TracedMove
from plybench.analysis.errors.pipeline import PassOptions, Pipeline
from plybench.analysis.errors.stores import AnalysisStores
from plybench.analysis.stats.moves import MoveRecord
from plybench.app import PlyBench
from plybench.common.enums import StateClass
from plybench.llm import LLMCallOptions, LLMConfig, LLMMessage, LLMModel, LLMResponse, LLMTokens, ModelConfig, OutputText, Provider

op = PlyBench(LLMConfig())

LEGAL = ("<A1>", "<A2>", "<A3>")
MATCHUP = MatchupId("exp", "tic_tac_toe:", "llm:player", "random:")
MODEL = ModelConfig(Provider.OPENAI, "judge-model", LLMCallOptions())
FREE_MODEL = ModelConfig(Provider.OPENAI, "local-model", LLMCallOptions())


class _StubModel(LLMModel):
    def extract_params(self, options: LLMCallOptions) -> None:
        return None


class _StubPricer:
    """$1 per 1M input, $2 per 1M output for the judge model; the local model is registered free, which
    the ledger has to report as unpriced rather than as a $0.00 bill."""

    _MODELS = {"judge-model": _StubModel("judge-model", "judge-model", input_cost=1.0, output_cost=2.0), "local-model": _StubModel("local-model", "local-model")}

    def resolve_model(self, provider: Provider, model_name: str) -> LLMModel:
        if model_name not in self._MODELS:
            raise ValueError(model_name)
        return self._MODELS[model_name]

    def calculate_cost(self, model_config: ModelConfig, tokens: LLMTokens) -> float:
        return self.resolve_model(model_config.provider, model_config.model_name).cost(tokens)


def _stats(n: int = 1, cached: int = 0, tokens: LLMTokens | None = None) -> RunStats:
    return RunStats(n=n, n_parsed=n, n_cached=cached, n_failed=0, tokens=tokens if tokens is not None else LLMTokens(), errors=())


def _traced(seq: int = 1, move: str = "<A1>") -> TracedMove:
    record = MoveRecord(StateClass.DECISION, move == "<A1>", 0.0, None, 100, None, len(LEGAL), 1)
    # the observation varies by move: identical prompts would be one cache entry, which is the response
    # cache doing its job but not what a two-move pass is meant to exercise here
    return TracedMove(MATCHUP, 1, seq, record, "so I will play A1", f"board {seq}", move, LEGAL, ("<A1>",))


def _funnel(moves: list[TracedMove]) -> FunnelResult:
    return FunnelResult("exp", op.registry.game_config("tic_tac_toe:"), op.registry.player_config("random:distribution=uniform"), moves)


class _StubGenerator:
    def __init__(self, tokens: LLMTokens) -> None:
        self.tokens = tokens
        self.calls = 0

    async def generate(
        self,
        model_config: ModelConfig,
        system: LLMMessage,
        messages: list[LLMMessage],
        output_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        reply = TraceConclusion(status="decided", move="<A1>", evidence="so I will play A1")
        return LLMResponse(Provider.OPENAI, "judge-model", self.tokens, [OutputText([""])], reply.model_dump_json(), TraceConclusion)


# --- the ledger -----------------------------------------------------------------------------------
def test_calls_of_one_step_accumulate_into_a_single_priced_row():
    ledger = CostLedger(_StubPricer())
    ledger.record("consistency", MODEL, _stats(tokens=LLMTokens(1_000_000, 0, 500_000)))
    ledger.record("consistency", MODEL, _stats(tokens=LLMTokens(1_000_000, 0, 500_000)))

    row = ledger.rows()[0]
    assert row.n_calls == 2
    assert row.tokens.input_tokens == 2_000_000
    assert row.cost == 4.0  # 2 in @ $1/M + 1 out @ $2/M


def test_a_cached_call_carries_no_tokens_so_the_bill_is_what_was_paid_now():
    ledger = CostLedger(_StubPricer())
    ledger.record("consistency", MODEL, _stats(n=10, cached=8, tokens=LLMTokens(200_000, 0, 0)))

    total = ledger.total()
    assert total.n_cached == 8
    assert total.cost == 0.2  # the two uncached calls only


def test_an_unpriced_model_reports_no_dollar_figure_rather_than_zero():
    ledger = CostLedger(_StubPricer())
    ledger.record("annotate", FREE_MODEL, _stats(tokens=LLMTokens(1_000_000, 0, 1_000_000)))

    assert ledger.total().cost is None
    assert ledger.total().cost_per_unit is None


def test_cost_per_move_counts_moves_covered_not_calls_made():
    """Induction batches many moves into one call, so pricing a larger run off the call count would
    understate it by the batch size."""
    ledger = CostLedger(_StubPricer())
    ledger.record("induce", MODEL, _stats(n=2, tokens=LLMTokens(1_000_000, 0, 0)), n_units=16)

    assert ledger.total().n_units == 16
    assert ledger.total().cost_per_unit == 1.0 / 16


def test_steps_stay_separate_rows_and_sum_into_the_total():
    ledger = CostLedger(_StubPricer())
    ledger.record("consistency", MODEL, _stats(tokens=LLMTokens(1_000_000, 0, 0)))
    ledger.record("annotate", MODEL, _stats(tokens=LLMTokens(0, 0, 1_000_000)))

    assert [row.step for row in ledger.rows()] == ["consistency", "annotate"]
    assert ledger.total().cost == 3.0
    assert ledger.total().model == "openai:judge-model"


# --- the pipeline feeding it ----------------------------------------------------------------------
def test_a_pass_records_what_it_spent_and_a_resumed_pass_records_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # every store path is cwd-relative, so this keeps the run out of the repo
    generator = _StubGenerator(LLMTokens(1_000_000, 0, 1_000_000))
    funnel = _funnel([_traced(1), _traced(2)])

    def run(ledger: CostLedger) -> None:
        pipeline = Pipeline(generator, MODEL, AnalysisStores(), ResponseCache(tmp_path / "responses"), ledger, PassOptions())
        asyncio.run(pipeline.run("consistency", [funnel]))

    first = CostLedger(_StubPricer())
    run(first)
    assert generator.calls == 2
    assert first.total().n_calls == 2
    assert first.total().cost == 6.0  # 2 calls, each 1M in @ $1 + 1M out @ $2

    resumed = CostLedger(_StubPricer())
    run(resumed)
    assert generator.calls == 2  # the store already holds both verdicts
    assert resumed.total().n_calls == 0
    assert not resumed.rows()


# --- judge configuration -------------------------------------------------------------------------
def test_a_judge_is_configured_in_the_same_syntax_as_a_player():
    """The judge and the players used two different config languages, so `reasoning_effort=high` was a
    flag in one place and a key in the other. One parser now serves both, which is also what lets a
    codebook be named after the judge that induced it."""
    text = "metacentrum:gemma-4:thinking_enabled=True,reasoning_effort=medium"
    config = ModelConfig.from_string(text)

    assert (config.provider, config.model_name) == (Provider.METACENTRUM, "gemma-4")
    assert config.options.thinking_enabled and config.options.reasoning_effort == "medium"
    assert config.to_string() == text, "round-trips, so a recorded judge id can be replayed verbatim"

    bare = ModelConfig.from_string("openai:gpt-5-mini")
    assert bare.to_string() == "openai:gpt-5-mini" and not bare.options.thinking_enabled


def test_the_judge_slug_is_a_usable_filename():
    slug = ModelConfig.from_string("openai:gpt-5.4:reasoning_effort=high").slug
    assert slug == "openai_gpt-5-4_reasoning_effort_high"
    assert not set(slug) & set("./:=, "), "goes into a path, so nothing that would split it or hide a directory"


def test_a_config_that_names_no_real_provider_is_refused():
    for text in ("nosuchprovider:model", "openai", "openai:"):
        try:
            ModelConfig.from_string(text)
        except ValueError:
            continue
        raise AssertionError(f"{text!r} should not parse")


def test_the_player_and_the_judge_agree_on_what_a_model_string_means():
    # the tail of a player config is exactly a judge config, which is the property that makes one parser
    # correct rather than merely convenient
    player = op.registry.player_config("llm:actions:text:openai:gpt-5-nano:thinking_enabled=True,reasoning_effort=low")
    assert ModelConfig.from_string("openai:gpt-5-nano:thinking_enabled=True,reasoning_effort=low") == player.params.model
