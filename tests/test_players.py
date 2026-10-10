"""Phase 4: players build via the registry and play end-to-end; config strings round-trip; the LLM
player is exercised with a stub LLM (no network) and its tracker records extras."""

from __future__ import annotations

import asyncio

import pytest
from plyllm import FailureKind, LLMCallError, LLMConfig, LLMResponse, LLMTokens, Provider
from plyllm.response import OutputText, ReasoningTrace

from plybench.app import PlyBench
from plybench.common.enums import GameResults
from plybench.player.llm_player import LLMPlayer, LLMPlayerTracker
from plybench.player.output_strategies import StructuredOutputStrategy
from plybench.player.player import PlayerOutput

# a full registry (games + players); LLMConfig() has no providers so this is fully offline
op = PlyBench(LLMConfig())
registry = op.registry


def _tic_tac_toe():
    return registry.build_engine(registry.game_config("tic_tac_toe:"))


def _play(i_config: str, o_config: str):
    engine = _tic_tac_toe()
    players = (
        registry.build_player(engine.game, registry.player_config(i_config), "i"),
        registry.build_player(engine.game, registry.player_config(o_config), "o"),
    )
    # engine built via registry.build_engine already has engine.trackers set to the registry
    return asyncio.run(engine.play(players))


# --- config round-trips ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "config_string",
    [
        "human:",
        "random:distribution=uniform",
        "optimal:stochastic=True",
        "mcts:max_simulations=200,rollout_count=1,uct_c=2.0",
        "llm:actions:structured:openai:gpt-5.4:thinking_enabled=True,reasoning_effort=high",
    ],
)
def test_player_config_round_trips(config_string: str):
    assert registry.player_config(config_string).to_string() == config_string


def test_player_paths():
    assert registry.player_config("random:distribution=uniform").path == "random_uniform"
    assert registry.player_config("optimal:stochastic=True").path == "optimal_stochastic"
    assert registry.player_config("human:").path == "human"
    assert registry.player_config("llm:actions:structured:openai:gpt-5.4:").path == "llm_actions_structured_openai_gpt-5.4"


def test_all_five_players_registered():
    assert set(registry.player_keys()) == {"human", "random", "mcts", "optimal", "llm"}


# --- end-to-end games with bots --------------------------------------------------------------
def test_random_vs_random_plays_to_terminal():
    tracker = _play("random:distribution=uniform", "random:distribution=normal")
    assert tracker.ending is not None
    assert tracker.ending.result in {GameResults.WIN, GameResults.LOSS, GameResults.DRAW}
    assert all(not step.move.startswith("FAIL") for step in tracker.steps)


def test_mcts_vs_random_plays_to_terminal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MCTS is fine, but keep any path-builder side effects out of the repo
    tracker = _play("mcts:max_simulations=50,rollout_count=1,uct_c=2.0", "random:distribution=uniform")
    assert tracker.ending is not None
    assert all(not step.move.startswith("FAIL") for step in tracker.steps)


def test_optimal_never_loses_to_random(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # OptimalPlayer solves + caches to ./cache; isolate it
    optimal = registry.player_config("optimal:stochastic=True")
    tracker = _play(optimal.to_string(), "random:distribution=uniform")
    # optimal is the i-player; against a random opponent it must never lose on tic-tac-toe
    assert tracker.get_result(optimal) in {GameResults.WIN, GameResults.DRAW}


# --- LLM player with a stub LLM --------------------------------------------------------------
class _StubLLM:
    async def generate(self, model_config, system, messages, output_schema=None):
        return LLMResponse(
            provider=model_config.provider,
            model_string=model_config.model_name,
            tokens=LLMTokens(input_tokens=10, output_tokens=20, reasoning_tokens=5),
            items=[ReasoningTrace(["thinking about C1R1"]), OutputText(['{"action": "C1R1"}'])],
            output_text='{"action": "C1R1"}',
            structured_output_type=output_schema,
        )


def _first_move(engine):
    engine.reset()
    pid = engine.game.get_player()
    os_obs = engine.game.get_observation(pid)
    os_moves = engine.game.get_legal_moves(pid)
    observation = engine.observation_class.from_openspiel(os_obs, engine.interface_transformer)
    moves = [engine.action_class.from_openspiel(m, engine.interface_transformer) for m in os_moves]
    return observation, moves


def test_llm_player_produces_move_and_tokens_with_stub_llm():
    engine = _tic_tac_toe()
    llm_player = LLMPlayer(
        registry.player_config("llm:actions:structured:openai:gpt-5.4:"),
        StructuredOutputStrategy(),
        _StubLLM(),  # type: ignore[arg-type]
        "i",
    )
    llm_player.initialize_policy(engine.game, engine.prompt_adapter)
    observation, moves = _first_move(engine)

    out: PlayerOutput = asyncio.run(llm_player(engine.game, observation, moves))

    assert out.action is not None and out.action.to_llm().string == "<C1R1>"
    assert (out.input_tokens, out.output_tokens, out.reasoning_tokens) == (10, 20, 5)
    assert out.reasoning_trace == "thinking about C1R1"

    # the LLM tracker persists the trace/output extras into GameStep.data
    data = LLMPlayerTracker().record(out)
    assert data["reasoning_trace"] == "thinking about C1R1"
    assert data["full_output"] == '{"action": "C1R1"}'


class _FailingLLM:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def generate(self, model_config, system, messages, output_schema=None):
        raise self._error


def _answer(text: str) -> LLMResponse:
    return LLMResponse.from_parts(Provider.OPENAI, "gpt-5.4", text, ["thinking"], LLMTokens(input_tokens=10, output_tokens=20, reasoning_tokens=5), None)


def _llm_move(llm) -> PlayerOutput:
    engine = _tic_tac_toe()
    llm_player = LLMPlayer(registry.player_config("llm:actions:structured:openai:gpt-5.4:"), StructuredOutputStrategy(), llm, "i")
    llm_player.initialize_policy(engine.game, engine.prompt_adapter)
    observation, moves = _first_move(engine)
    return asyncio.run(llm_player(engine.game, observation, moves))


@pytest.mark.parametrize(
    ("kind", "reason_prefix"),
    [(FailureKind.UNPARSEABLE, "Wrong action format ("), (FailureKind.REFUSAL, "Refusal (")],
)
def test_an_unplayable_answer_is_a_failed_move_not_a_crash(kind, reason_prefix):
    """Raising aborted the run, and the resume replayed the game until the model happened to answer, so
    these failures never reached the results. They are the model's move and are recorded like one."""
    out = _llm_move(_FailingLLM(LLMCallError(kind, "no", _answer('{"action": 4'))))

    assert out.action is None
    assert out.failure_reason is not None and out.failure_reason.startswith(reason_prefix)
    assert out.full_output == '{"action": 4'
    assert (out.input_tokens, out.output_tokens, out.reasoning_tokens) == (10, 20, 5), "the failed call's cost is kept"


def test_a_transport_failure_still_aborts_the_move():
    # says nothing about the model, so the game is better replayed than scored
    with pytest.raises(LLMCallError):
        _llm_move(_FailingLLM(LLMCallError(FailureKind.TIMEOUT, "slow")))
