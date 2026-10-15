"""Phase 4: players build via the registry and play end-to-end; config strings round-trip; the LLM
player is exercised with a stub LLM (no network) and its tracker records extras."""

from __future__ import annotations

import asyncio
import fcntl
import json
import threading

import pytest
from plyllm import FailureKind, LLMCallError, LLMConfig, LLMResponse, LLMTokens, Provider
from plyllm.response import OutputText, ReasoningTrace

from plybench.app import PlyBench
from plybench.common.enums import GameResults, MoveFailure
from plybench.player.llm_player import _BLOCK_RETRIES, LLMPlayer, LLMPlayerTracker
from plybench.player.output_strategies import StructuredOutputStrategy
from plybench.player.player import PlayerOutput
from plybench.trackers.prompt_block import PromptBlock

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


def _llm_move(llm, block_log=None) -> PlayerOutput:
    engine = _tic_tac_toe()
    llm_player = LLMPlayer(registry.player_config("llm:actions:structured:openai:gpt-5.4:"), StructuredOutputStrategy(), llm, "i", block_log)
    llm_player.initialize_policy(engine.game, engine.prompt_adapter)
    observation, moves = _first_move(engine)
    return asyncio.run(llm_player(engine.game, observation, moves))


@pytest.mark.parametrize(
    ("kind", "failure"),
    [(FailureKind.UNPARSEABLE, MoveFailure.MALFORMED), (FailureKind.REFUSAL, MoveFailure.REFUSAL), (FailureKind.PROMPT_BLOCKED, MoveFailure.PROMPT_BLOCKED)],
)
def test_an_unplayable_answer_is_a_failed_move_not_a_crash(kind, failure):
    """Raising aborted the run, and the resume replayed the game until the model happened to answer, so
    these failures never reached the results. They are the model's move and are recorded like one."""
    out = _llm_move(_FailingLLM(LLMCallError(kind, "no", _answer('{"action": 4'))))

    assert out.action is None
    assert out.failure is failure
    assert out.full_output == '{"action": 4'
    assert (out.input_tokens, out.output_tokens, out.reasoning_tokens) == (10, 20, 5), "the failed call's cost is kept"


def test_a_transport_failure_still_aborts_the_move():
    # says nothing about the model, so the game is better replayed than scored
    with pytest.raises(LLMCallError):
        _llm_move(_FailingLLM(LLMCallError(FailureKind.TIMEOUT, "slow")))


class _ScriptedLLM:
    # answers each call with the next outcome: an exception is raised, a response is returned
    def __init__(self, *outcomes) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def generate(self, model_config, system, messages, output_schema=None):
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _blocked() -> LLMCallError:
    return LLMCallError(FailureKind.PROMPT_BLOCKED, "prompt blocked (OTHER)", _answer(""))


def _block_records(block_log) -> list[PromptBlock]:
    return [PromptBlock.from_dict(json.loads(line)) for line in block_log.read_text().splitlines()]


def test_a_blocked_prompt_is_sent_again_and_recorded(tmp_path):
    """The same Gemini prompt was blocked in one game and answered in another, so a block is retried
    rather than scored as the model's failure, and kept on record to investigate."""
    llm = _ScriptedLLM(_blocked(), _blocked(), _answer('{"action": "C1R1"}'))
    block_log = tmp_path / "llm_blocks.jsonl"

    out = _llm_move(llm, block_log)

    assert llm.calls == 3
    assert out.action is not None and out.action.to_llm().string == "<C1R1>"
    blocks = _block_records(block_log)
    assert [block.attempt for block in blocks] == [1, 2]
    assert all(block.max_attempts == _BLOCK_RETRIES + 1 for block in blocks)
    assert blocks[0].player == "llm:actions:structured:openai:gpt-5.4:" and blocks[0].reason == "prompt blocked (OTHER)"
    assert blocks[0].input_tokens == 10 and "C1R1" in blocks[0].prompt_message


def test_a_prompt_blocked_on_every_attempt_is_a_failed_move(tmp_path):
    llm = _ScriptedLLM(*[_blocked() for _ in range(_BLOCK_RETRIES + 1)])
    block_log = tmp_path / "llm_blocks.jsonl"

    out = _llm_move(llm, block_log)

    assert llm.calls == _BLOCK_RETRIES + 1
    assert out.action is None and out.failure is MoveFailure.PROMPT_BLOCKED
    assert [block.attempt for block in _block_records(block_log)] == list(range(1, _BLOCK_RETRIES + 2))


def test_a_transport_failure_during_retries_still_aborts_and_keeps_the_blocks(tmp_path):
    llm = _ScriptedLLM(_blocked(), LLMCallError(FailureKind.TIMEOUT, "slow"))
    block_log = tmp_path / "llm_blocks.jsonl"

    with pytest.raises(LLMCallError):
        _llm_move(llm, block_log)

    assert [block.attempt for block in _block_records(block_log)] == [1]


def test_a_refusal_is_not_retried_or_recorded_as_a_block(tmp_path):
    llm = _ScriptedLLM(LLMCallError(FailureKind.REFUSAL, "no", _answer("I won't")), _answer('{"action": "C1R1"}'))
    block_log = tmp_path / "llm_blocks.jsonl"

    out = _llm_move(llm, block_log)

    assert llm.calls == 1 and out.action is None
    assert not block_log.exists()


_BLOCK = PromptBlock("2026-10-15T12:00:00+00:00", "llm:actions:text:gemini:gemini-4-argon:", "prompt blocked (OTHER)", 2, 4, None, "rules", "your move")


def test_a_prompt_block_round_trips():
    assert PromptBlock.from_dict(json.loads(json.dumps(_BLOCK.to_dict()))) == _BLOCK


def test_appending_a_prompt_block_waits_for_another_writer(tmp_path):
    # several benchmark processes share one block log; a writer holding the lock must finish its line first
    block_log = tmp_path / "llm_blocks.jsonl"
    with open(block_log, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        writer = threading.Thread(target=_BLOCK.append, args=(block_log,))
        writer.start()
        writer.join(timeout=0.2)
        assert writer.is_alive() and block_log.read_text() == ""
        fcntl.flock(other, fcntl.LOCK_UN)
    writer.join(timeout=5)

    assert not writer.is_alive()
    assert _block_records(block_log) == [_BLOCK]
