"""Phase 2 core tests: player-config parsing via the registry, the per-player tracker registry,
and GameTracker round-trip. Uses an instance-scoped Registry (no globals)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Never

import numpy as np
import pytest

from plybench.analysis.usage import game_usage
from plybench.common.enums import GameResults, MoveFailure
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.llm_interface import LLMAction
from plybench.player.player import PlayerIdentifier, PlayerOutput
from plybench.player.spec import PlayerSpec
from plybench.registry import Registry
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.llm_call import LLMCall
from plybench.trackers.player_tracker import PlayerTracker


# --- stubs -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class _StubParams(PlayerParams):
    label: str = "x"

    @classmethod
    def from_string(cls, params_string: str) -> "_StubParams":
        return cls(params_string or "x")

    def to_string(self) -> str:
        return self.label

    @property
    def path_suffix(self) -> str:
        return self.label


class _LLMTracker(PlayerTracker):
    def record(self, player_output: PlayerOutput) -> dict:
        return {"reasoning_trace": player_output.reasoning_trace} if player_output.reasoning_trace else {}


class _PassThroughTracker(PlayerTracker):
    # records whatever the player put in its data, as an agent's tracker may
    def record(self, player_output: PlayerOutput) -> dict:
        return dict(player_output.data or {})


class _StubLLMAction(InterfaceAction):
    def __init__(self, string: str) -> None:
        self._string = string

    @staticmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> _StubLLMAction:
        return _StubLLMAction(action.string)

    def to_openspiel(self) -> OpenSpielAction:
        return OpenSpielAction(0, self._string)

    def to_llm(self) -> LLMAction:
        return LLMAction(self._string)


class _StubObservation(InterfaceObservation):
    def __init__(self) -> None:
        pass

    @staticmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> _StubObservation:
        return _StubObservation()

    def __str__(self) -> str:
        return "OBS"


def _unused_builder(game: TurnBasedGame, cfg: PlayerConfig, pid: PlayerIdentifier) -> Never:
    raise AssertionError("These tracker tests do not build players")


registry = Registry()
# build is unused in these tests (they drive GameTracker directly).
registry.register_player(PlayerSpec("stub", _StubParams, _unused_builder))
registry.register_player(PlayerSpec("ai", _StubParams, _unused_builder, tracker=_LLMTracker()))
registry.register_player(PlayerSpec("agent", _StubParams, _unused_builder, tracker=_PassThroughTracker()))


# --- tests -----------------------------------------------------------------------------------
def test_player_config_string_round_trip():
    cfg = PlayerConfig("stub", _StubParams("a"))
    assert cfg.to_string() == "stub:a"
    assert registry.player_config("stub:a") == cfg
    assert cfg.path == "stub_a"
    # hash is a stable 12-char hex digest of the serialization, distinct per config
    assert cfg.hash == PlayerConfig("stub", _StubParams("a")).hash
    assert len(cfg.hash) == 12 and all(c in "0123456789abcdef" for c in cfg.hash)
    assert cfg.hash != PlayerConfig("stub", _StubParams("b")).hash


def test_llm_tracker_records_tokens_and_data():
    i = PlayerConfig("ai", _StubParams("i"))
    o = PlayerConfig("stub", _StubParams("o"))
    tracker = GameTracker(1, i, o, {})

    llm_out = PlayerOutput(action=_StubLLMAction("<a1>"), input_tokens=11, output_tokens=22, reasoning_tokens=7, reasoning_trace="because")
    tracker.add_move(i, _StubObservation(), llm_out, "STATE", registry)

    step = tracker.steps[-1]
    assert (step.input_tokens, step.output_tokens, step.reasoning_tokens) == (11, 22, 7)
    assert step.data == {"reasoning_trace": "because"}
    assert step.move == "<a1>"


def test_tokens_recorded_generically_but_noop_tracker_adds_no_data():
    o = PlayerConfig("stub", _StubParams("o"))  # no tracker registered for 'stub'
    tracker = GameTracker(1, PlayerConfig("ai", _StubParams("i")), o, {})

    out = PlayerOutput(action=_StubLLMAction("<b2>"), input_tokens=5, output_tokens=9, reasoning_tokens=3)
    tracker.add_move(o, _StubObservation(), out, "STATE", registry)

    step = tracker.steps[-1]
    assert (step.input_tokens, step.output_tokens, step.reasoning_tokens) == (5, 9, 3)
    assert step.data is None


def test_simple_bot_step_is_minimal():
    o = PlayerConfig("stub", _StubParams("o"))
    tracker = GameTracker(1, PlayerConfig("ai", _StubParams("i")), o, {})
    tracker.add_move(o, _StubObservation(), PlayerOutput(action=_StubLLMAction("<b2>")), "STATE", registry)
    assert set(tracker.steps[-1].to_dict()) == {"seq", "player_name", "player_hash", "serialized_state", "observation", "move"}


def test_fail_move_records_its_failure():
    i = PlayerConfig("ai", _StubParams("i"))
    tracker = GameTracker(1, i, PlayerConfig("stub", _StubParams("o")), {})
    tracker.add_move(i, _StubObservation(), PlayerOutput(action=None, failure=MoveFailure.ILLEGAL, failure_detail="<Z9>"), "STATE", registry)
    assert tracker.steps[-1].move == "FAIL: Illegal action selected (<Z9>)"


def test_a_failure_without_a_reason_is_recorded_as_no_action():
    i = PlayerConfig("ai", _StubParams("i"))
    tracker = GameTracker(1, i, PlayerConfig("stub", _StubParams("o")), {})
    tracker.add_move(i, _StubObservation(), PlayerOutput(action=None), "STATE", registry)
    assert tracker.steps[-1].move == "FAIL: No action ()"


@pytest.mark.parametrize(
    ("move", "failure"),
    [
        # recorded before MoveFailure existed, so results already on disk parse the same way
        ("FAIL: Wrong action format ()", MoveFailure.MALFORMED),
        ("FAIL: Illegal action selected (<Action: <C1R1>>)", MoveFailure.ILLEGAL),
        ("FAIL: Refusal (Model gemini-4-argon refused the request: prompt blocked (OTHER))", MoveFailure.REFUSAL),
        ("FAIL: Prompt blocked (Provider blocked the prompt to model gemini-4-argon: prompt blocked (OTHER))", MoveFailure.PROMPT_BLOCKED),
        ("FAIL: None", None),
        ("<C4>", None),
    ],
)
def test_a_recorded_move_gives_back_its_failure(move: str, failure: MoveFailure | None):
    assert MoveFailure.from_move(move) is failure


@pytest.mark.parametrize("failure", list(MoveFailure))
def test_every_failure_round_trips_through_its_move(failure: MoveFailure):
    assert MoveFailure.from_move(failure.to_move("detail (with brackets)")) is failure


def test_game_tracker_round_trip():
    i = PlayerConfig("ai", _StubParams("i"))
    o = PlayerConfig("stub", _StubParams("o"))
    tracker = GameTracker(3, i, o, {"seed": 1}, other_params={"k": "v"})

    tracker.add_move(i, _StubObservation(), PlayerOutput(action=_StubLLMAction("<a>"), input_tokens=4, output_tokens=6), "S1", registry)
    tracker.add_move(o, _StubObservation(), PlayerOutput(action=_StubLLMAction("<b>")), "S2", registry)
    tracker.end_game(GameResults.WIN, _StubObservation())

    data = tracker.to_dict()
    restored = GameTracker.from_dict(data, registry)

    assert restored.to_dict() == data
    assert restored.i_player == i and restored.o_player == o
    assert restored.get_result(i) == GameResults.WIN
    assert restored.get_result(o) == GameResults.LOSS
    assert (restored.steps[0].input_tokens, restored.steps[0].output_tokens) == (4, 6)
    assert restored.steps[1].input_tokens is None and restored.steps[1].output_tokens is None


def test_player_config_requires_registered_key():
    with pytest.raises(ValueError):
        registry.player_config("unregistered:foo")


@pytest.mark.parametrize("data", [{"pair": (1, 2)}, {"counts": {3: 1}}, {"n": np.int64(3)}, {"empty": float("nan")}])
def test_step_data_that_json_would_change_is_refused_when_the_move_is_recorded(data: dict[str, Any]):
    agent = PlayerConfig("agent", _StubParams("i"))
    tracker = GameTracker(1, agent, PlayerConfig("stub", _StubParams("o")), {})

    with pytest.raises(ValueError, match="step data recorded for agent"):
        tracker.add_move(agent, _StubObservation(), PlayerOutput(action=_StubLLMAction("<a>"), data=data), "STATE", registry)
    assert tracker.steps == []


def test_step_data_of_plain_json_types_is_recorded():
    agent = PlayerConfig("agent", _StubParams("i"))
    tracker = GameTracker(1, agent, PlayerConfig("stub", _StubParams("o")), {})
    data = {"key": "x..", "context": None, "counts": {"a1": 2}, "random": False, "path": [1, 2.5]}

    tracker.add_move(agent, _StubObservation(), PlayerOutput(action=_StubLLMAction("<a>"), data=data), "STATE", registry)

    assert tracker.steps[-1].data == data


def _call(role: str, input_tokens: int, output_tokens: int) -> LLMCall:
    return LLMCall(role, "model-x", "system", "prompt", "", "answer", input_tokens, output_tokens, 1, seed=7)


def test_a_move_of_several_calls_records_each_and_their_token_totals():
    agent = PlayerConfig("agent", _StubParams("i"))
    tracker = GameTracker(1, agent, PlayerConfig("stub", _StubParams("o")), {})
    output = PlayerOutput(action=_StubLLMAction("<a>"))
    output.add_call(_call("plan", 10, 5))
    output.add_call(_call("act", 20, 8))

    tracker.add_move(agent, _StubObservation(), output, "STATE", registry)

    step = tracker.steps[-1]
    assert [call.role for call in step.calls] == ["plan", "act"]
    assert (step.input_tokens, step.output_tokens, step.reasoning_tokens) == (30, 13, 2)
    restored = GameTracker.from_dict(json.loads(json.dumps(tracker.to_dict())), registry)
    assert restored.steps[-1].calls == step.calls
    # a move without its own calls stores none
    plain = GameTracker(1, agent, PlayerConfig("stub", _StubParams("o")), {})
    plain.add_move(agent, _StubObservation(), PlayerOutput(action=_StubLLMAction("<a>")), "STATE", registry)
    assert "calls" not in plain.to_dict()["steps"][0]


def test_usage_counts_the_calls_of_a_move_not_the_move():
    agent = PlayerConfig("agent", _StubParams("i"))
    tracker = GameTracker(1, agent, PlayerConfig("stub", _StubParams("o")), {})
    several = PlayerOutput(action=_StubLLMAction("<a>"))
    for role in ("plan", "act", "check"):
        several.add_call(_call(role, 1, 1))
    tracker.add_move(agent, _StubObservation(), several, "S1", registry)
    tracker.add_move(agent, _StubObservation(), PlayerOutput(action=_StubLLMAction("<b>"), input_tokens=4, output_tokens=2), "S2", registry)

    usage = game_usage(tracker, agent)

    assert usage.calls == 4
    assert (usage.tokens.input_tokens, usage.tokens.output_tokens) == (7, 5)
