"""Phase 2 core tests: player-config parsing via the registry, the per-player tracker registry,
and GameTracker round-trip. Uses an instance-scoped Registry (no globals)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Never

import pytest

from plybench.common.enums import GameResults
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.llm_interface import LLMAction
from plybench.player.player import PlayerIdentifier, PlayerOutput
from plybench.player.spec import PlayerSpec
from plybench.registry import Registry
from plybench.trackers.game_tracker import GameTracker
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


def test_fail_move_records_failure_reason():
    i = PlayerConfig("ai", _StubParams("i"))
    tracker = GameTracker(1, i, PlayerConfig("stub", _StubParams("o")), {})
    tracker.add_move(i, _StubObservation(), PlayerOutput(action=None, failure_reason="illegal"), "STATE", registry)
    assert tracker.steps[-1].move == "FAIL: illegal"


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
