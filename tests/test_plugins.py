"""Player plugins: an installed package registers its players through the `plybench.players` entry point group, so every
PlyBench in the environment knows them. Here the entry points are fakes; nothing is installed."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from plyllm import LLM, LLMConfig

from plybench.app import PlyBench
from plybench.configs.player_params import PlayerParams
from plybench.player.plugins import PLAYER_PLUGINS
from plybench.player.simple.random_player import RandomPlayer
from plybench.player.spec import PlayerSpec
from plybench.registry import Registry


@dataclass(frozen=True, eq=True)
class _AgentParams(PlayerParams):
    label: str = ""

    @classmethod
    def from_string(cls, params_string: str) -> _AgentParams:
        return cls(params_string)

    def to_string(self) -> str:
        return self.label

    @property
    def path_suffix(self) -> str:
        return self.label


@dataclass
class _EntryPoint:
    # the part of importlib.metadata.EntryPoint the loader uses
    name: str
    register: Callable[[Registry, LLM], None]
    value: str = "fake.module:register"

    def load(self) -> Callable[[Registry, LLM], None]:
        return self.register


def _install(monkeypatch: pytest.MonkeyPatch, *plugins: _EntryPoint) -> None:
    # what importlib.metadata.entry_points would list if packages with these plugins were installed
    def listed(group: str) -> list[_EntryPoint]:
        return list(plugins) if group == PLAYER_PLUGINS else []

    monkeypatch.setattr(importlib.metadata, "entry_points", listed)


def _register_agent(seen: list[LLM]) -> Callable[[Registry, LLM], None]:
    def register(registry: Registry, llm: LLM) -> None:
        seen.append(llm)
        registry.register_player(PlayerSpec("agent", _AgentParams, lambda game, cfg, pid: RandomPlayer(cfg, pid)))

    return register


def test_an_installed_plugin_registers_its_players_with_the_llm(monkeypatch):
    seen: list[LLM] = []
    _install(monkeypatch, _EntryPoint("agent", _register_agent(seen)))

    op = PlyBench(LLMConfig())

    assert op.registry.player_config("agent:memory=on").params == _AgentParams("memory=on")
    assert seen == [op.llm]


def test_a_plugin_that_fails_names_itself(monkeypatch):
    def broken(registry: Registry, llm: LLM) -> None:
        raise ImportError("no module named 'missing'")

    _install(monkeypatch, _EntryPoint("broken", broken))

    with pytest.raises(RuntimeError, match="Player plugin 'broken' .* failed to register: no module named 'missing'"):
        PlyBench(LLMConfig())


def test_a_plugin_cannot_take_a_builtin_key(monkeypatch):
    def takes_random(registry: Registry, llm: LLM) -> None:
        registry.register_player(PlayerSpec("random", _AgentParams, lambda game, cfg, pid: RandomPlayer(cfg, pid)))

    _install(monkeypatch, _EntryPoint("thief", takes_random))

    with pytest.raises(ValueError, match=r"Player plugin 'thief' .* already taken: \['random'\]"):
        PlyBench(LLMConfig())
