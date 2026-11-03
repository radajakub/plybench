from __future__ import annotations

import importlib.metadata

from plyllm import LLM

from plybench.registry import Registry

# the entry point group through which an installed package (e.g. plyagents) registers its players. Each entry point names a
# function with the signature of register_builtin_players: register(registry, llm)
PLAYER_PLUGINS = "plybench.players"


def register_plugin_players(registry: Registry, llm: LLM) -> None:
    # every installed plugin is registered, so any PlyBench in the environment can build its players and read their records
    for entry_point in sorted(importlib.metadata.entry_points(group=PLAYER_PLUGINS), key=lambda entry_point: entry_point.name):
        before = {key: registry.resolve_player(key) for key in registry.player_keys()}
        try:
            entry_point.load()(registry, llm)
        except Exception as error:
            # a broken plugin stops every PlyBench in the environment, so the error names it
            raise RuntimeError(f"Player plugin {entry_point.name!r} ({entry_point.value}) failed to register: {error}") from error
        replaced = sorted(key for key, spec in before.items() if registry.resolve_player(key) is not spec)
        if replaced:
            raise ValueError(f"Player plugin {entry_point.name!r} ({entry_point.value}) registers keys that are already taken: {replaced}")
