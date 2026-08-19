from __future__ import annotations

import json
import math
from typing import Any

import pyspiel as sp

from plybench.common.enums import StateClass
from plybench.common.serializable import Saveable, Serializable
from plybench.core.game import TurnBasedGame, TurnBasedState

# a depth-limited search's memo table, keyed by (player to move, position, remaining plies)
type DepthMemo = dict[tuple[int, str, int], float]


class AVQ(Serializable):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AVQ:
        A = [int(a) for a in data["A"]]
        V = float(data["V"])
        Q = {int(k): float(v) for k, v in data["Q"].items()}
        return cls(A, V, Q)

    def __init__(self, A: list[int], V: float, Q: dict[int, float]) -> None:
        self._A = A  # optimal actions in this state
        self._V = V  # value of the state
        self._Q = Q  # Q-values of the actions available in this state

    def to_dict(self) -> dict[str, Any]:
        return {"A": self._A, "V": self._V, "Q": self._Q}

    def A(self) -> list[int]:
        return self._A

    def check_optimal(self, a: int) -> bool:
        return a in self._A

    def is_trivial(self, actions: list[int]) -> bool:
        return set(self._A) == set(actions)

    def classify_state(self, actions: list[int], loss_value: float) -> StateClass:
        if not self.is_trivial(actions):
            return StateClass.DECISION
        return StateClass.LOST if self._V == loss_value else StateClass.DONT_CARE

    def V(self) -> float:
        return self._V

    def Q(self, a: int) -> float:
        if a not in self._Q:
            raise ValueError(f"Q-value not found for action {a}")
        return self._Q[a]


class AVQCache(Saveable[[], str, []]):
    @classmethod
    def from_dict(cls, data: dict[Any, dict[str, Any]]) -> AVQCache:
        cache = {int(player): {str(state): AVQ.from_dict(avq) for state, avq in player_cache.items()} for player, player_cache in data.items()}
        return cls(cache)

    @classmethod
    def load(cls, filepath: str) -> AVQCache:
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))

    def __init__(self, cache: dict[int, dict[str, AVQ]] | None = None) -> None:
        self.cache: dict[int, dict[str, AVQ]] = cache if cache is not None else {}

    def __getitem__(self, ps: tuple[int, str]) -> AVQ | None:
        player, state = ps
        return self.cache.get(player, {}).get(state, None)

    def __setitem__(self, ps: tuple[int, str], avq: AVQ) -> None:
        player, state = ps
        self.cache.setdefault(player, {})[state] = avq

    def to_dict(self) -> dict[str, Any]:
        return {str(player): {state: avq.to_dict() for state, avq in player_cache.items()} for player, player_cache in self.cache.items()}


def solve_game(game: TurnBasedGame) -> AVQCache:
    # clone the state (reset would also reset the random seed)
    state = game.state.clone()
    cache = AVQCache()
    _minimax(state, cache)
    return cache


def _minimax(state: sp.State, cache: AVQCache) -> float:
    if state.is_terminal():
        v0, _ = TurnBasedState.get_rewards(state)
        os0 = TurnBasedState.get_observation(state, 0).state
        os1 = TurnBasedState.get_observation(state, 1).state
        cache[0, os0] = AVQ(A=[], V=v0, Q={})
        cache[1, os1] = AVQ(A=[], V=-v0, Q={})
        return v0

    player = TurnBasedState.get_player(state)
    observation = TurnBasedState.get_observation(state, player)

    entry = cache[player, observation.state]
    if entry is not None:
        v_cached = entry.V()
        return v_cached if player == 0 else -v_cached

    legal_actions = TurnBasedState.get_legal_moves(state, player)

    v_star = -math.inf
    a_star: list[int] = []
    q_star: dict[int, float] = {}

    for a in legal_actions:
        v0 = _minimax(TurnBasedState.child(state, a.number), cache)
        v = v0 if player == 0 else -v0
        q_star[a.number] = v

        if v > v_star:
            v_star = v
            a_star = [a.number]
        elif abs(v - v_star) < 1e-12:
            a_star.append(a.number)

    cache[player, observation.state] = AVQ(A=a_star, V=v_star, Q=q_star)
    return v_star if player == 0 else -v_star


def depth_limited_value(state: sp.State, depth: int, memo: DepthMemo) -> float:
    if TurnBasedState.is_terminal(state):
        return TurnBasedState.get_rewards(state)[0]
    if depth <= 0:
        return 0.0

    player = TurnBasedState.get_player(state)
    # the observation string alone, not TurnBasedState.get_observation: the action histories that carries
    # are not part of the key, and building them per node per depth is most of the cost of the search
    key = (player, state.observation_string(player), depth)
    cached = memo.get(key)
    if cached is not None:
        return cached

    values = [depth_limited_value(TurnBasedState.child(state, action), depth - 1, memo) for action in state.legal_actions()]
    # player 0's units, like _minimax's return value and unlike the AVQ it stores, which is the mover's. So
    # player 1 minimises here rather than maximising, which holds only because these games are zero sum
    value = max(values) if player == 0 else min(values)
    memo[key] = value
    return value
