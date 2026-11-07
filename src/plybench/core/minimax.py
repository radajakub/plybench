from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

import pyspiel as sp

from plybench.common.enums import GameResults, StateClass
from plybench.common.paths import MinimaxPathBuilder
from plybench.common.serializable import Saveable, Serializable
from plybench.core.game import TurnBasedGame, TurnBasedState


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
    # from a new initial state, whatever state the game is in: a game in progress (with chance, its outcomes so far) would give
    # the values of that subgame only, and load_or_solve keeps them for good. Not game.reset(), which would also reset the random seed
    state = game.game.new_initial_state()
    cache = AVQCache()

    if game.multiplayer():
        raise ValueError(f"No solver for {game.game_name}: only single and two player games are supported")

    # OpenSpiel marks single-player games general-sum, so only a two-player game must be zero-sum
    if game.two_player() and not game.is_zero_sum():
        raise ValueError(f"No solver for {game.game_name}: only zero sum two player games are supported")

    if game.is_perfect_information():
        _minimax(state, cache)
    elif game.single_player():
        _solve_single_player(state, cache)
    else:
        # an equilibrium of a game with hidden information is a mixed policy, so its optimal actions need their own definition
        raise ValueError(f"No solver for {game.game_name}: imperfect information with two players")

    return cache


def load_or_solve(game: TurnBasedGame) -> AVQCache:
    # the solved values of a game, read from the minimax cache, or solved and saved there on first use; shared by the optimal
    # player and any agent that needs the solution (e.g. an oracle critic in plyagents)
    path = MinimaxPathBuilder().cache(game.game_name, game.params).resolve()
    if not path.exists():
        solve_game(game).save(str(path))
    return _read_solved(path)


# once per process: a player is built for every game, and the file does not change under a run; callers share the result and
# only read it. Keyed by the absolute path, so another working directory reads its own file
@functools.cache
def _read_solved(path: Path) -> AVQCache:
    return AVQCache.load(str(path))


def optimal_outcomes(game: TurnBasedGame) -> dict[GameResults, float]:
    # the exact probabilities of a win, a draw and a loss of a single-player game under the solved policy, which picks uniformly among the optimal
    # actions (as optimal:stochastic=True does): every chance outcome followed with its probability, no sampling. A result is the sign of the return,
    # as the engine decides it. With an opponent, the outcome depends on the opponent's policy, so it is not defined here
    if not game.single_player():
        raise ValueError(f"The outcomes of the optimal policy are defined for single-player games only, not {game.game_name}")

    cache = load_or_solve(game)
    outcomes: dict[GameResults, float] = {GameResults.WIN: 0.0, GameResults.DRAW: 0.0, GameResults.LOSS: 0.0}

    def follow(state: sp.State, probability: float) -> None:
        if state.is_terminal():
            reward = state.returns()[0]
            outcomes[GameResults.WIN if reward > 0 else GameResults.LOSS if reward < 0 else GameResults.DRAW] += probability
        elif TurnBasedState.is_chance_node(state):
            for action, p in TurnBasedState.chance_outcomes(state):
                follow(TurnBasedState.child(state, action), probability * p)
        else:
            entry = cache[0, TurnBasedState.get_observation(state, 0).state]
            if entry is None:
                raise ValueError(f"No solved entry for {TurnBasedState.get_observation(state, 0).state!r}")
            for action in entry.A():
                follow(TurnBasedState.child(state, action), probability / len(entry.A()))

    follow(game.game.new_initial_state(), 1.0)

    return outcomes


def _best(q: dict[int, float]) -> AVQ:
    # the optimal actions are those within 1e-12 of the best value, in the order of q
    v_star = -math.inf
    a_star: list[int] = []
    for a, v in q.items():
        if v > v_star:
            v_star = v
            a_star = [a]
        elif abs(v - v_star) < 1e-12:
            a_star.append(a)
    return AVQ(A=a_star, V=v_star, Q=q)


def _minimax(state: sp.State, cache: AVQCache) -> float:
    # negamax with player 0's value as the return; a chance node takes the expectation over its outcomes and has no entry
    if state.is_terminal():
        # the returns, the sum of the rewards over the game: the terminal reward in a game that has only that one
        v0 = state.returns()[0]
        for player in range(TurnBasedState.num_players(state)):
            cache[player, TurnBasedState.get_observation(state, player).state] = AVQ(A=[], V=v0 if player == 0 else -v0, Q={})
        return v0

    if TurnBasedState.is_chance_node(state):
        return sum(p * _minimax(TurnBasedState.child(state, a), cache) for a, p in TurnBasedState.chance_outcomes(state))

    player = TurnBasedState.get_player(state)
    observation = TurnBasedState.get_observation(state, player)

    entry = cache[player, observation.state]
    if entry is not None:
        v_cached = entry.V()
        return v_cached if player == 0 else -v_cached

    q_star: dict[int, float] = {}
    for a in TurnBasedState.get_legal_moves(state, player):
        v0 = _minimax(TurnBasedState.child(state, a.number), cache)
        q_star[a.number] = v0 if player == 0 else -v0

    avq = _best(q_star)
    cache[player, observation.state] = avq
    return avq.V() if player == 0 else -avq.V()


def _solve_single_player(state: sp.State, cache: AVQCache) -> None:
    # one player with hidden information: a decision is taken per information state, so its values are expectations over the histories in it, weighted by
    # their chance probability. An information state only holds histories that passed the earlier ones (perfect recall), so they are decided from the
    # deepest up, and a history's value follows the decisions already taken below it
    decisions: dict[str, list[tuple[float, sp.State]]] = {}
    terminals: dict[str, list[tuple[float, float]]] = {}
    depths: dict[str, int] = {}
    _collect(state, 1.0, decisions, terminals, depths)

    chosen: dict[str, int] = {}
    values: dict[str, float] = {}

    def value(history: sp.State) -> float:
        key = history.history_str()
        if key not in values:
            if history.is_terminal():
                values[key] = history.returns()[0]
            elif TurnBasedState.is_chance_node(history):
                values[key] = sum(p * value(TurnBasedState.child(history, a)) for a, p in TurnBasedState.chance_outcomes(history))
            else:
                values[key] = value(TurnBasedState.child(history, chosen[TurnBasedState.get_observation(history, 0).state]))
        return values[key]

    for info_state in sorted(decisions, key=lambda info_state: depths[info_state], reverse=True):
        histories = decisions[info_state]
        reach = sum(p for p, _ in histories)
        q = {a.number: sum(p * value(TurnBasedState.child(h, a.number)) for p, h in histories) / reach for a in TurnBasedState.get_legal_moves(histories[0][1], 0)}
        avq = _best(q)
        chosen[info_state] = avq.A()[0]
        cache[0, info_state] = avq

    for info_state, outcomes in terminals.items():
        cache[0, info_state] = AVQ(A=[], V=sum(p * v for p, v in outcomes) / sum(p for p, _ in outcomes), Q={})


def _collect(state: sp.State, reach: float, decisions: dict[str, list[tuple[float, sp.State]]], terminals: dict[str, list[tuple[float, float]]], depths: dict[str, int]) -> None:
    # every history of a single-player game with its chance probability, grouped by the player's information state; the depth is the number of its own
    # actions, the same for every history of an information state under perfect recall
    if state.is_terminal():
        terminals.setdefault(TurnBasedState.get_observation(state, 0).state, []).append((reach, state.returns()[0]))
        return
    if TurnBasedState.is_chance_node(state):
        for a, p in TurnBasedState.chance_outcomes(state):
            _collect(TurnBasedState.child(state, a), reach * p, decisions, terminals, depths)
        return

    info_state = TurnBasedState.get_observation(state, 0).state
    depth = sum(1 for step in state.full_history() if step.player == 0)
    if depths.setdefault(info_state, depth) != depth:
        raise ValueError(f"Information state {info_state!r} is reached after different numbers of the player's actions: the game has no perfect recall")
    decisions.setdefault(info_state, []).append((reach, state))
    for a in state.legal_actions():
        _collect(TurnBasedState.child(state, a), reach, decisions, terminals, depths)
