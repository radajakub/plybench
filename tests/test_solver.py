"""The solver on games with chance, with one player, and with hidden information, against brute force and hand-computed values."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pyspiel as sp
import pytest

from plybench.core.game import TurnBasedGame, TurnBasedState
from plybench.core.minimax import AVQCache, solve_game


class _Game(TurnBasedGame):
    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(game_type=name, game_name=name, params=params)


# a hidden coin (0 with probability 0.7); the player may pay 0.1 to see a signal that equals the coin with probability 0.8, then guesses it: +1 right,
# -1 wrong. Hand-computed: no look 0.4; look 0.5 (signal 0: P = 0.62, guess 0; signal 1: P = 0.38, guess 1)
_LOOK, _SKIP = 0, 1
_COIN_TYPE = sp.GameType(
    short_name="plybench_test_noisy_coin",
    long_name="plybench test noisy coin",
    dynamics=sp.GameType.Dynamics.SEQUENTIAL,
    chance_mode=sp.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=sp.GameType.Information.IMPERFECT_INFORMATION,
    utility=sp.GameType.Utility.GENERAL_SUM,
    reward_model=sp.GameType.RewardModel.TERMINAL,
    max_num_players=1,
    min_num_players=1,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
)
_COIN_INFO = sp.GameInfo(num_distinct_actions=2, max_chance_outcomes=2, num_players=1, min_utility=-1.1, max_utility=1.0, utility_sum=None, max_game_length=4)


class _CoinGame(sp.Game):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(_COIN_TYPE, _COIN_INFO, params or {})

    def new_initial_state(self) -> _CoinState:
        return _CoinState(self)

    def make_py_observer(self, iig_obs_type: Any = None, params: Any = None) -> _CoinObserver:
        return _CoinObserver()


class _CoinState(sp.State):
    def __init__(self, game: _CoinGame) -> None:
        super().__init__(game)
        self.coin: int | None = None
        self.looked: bool | None = None
        self.signal: int | None = None
        self.guess: int | None = None

    def current_player(self) -> int:
        if self.guess is not None:
            return sp.PlayerId.TERMINAL
        if self.coin is None or (self.looked and self.signal is None):
            return sp.PlayerId.CHANCE
        return 0

    def _legal_actions(self, player: int) -> list[int]:
        return [0, 1]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        if self.coin is None:
            return [(0, 0.7), (1, 0.3)]
        return [(self.coin, 0.8), (1 - self.coin, 0.2)]

    def _apply_action(self, action: int) -> None:
        if self.coin is None:
            self.coin = action
        elif self.looked is None:
            self.looked = action == _LOOK
        elif self.looked and self.signal is None:
            self.signal = action
        else:
            self.guess = action

    def _action_to_string(self, player: int, action: int) -> str:
        if player == sp.PlayerId.CHANCE:
            return f"side {action}"
        return ("look", "skip")[action] if self.looked is None else f"guess {action}"

    def is_terminal(self) -> bool:
        return self.guess is not None

    def returns(self) -> list[float]:
        if self.guess is None:
            return [0.0]
        return [(1.0 if self.guess == self.coin else -1.0) - (0.1 if self.looked else 0.0)]


class _CoinObserver:
    # what the player knows: its own choices and the signal, never the coin
    def __init__(self) -> None:
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, Any] = {}

    def set_from(self, state: _CoinState, player: int) -> None:
        pass

    def string_from(self, state: _CoinState, player: int) -> str:
        return f"looked={state.looked} signal={state.signal} guess={state.guess}"


sp.register_game(_COIN_TYPE, _CoinGame)


# two players race to 3 from 0, alternating; a move steps 1 for sure or rolls 0 or 2 with equal chance; the first to reach 3 wins (+1), a draw after 6 moves.
# The observation is the whole state, move count included, which the solver's key needs (pig's omits its move counter)
_STEP, _ROLL = 0, 1
_RACE_TYPE = sp.GameType(
    short_name="plybench_test_race",
    long_name="plybench test race",
    dynamics=sp.GameType.Dynamics.SEQUENTIAL,
    chance_mode=sp.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=sp.GameType.Information.PERFECT_INFORMATION,
    utility=sp.GameType.Utility.ZERO_SUM,
    reward_model=sp.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
)
_RACE_INFO = sp.GameInfo(num_distinct_actions=2, max_chance_outcomes=3, num_players=2, min_utility=-1.0, max_utility=1.0, utility_sum=0.0, max_game_length=12)


class _RaceGame(sp.Game):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(_RACE_TYPE, _RACE_INFO, params or {})

    def new_initial_state(self) -> _RaceState:
        return _RaceState(self)

    def make_py_observer(self, iig_obs_type: Any = None, params: Any = None) -> _RaceObserver:
        return _RaceObserver()


class _RaceState(sp.State):
    def __init__(self, game: _RaceGame) -> None:
        super().__init__(game)
        self.positions = [0, 0]
        self.mover = 0
        self.rolling = False
        self.moves = 0

    def current_player(self) -> int:
        if self.is_terminal():
            return sp.PlayerId.TERMINAL
        return sp.PlayerId.CHANCE if self.rolling else self.mover

    def _legal_actions(self, player: int) -> list[int]:
        return [_STEP, _ROLL]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        return [(0, 0.5), (2, 0.5)]

    def _apply_action(self, action: int) -> None:
        if self.rolling or action == _STEP:
            self.positions[self.mover] += action if self.rolling else 1
            self.rolling = False
            self.mover = 1 - self.mover
            self.moves += 1
        else:
            self.rolling = True

    def _action_to_string(self, player: int, action: int) -> str:
        return f"rolled {action}" if player == sp.PlayerId.CHANCE else ("step", "roll")[action]

    def is_terminal(self) -> bool:
        return max(self.positions) >= 3 or self.moves >= 6

    def returns(self) -> list[float]:
        if max(self.positions) < 3:
            return [0.0, 0.0]
        return [1.0, -1.0] if self.positions[0] >= 3 else [-1.0, 1.0]


class _RaceObserver:
    def __init__(self) -> None:
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, Any] = {}

    def set_from(self, state: _RaceState, player: int) -> None:
        pass

    def string_from(self, state: _RaceState, player: int) -> str:
        return f"positions={state.positions} mover={state.mover} rolling={state.rolling} moves={state.moves}"


sp.register_game(_RACE_TYPE, _RaceGame)


def _expectiminimax(state: sp.State, entries: dict[tuple[int, str], float]) -> float:
    # player 0's value by plain recursion over the tree, no memo; records every decision state's value from the mover's side
    if state.is_terminal():
        return state.returns()[0]
    if state.is_chance_node():
        return sum(p * _expectiminimax(state.child(a), entries) for a, p in state.chance_outcomes())
    player = state.current_player()
    sign = 1.0 if player == 0 else -1.0
    value = max(sign * _expectiminimax(state.child(a), entries) for a in state.legal_actions())
    entries[player, state.observation_string(player)] = value
    return sign * value


@pytest.mark.parametrize(
    ("name", "params"),
    [("catch", {"rows": 5, "columns": 3}), ("plybench_test_race", None)],
)
def test_perfect_information_with_chance_matches_brute_force(name: str, params: dict[str, Any] | None) -> None:
    game = _Game(name, params)
    cache = solve_game(game)
    entries: dict[tuple[int, str], float] = {}
    _expectiminimax(game.state.clone(), entries)
    assert entries
    for (player, observation), value in entries.items():
        entry = cache[player, observation]
        assert entry is not None
        assert entry.V() == pytest.approx(value)


def test_single_player_has_no_entries_for_a_second_seat() -> None:
    cache = solve_game(_Game("catch", {"rows": 5, "columns": 3}))
    assert set(cache.cache) == {0}


def _policy_value(state: sp.State, policy: dict[str, int]) -> float:
    if state.is_terminal():
        return state.returns()[0]
    if state.is_chance_node():
        return sum(p * _policy_value(state.child(a), policy) for a, p in state.chance_outcomes())
    return _policy_value(state.child(policy[state.information_state_string(0)]), policy)


def test_imperfect_information_single_player() -> None:
    game = _Game("plybench_test_noisy_coin")
    cache = solve_game(game)
    root = cache[0, "looked=None signal=None guess=None"]
    assert root is not None
    assert root.V() == pytest.approx(0.5)
    assert root.A() == [_LOOK]
    assert root.Q(_SKIP) == pytest.approx(0.4)
    # the signal's information states: conditional values, not values of one history
    signal_0 = cache[0, "looked=True signal=0 guess=None"]
    signal_1 = cache[0, "looked=True signal=1 guess=None"]
    assert signal_0 is not None and signal_1 is not None
    assert signal_0.A() == [0] and signal_0.V() == pytest.approx(2 * 0.56 / 0.62 - 1 - 0.1)
    assert signal_1.A() == [1] and signal_1.V() == pytest.approx(2 * 0.24 / 0.38 - 1 - 0.1)
    # the best deterministic policy, by enumeration, has the root value
    decisions = [key for (_, key), avq in _decision_entries(cache)]
    best = max(_policy_value(game.state.clone(), dict(zip(decisions, actions))) for actions in itertools.product([0, 1], repeat=len(decisions)))
    assert best == pytest.approx(root.V())


def _decision_entries(cache: AVQCache) -> list[tuple[tuple[int, str], Any]]:
    return [((player, key), avq) for player, entries in cache.cache.items() for key, avq in entries.items() if avq.A()]


def test_two_player_imperfect_information_is_refused() -> None:
    with pytest.raises(ValueError, match="imperfect information"):
        solve_game(_Game("kuhn_poker"))


def test_histories_hide_chance_and_other_players_private_actions() -> None:
    state = sp.load_game("plybench_test_noisy_coin").new_initial_state()
    for action in (1, _LOOK, 0):
        state.apply_action(action)
    observation = TurnBasedState.get_observation(state, 0)
    assert observation.state == "looked=True signal=0 guess=None"
    assert [action.number for action in observation.i_actions] == [_LOOK]
    assert observation.o_actions == []

    kuhn = sp.load_game("kuhn_poker").new_initial_state()
    for action in (0, 1, 1):  # deal, deal, player 0 bets
        kuhn.apply_action(action)
    assert [action.string for action in TurnBasedState.get_observation(kuhn, 1).o_actions] == []
    assert [action.string for action in TurnBasedState.get_observation(kuhn, 0).i_actions] == ["Bet"]
