"""The solver on games with chance, with one player, and with hidden information, against brute force and hand-computed values."""

from __future__ import annotations

import itertools
from typing import Any

import pyspiel as sp
import pytest
from toy_games import LOOK, SKIP, ToyGame

from plybench.core.game import TurnBasedState
from plybench.core.minimax import AVQCache, optimal_outcomes, solve_game
from plybench.games.monty_hall.monty_hall import MontyHallGame, MontyHallGameParams


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
    game = ToyGame(name, params)
    cache = solve_game(game)
    entries: dict[tuple[int, str], float] = {}
    _expectiminimax(game.state.clone(), entries)
    assert entries
    for (player, observation), value in entries.items():
        entry = cache[player, observation]
        assert entry is not None
        assert entry.V() == pytest.approx(value)


def test_single_player_has_no_entries_for_a_second_seat() -> None:
    cache = solve_game(ToyGame("catch", {"rows": 5, "columns": 3}))
    assert set(cache.cache) == {0}


def _policy_value(state: sp.State, policy: dict[str, int]) -> float:
    if state.is_terminal():
        return state.returns()[0]
    if state.is_chance_node():
        return sum(p * _policy_value(state.child(a), policy) for a, p in state.chance_outcomes())
    return _policy_value(state.child(policy[state.information_state_string(0)]), policy)


def test_imperfect_information_single_player() -> None:
    game = ToyGame("plybench_test_noisy_coin")
    cache = solve_game(game)
    root = cache[0, "looked=None signal=None guess=None"]
    assert root is not None
    assert root.V() == pytest.approx(0.5)
    assert root.A() == [LOOK]
    assert root.Q(SKIP) == pytest.approx(0.4)
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
        solve_game(ToyGame("kuhn_poker"))


def test_histories_hide_chance_and_other_players_private_actions() -> None:
    state = sp.load_game("plybench_test_noisy_coin").new_initial_state()
    for action in (1, LOOK, 0):
        state.apply_action(action)
    observation = TurnBasedState.get_observation(state, 0)
    assert observation.state == "looked=True signal=0 guess=None"
    assert [action.number for action in observation.i_actions] == [LOOK]
    assert observation.o_actions == []

    kuhn = sp.load_game("kuhn_poker").new_initial_state()
    for action in (0, 1, 1):  # deal, deal, player 0 bets
        kuhn.apply_action(action)
    assert [action.string for action in TurnBasedState.get_observation(kuhn, 1).o_actions] == []
    assert [action.string for action in TurnBasedState.get_observation(kuhn, 0).i_actions] == ["Bet"]


def test_a_game_in_progress_is_solved_from_its_start() -> None:
    # Monty Hall with the car placed and a door picked: solving from the current state would know the car; the values must be those of the whole game
    fresh = solve_game(MontyHallGame(MontyHallGameParams(num_doors=3, final="pick", host="hell")))
    game = MontyHallGame(MontyHallGameParams(num_doors=3, final="pick", host="hell"))
    game.step_chance(1)
    game.step(game.get_legal_moves(0)[0])
    assert solve_game(game).to_dict() == fresh.to_dict()
    no_door = fresh[0, "picked=1 opened=none behind=- final=- car=-"]
    assert no_door is not None and no_door.V() == pytest.approx(0.0)


def test_the_optimal_outcomes_need_a_single_player_game() -> None:
    with pytest.raises(ValueError, match="single-player"):
        optimal_outcomes(ToyGame("tic_tac_toe"))
