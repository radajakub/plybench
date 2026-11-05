"""Monty Hall: the solver gives the known values of the standard, four-door, ignorant-host and hell-host variants, the optimal and random players win as
often as those values say, the player never sees where the car is before the end, hidden rules keep the host's rule out of the prompt, and a recorded
state reads back."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pyspiel as sp
import pytest
from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.common.enums import GameResults, ObservationType
from plybench.configs.matchup import Matchup
from plybench.core.minimax import solve_game
from plybench.games.monty_hall.monty_hall import MontyHallEngine
from plybench.harness.matchup import run_matchup_concurrent
from plybench.player.output_strategies import StructuredOutputStrategy

op = PlyBench(LLMConfig())
registry = op.registry

START = "picked=- opened=- behind=- final=- car=-"


def _engine(params: str) -> MontyHallEngine:
    engine = registry.build_engine(registry.game_config(f"monty_hall:{params}"))
    assert isinstance(engine, MontyHallEngine)
    return engine


def test_three_doors_switching_wins_two_thirds():
    cache = solve_game(_engine("").game)
    root = cache[0, START]
    assert root is not None and root.V() == pytest.approx(1 / 3)  # win 2/3, lose 1/3
    for picked, opened in [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]:
        entry = cache[0, f"picked={picked} opened={opened} behind=goat final=- car=-"]
        assert entry is not None
        stay, switch = 3, 4
        assert entry.Q(stay) == pytest.approx(-1 / 3) and entry.Q(switch) == pytest.approx(1 / 3)
        assert entry.A() == [switch]


def test_four_doors_another_closed_door_wins_three_eighths():
    cache = solve_game(_engine("num_doors=4,final=pick,host=standard").game)
    entry = cache[0, "picked=1 opened=2 behind=goat final=- car=-"]
    assert entry is not None
    # door numbers 1, 3, 4 are actions 0, 2, 3; the first pick wins 1/4, each other closed door 3/8
    assert entry.Q(0) == pytest.approx(2 * 1 / 4 - 1)
    assert entry.Q(2) == pytest.approx(2 * 3 / 8 - 1) and entry.Q(3) == pytest.approx(2 * 3 / 8 - 1)
    assert entry.A() == [2, 3]
    root = cache[0, START]
    assert root is not None and root.V() == pytest.approx(2 * 3 / 8 - 1)


def test_an_ignorant_host_makes_switching_worthless():
    cache = solve_game(_engine("num_doors=3,final=switch,host=ignorant").game)
    goat = cache[0, "picked=1 opened=2 behind=goat final=- car=-"]
    car = cache[0, "picked=1 opened=2 behind=car final=- car=-"]
    assert goat is not None and car is not None
    # given a goat behind the opened door, the car is behind either closed door with 1/2; given the car, both lose
    assert goat.Q(3) == pytest.approx(0.0) and goat.Q(4) == pytest.approx(0.0)
    assert car.V() == pytest.approx(-1.0)
    root = cache[0, START]
    assert root is not None and root.V() == pytest.approx(2 / 3 * 0.0 + 1 / 3 * -1.0)


def test_a_hell_host_makes_an_open_door_mean_stay():
    cache = solve_game(_engine("num_doors=3,final=pick,host=hell").game)
    offer = cache[0, "picked=1 opened=2 behind=goat final=- car=-"]
    no_door = cache[0, "picked=1 opened=none behind=- final=- car=-"]
    assert offer is not None and no_door is not None
    # the host opens a door only when the first pick hides the car: then staying (door 1, action 0) wins; else the car is behind door 2 or 3 with 1/2
    assert offer.Q(0) == pytest.approx(1.0) and offer.Q(2) == pytest.approx(-1.0) and offer.A() == [0]
    assert no_door.Q(0) == pytest.approx(-1.0) and no_door.Q(1) == pytest.approx(0.0) and no_door.Q(2) == pytest.approx(0.0) and no_door.A() == [1, 2]
    root = cache[0, START]
    assert root is not None and root.V() == pytest.approx(1 / 3 * 1.0 + 2 / 3 * 0.0)  # win 2/3


def test_hidden_rules_keep_the_host_rule_out_of_the_prompt():
    stated, hidden = _engine("num_doors=3,final=pick,host=hell"), _engine("num_doors=3,final=pick,host=hell,rules=hidden")
    stated_head, hidden_head = stated.prompt_adapter.head_prompt, hidden.prompt_adapter.head_prompt
    assert stated_head is not None and hidden_head is not None
    assert "only when your first pick hides the car" in stated_head
    assert "only when" not in hidden_head and "You are not told how the host decides" in hidden_head
    # the strings and paths of stated rules are those from before the param existed
    assert stated.game_config.params.to_string() == "num_doors=3,final=pick,host=hell" and stated.game_config.params.path_suffix == "3_doors_pick_hell"
    assert hidden.game_config.params.to_string() == "num_doors=3,final=pick,host=hell,rules=hidden"
    assert hidden.game_config.params.path_suffix == "3_doors_pick_hell_hidden"

    hidden.reset()
    game = hidden.game
    game.step_chance(1)  # the car behind door 2
    game.step(game.get_legal_moves(0)[0])  # the player picks door 1
    game.step_chance(3)  # the host opens no door
    observation = hidden.observation_class.from_openspiel(game.get_observation(0), hidden.interface_transformer)
    legal = [hidden.action_class.from_openspiel(move, hidden.interface_transformer) for move in game.get_legal_moves(0)]
    text = str(hidden.prompt_adapter.build_messages(observation, legal, StructuredOutputStrategy())[1][0])
    assert "you picked door 1, the host opened no door" in text and "<door:1>, <door:2>, <door:3>" in text


def test_invalid_params_are_refused():
    with pytest.raises(ValueError, match="needs 3 doors"):
        registry.game_config("monty_hall:num_doors=4,final=switch,host=standard")
    with pytest.raises(ValueError, match="Invalid host"):
        registry.game_config("monty_hall:host=lazy")
    with pytest.raises(ValueError, match="needs the final choice `pick`"):
        registry.game_config("monty_hall:num_doors=3,final=switch,host=hell")
    with pytest.raises(ValueError, match="Invalid rules"):
        registry.game_config("monty_hall:rules=secret")


@pytest.mark.parametrize(("player", "expected"), [("optimal:stochastic=False", 2 / 3), ("random:distribution=uniform", 1 / 2)])
def test_win_rates_match_the_solved_values(tmp_path, monkeypatch, player, expected):
    # optimal always switches (2/3); random picks stay or switch uniformly (1/2); 600 games, a standard error of about 0.02
    monkeypatch.chdir(tmp_path)
    n = 600
    matchup = Matchup(registry.game_config("monty_hall:"), registry.player_config(player), registry.player_config("none:"), n)
    asyncio.run(run_matchup_concurrent(op, matchup, experiment="monty", seed=0))
    records = [json.loads(path.read_text()) for path in Path("results").rglob("game_*.json")]
    assert len(records) == n
    wins = sum(record["ending"]["result"] == GameResults.WIN.value for record in records)
    assert abs(wins / n - expected) < 0.06


def test_the_player_never_sees_the_car_before_the_end():
    engine = _engine("")
    adapter = engine.prompt_adapter
    engine.reset()
    game = engine.game
    for outcome in (2, 0, 1):  # the car behind door 3, the player picks door 1, the host opens door 2
        if game.is_chance_node():
            game.step_chance(outcome)
        else:
            game.step(game.get_legal_moves(0)[outcome])
    observation = engine.observation_class.from_openspiel(game.get_observation(0), engine.interface_transformer)
    legal = [engine.action_class.from_openspiel(move, engine.interface_transformer) for move in game.get_legal_moves(0)]
    prompts = [adapter.for_observation_type(t).build_messages(observation, legal, StructuredOutputStrategy())[1][0] for t in ObservationType]

    assert str(prompts[0]) == str(prompts[1])
    text = str(prompts[0])
    assert "you picked door 1, the host opened door 2, which hides a goat" in text
    assert "<stay>" in text and "<switch>" in text
    # the rules say a car is behind one door; where, only the game's end tells ("the car is behind door 3")
    assert "car is behind door" not in text and "door 3," not in text and "opponent" not in text


def test_a_recorded_state_reads_back():
    engine = _engine("num_doors=4,final=pick,host=ignorant")
    engine.reset()
    for action in (3, 0, 1):
        engine.game.state.apply_action(action)
    serialized = engine.game.serialize_state()
    expected = engine.game.get_observation(0).state
    engine.reset()
    engine.game.deserialize_state(serialized)
    assert engine.game.get_observation(0).state == expected == "picked=1 opened=2 behind=goat final=- car=-"
    assert sp.load_game("monty_hall").get_type().information == sp.GameType.Information.IMPERFECT_INFORMATION


def test_the_display_draws_the_doors_and_hides_the_car_until_the_end():
    engine = _engine("")
    engine.reset()
    game = engine.game
    game.step_chance(2)  # the car behind door 3
    game.step(game.get_legal_moves(0)[0])  # the player picks door 1
    game.step_chance(1)  # the host opens door 2

    def display() -> str:
        # the drawing only, without the state line under it
        return str(engine.observation_class.from_openspiel(game.get_observation(0), engine.interface_transformer)).split("\npicked=")[0]

    middle = display()
    assert "door 1" in middle and "door 3" in middle and "picked" in middle and "opened" in middle
    assert middle.count("?") == 2 and middle.count("goat") == 1 and "car" not in middle

    game.step(game.get_legal_moves(0)[1])  # switch to door 3
    end = display()
    assert "?" not in end and end.count("goat") == 2 and "|  car  |" in end and "final" in end
