"""Single-player games through the harness: the player keeps the first seat whatever the colour rule, the empty seat is the `none` player, a matchup
that puts `none` in the wrong seat is refused, and the prompt names no opponent."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from plyllm import LLMConfig
from toy_games import ToyEngine

from plybench.app import PlyBench
from plybench.common.enums import GameResults, ObservationType
from plybench.configs.game_params import NoGameParams
from plybench.configs.matchup import Matchup
from plybench.core.interface import InterfaceObservation
from plybench.games.spec import GameSpec
from plybench.harness.matchup import COLOUR_RULES, run_matchup_concurrent

op = PlyBench(LLMConfig())
registry = op.registry
registry.register_game(GameSpec("toy_coin", NoGameParams, lambda game_config: ToyEngine("plybench_test_noisy_coin"), solvable=True))

COIN = registry.game_config("toy_coin:")
RANDOM = registry.player_config("random:distribution=uniform")
NONE = registry.player_config("none:")


@pytest.mark.parametrize("colour_rule", sorted(COLOUR_RULES))
def test_a_single_player_matchup_keeps_its_player_in_the_first_seat(tmp_path, monkeypatch, colour_rule):
    monkeypatch.chdir(tmp_path)
    # six rounds: both colour rules would give the o-player the first seat in three of them
    tracker = asyncio.run(run_matchup_concurrent(op, Matchup(COIN, RANDOM, NONE, 6), experiment="solo", colour_rule=COLOUR_RULES[colour_rule], seed=1))

    assert tracker.is_complete()
    records = [json.loads(path.read_text()) for path in sorted(Path("results").rglob("game_*.json"))]
    assert len(records) == 6
    for record in records:
        assert record["i_player"] == RANDOM.to_string() and record["o_player"] == NONE.to_string()
        assert record["steps"] and all(step["player_name"] == RANDOM.to_string() for step in record["steps"])
        assert record["ending"]["result"] in (GameResults.WIN.value, GameResults.LOSS.value)


@pytest.mark.parametrize(
    ("game", "i", "o"),
    [
        ("toy_coin:", "random:distribution=uniform", "random:distribution=uniform"),
        ("toy_coin:", "none:", "none:"),
        ("toy_coin:", "none:", "random:distribution=uniform"),
        ("tic_tac_toe:", "random:distribution=uniform", "none:"),
        ("tic_tac_toe:", "none:", "random:distribution=uniform"),
    ],
)
def test_none_in_the_wrong_seat_is_refused(tmp_path, monkeypatch, game, i, o):
    monkeypatch.chdir(tmp_path)
    matchup = Matchup(registry.game_config(game), registry.player_config(i), registry.player_config(o), 1)
    with pytest.raises(ValueError, match="none"):
        asyncio.run(run_matchup_concurrent(op, matchup, experiment="wrong"))
    assert not Path("results").exists()


def _first_decision(engine: ToyEngine) -> InterfaceObservation:
    # the coin's first decision, after chance hid the coin
    engine.reset()
    engine.game.step_chance(0)
    return engine.observation_class.from_openspiel(engine.game.get_observation(0), engine.interface_transformer)


def test_the_none_player_refuses_to_move():
    engine = ToyEngine("plybench_test_noisy_coin")
    observation = _first_decision(engine)
    player = registry.build_player(engine.game, NONE, "o")
    with pytest.raises(RuntimeError, match="never asked to move"):
        asyncio.run(player(engine.game, observation, []))


@pytest.mark.parametrize("observation_type", [ObservationType.ACTIONS, ObservationType.STATE])
def test_a_single_player_prompt_names_no_opponent(observation_type):
    engine = ToyEngine("plybench_test_noisy_coin")
    observation = _first_decision(engine)
    adapter = engine.prompt_adapter.for_observation_type(observation_type)
    prompt = adapter.observation_prompt(adapter.format_observation(observation))

    assert prompt.startswith("You ")
    assert "opponent" not in prompt and "You play as" not in prompt
    assert "beat the user" not in adapter.system_prompt
