"""Bandit: the solver gives Q*(a) = 2 p_a - 1 per button (the mean of a Bernoulli button with rewards -1 and +1), the shift moves the probabilities cyclically over fixed buttons (each probability on each
button once over K shifts), the hidden rules keep the probabilities out of a prompt that is then the same for every instance and shift, the optimal
and random players win as often as the probabilities say, and a recorded state reads back."""

from __future__ import annotations

import asyncio
import json
from fractions import Fraction
from pathlib import Path

import pyspiel as sp
import pytest
from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.common.enums import GameResults
from plybench.configs.matchup import Matchup
from plybench.core.minimax import optimal_outcomes, solve_game
from plybench.games.bandit.bandit import BanditEngine, BanditGameParams, BanditObservation
from plybench.harness.matchup import run_matchup_concurrent
from plybench.player.output_strategies import StructuredOutputStrategy

op = PlyBench(LLMConfig())
registry = op.registry

START = "pressed=- reward=-"


def _engine(params: str) -> BanditEngine:
    engine = registry.build_engine(registry.game_config(f"bandit:{params}"))
    assert isinstance(engine, BanditEngine)
    return engine


def _prompt(engine: BanditEngine) -> str:
    engine.reset()
    game = engine.game
    observation = engine.observation_class.from_openspiel(game.get_observation(0), engine.interface_transformer)
    legal = [engine.action_class.from_openspiel(move, engine.interface_transformer) for move in game.get_legal_moves(0)]
    return str(engine.prompt_adapter.build_messages(observation, legal, StructuredOutputStrategy())[1][0])


@pytest.mark.parametrize(("p", "shift"), [("3/4;1/4;1/4;1/4;1/4", 0), ("0.6;0.4;0.4;0.4;0.4", 3), ("1/10;1/2;9/10", 1)])
def test_each_button_is_worth_twice_its_probability_minus_one(p, shift):
    engine = _engine(f"p={p},shift={shift}")
    chances = [Fraction(chance) for chance in p.split(";")]
    root = solve_game(engine.game)[0, START]
    assert root is not None
    k = len(chances)
    for button in range(k):
        assert root.Q(button) == pytest.approx(float(2 * chances[(button - shift) % k] - 1))
    best = max(chances)
    assert root.A() == [button for button in range(k) if chances[(button - shift) % k] == best]


def test_shifts_put_the_best_probability_on_each_button_once():
    best = []
    for shift in range(5):
        root = solve_game(_engine(f"p=0.6;0.4;0.4;0.4;0.4,shift={shift}").game)[0, START]
        assert root is not None
        best.extend(root.A())
    assert best == [0, 1, 2, 3, 4]
    # a shift of K is the shift 0 again, the game itself; only the string and the path tell them apart
    five, zero = _engine("shift=5").game_config.params, _engine("shift=0").game_config.params
    assert isinstance(five, BanditGameParams) and isinstance(zero, BanditGameParams) and five.chances == zero.chances


def test_hidden_rules_keep_the_probabilities_out_of_the_prompt():
    stated, hidden = _prompt(_engine("p=0.6;0.4;0.4;0.4;0.4,shift=2")), _prompt(_engine("p=0.6;0.4;0.4;0.4;0.4,shift=2,rules=hidden"))
    assert "The red button wins with probability 2/5" in stated and "the green button wins with probability 3/5" in stated
    assert "probability" in hidden and "2/5" not in hidden and "3/5" not in hidden and "You are not told these probabilities." in hidden
    assert "Currently, you have not pressed a button yet." in hidden and "<button:red>, <button:blue>, <button:green>, <button:yellow>, <button:purple>" in hidden
    # with the probabilities hidden, every instance and shift of K buttons gives the same prompt: what is learned comes from the games
    for params in ("p=3/4;1/4;1/4;1/4;1/4", "p=0.6;0.4;0.4;0.4;0.4,shift=4"):
        assert _prompt(_engine(f"{params},rules=hidden")) == hidden


def test_strings_paths_and_the_cache():
    engine = _engine("p=0.6;0.4;0.4,shift=1,rules=hidden")
    params = engine.game_config.params
    # a decimal and a fraction are one probability
    assert params.to_string() == "p=3/5;2/5;2/5,shift=1,rules=hidden" and params.path_suffix == "3_buttons_p_3-5_2-5_2-5_shift_1_hidden"
    assert registry.game_config(f"bandit:{params.to_string()}") == engine.game_config
    # the rules change only the prompt: a stated and a hidden variant are one OpenSpiel game, and so share the minimax cache
    assert str(engine.game.game) == str(_engine("p=3/5;2/5;2/5,shift=1").game.game) == "bandit(p=3/5;2/5;2/5,shift=1)"


def test_invalid_params_are_refused():
    with pytest.raises(ValueError, match="2 to 8 buttons"):
        registry.game_config("bandit:p=1/2")
    with pytest.raises(ValueError, match="2 to 8 buttons"):
        registry.game_config("bandit:p=" + ";".join(["1/2"] * 9))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        registry.game_config("bandit:p=1/2;3/2")
    with pytest.raises(ValueError, match="non-negative"):
        registry.game_config("bandit:shift=-1")
    with pytest.raises(ValueError, match="Invalid rules"):
        registry.game_config("bandit:rules=hidden_car")


@pytest.mark.parametrize(("player", "expected"), [("optimal:stochastic=False", 3 / 4), ("random:distribution=uniform", (3 / 4 + 4 * 1 / 4) / 5)])
def test_win_rates_match_the_probabilities(tmp_path, monkeypatch, player, expected):
    # optimal always presses the best button (3/4); random presses each with 1/5 (7/20); 600 games, a standard error of about 0.02
    monkeypatch.chdir(tmp_path)
    n = 600
    matchup = Matchup(registry.game_config("bandit:"), registry.player_config(player), registry.player_config("none:"), n)
    asyncio.run(run_matchup_concurrent(op, matchup, experiment="bandit", seed=0))
    records = [json.loads(path.read_text()) for path in Path("results").rglob("game_*.json")]
    assert len(records) == n
    wins = sum(record["ending"]["result"] == GameResults.WIN.value for record in records)
    assert abs(wins / n - expected) < 0.06


def test_the_optimal_policy_wins_with_the_best_probability(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outcomes = optimal_outcomes(_engine("p=0.6;0.4;0.4;0.4;0.4,shift=3").game)
    assert outcomes[GameResults.WIN] == pytest.approx(0.6) and outcomes[GameResults.LOSS] == pytest.approx(0.4) and outcomes[GameResults.DRAW] == 0.0


def test_the_result_is_seen_only_at_the_end_and_a_recorded_state_reads_back():
    engine = _engine("shift=1")
    engine.reset()
    game = engine.game
    game.step(game.get_legal_moves(0)[2])  # the player presses the green button
    # the outcomes index the game's rewards (-1, +1): the green button loses with 3/4
    assert game.is_chance_node() and game.state.chance_outcomes() == [(0, pytest.approx(3 / 4)), (1, pytest.approx(1 / 4))]
    game.step_chance(1)  # a win
    assert game.get_returns() == [1.0]
    end = BanditObservation.from_openspiel(game.get_observation(0), engine.interface_transformer)
    assert end.pressed == "green" and end.reward == 1
    serialized = game.serialize_state()
    engine.reset()
    engine.game.deserialize_state(serialized)
    assert engine.game.get_observation(0).state == "pressed=green reward=+1"
    assert sp.load_game("bandit").get_type().information == sp.GameType.Information.IMPERFECT_INFORMATION


def test_a_certain_button_has_one_outcome():
    engine = _engine("p=1;0")
    engine.reset()
    game = engine.game
    game.step(game.get_legal_moves(0)[0])
    assert game.state.chance_outcomes() == [(1, pytest.approx(1.0))]
    root = solve_game(engine.game)[0, START]
    assert root is not None and root.Q(0) == pytest.approx(1.0) and root.Q(1) == pytest.approx(-1.0)


def test_a_switch_changes_the_probabilities_from_its_epoch_on():
    config = registry.game_config("bandit:p=0;1,switch=0.8;0.2,switch_epoch=2,shift=1,rules=hidden,stationarity=omitted")
    assert config.to_string() == "bandit:p=0;1,shift=1,rules=hidden,switch=4/5;1/5,switch_epoch=2,stationarity=omitted"
    assert config.path == "bandit_2_buttons_p_0_1_shift_1_hidden_switch_4-5_1-5_epoch_2_stationarity_omitted"
    assert registry.game_config(config.to_string()) == config
    # before the switch the run's own game; from it on a game without a switch, with the same shift, rules and prompt
    assert config.at_epoch(0) is config and config.at_epoch(1) is config
    after = config.at_epoch(2)
    assert after == config.at_epoch(5) and after.to_string() == "bandit:p=4/5;1/5,shift=1,rules=hidden,stationarity=omitted"
    root = solve_game(registry.build_engine(after).game)[0, START]
    assert root is not None and root.A() == [1] and root.Q(1) == pytest.approx(0.6)
    # a game without a switch is the same in every epoch
    plain = registry.game_config("bandit:p=0.6;0.4")
    assert plain.at_epoch(3) is plain


def test_the_stationarity_sentence_can_be_omitted():
    stated, omitted = _prompt(_engine("p=0.6;0.4,rules=hidden")), _prompt(_engine("p=0.6;0.4,rules=hidden,stationarity=omitted"))
    assert "Each button wins with its own fixed probability, which does not change between games. You are not told these probabilities." in stated
    assert "does not change" not in omitted and "Each button wins with its own probability. You are not told these probabilities." in omitted
    assert omitted.replace("its own probability.", "its own fixed probability, which does not change between games.") == stated


def test_invalid_switches_are_refused():
    with pytest.raises(ValueError, match="needs both"):
        registry.game_config("bandit:p=0;1,switch=1;0")
    with pytest.raises(ValueError, match="needs both"):
        registry.game_config("bandit:p=0;1,switch_epoch=2")
    with pytest.raises(ValueError, match="as many probabilities"):
        registry.game_config("bandit:p=0;1,switch=1;0;0,switch_epoch=2")
    with pytest.raises(ValueError, match="at least 1"):
        registry.game_config("bandit:p=0;1,switch=1;0,switch_epoch=0")
    with pytest.raises(ValueError, match="Invalid stationarity"):
        registry.game_config("bandit:stationarity=dropped")
