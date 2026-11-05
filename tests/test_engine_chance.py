"""The engine on games with chance and with one player: chance is sampled from its own seeded stream and never recorded as a move, and a single-player
game asks only the first seat and scores its return."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from toy_games import LOOK, ROLL, SKIP, ToyEngine

from plybench.common.enums import GameResults
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.trackers.game_tracker import GameTracker
from plybench.utils.seed import derive_seed


@dataclass(frozen=True)
class _Params(PlayerParams):
    name: str

    @classmethod
    def from_string(cls, params_string: str) -> _Params:
        return cls(params_string)

    def to_string(self) -> str:
        return self.name

    @property
    def path_suffix(self) -> str:
        return self.name


type Policy = Callable[[Player, list[InterfaceAction]], InterfaceAction]


class _Scripted(Player):
    def __init__(self, name: str, identifier: PlayerIdentifier, policy: Policy) -> None:
        super().__init__(PlayerConfig("scripted", _Params(name)), identifier)
        self.policy = policy
        self.calls = 0

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        pass

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        self.calls += 1
        return PlayerOutput(action=self.policy(self, legal_moves))

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""


def _uniform(player: Player, moves: list[InterfaceAction]) -> InterfaceAction:
    return moves[int(player.rng.choice(len(moves)))]


def _number(number: int) -> Policy:
    return lambda player, moves: next(move for move in moves if move.number == number)


def _never(player: Player, moves: list[InterfaceAction]) -> InterfaceAction:
    raise AssertionError("the empty seat of a single-player game was asked to move")


def _play(name: str, players: tuple[Player, Player], seed: int | None) -> GameTracker:
    return asyncio.run(ToyEngine(name).play(players, seed=seed))


def _race(seed: int | None) -> GameTracker:
    return _play("plybench_test_race", (_Scripted("a", "i", _uniform), _Scripted("b", "o", _uniform)), seed)


def test_chance_is_sampled_not_recorded_and_fixed_by_the_seed():
    games = [_race(seed) for seed in range(20)]
    # the recorded moves are the players' only; the rolls are in the serialized states
    assert all(step.move in ("<step>", "<roll>") for game in games for step in game.steps)
    assert all(game.ending is not None and game.ending.result in (GameResults.WIN, GameResults.LOSS, GameResults.DRAW) for game in games)
    assert any(step.move == "<roll>" for game in games for step in game.steps)

    again = [_race(seed) for seed in range(20)]
    assert [[step.serialized_state for step in game.steps] for game in games] == [[step.serialized_state for step in game.steps] for game in again]


def test_chance_draws_do_not_shift_the_players_draws():
    # player i always rolls, so its draws are untouched; player o draws uniformly: its moves must not depend on what chance did before them
    def moves_of_o(seed: int) -> list[str]:
        game = _play("plybench_test_race", (_Scripted("a", "i", _number(ROLL)), _Scripted("b", "o", _uniform)), seed)
        return [step.move for step in game.steps if step.player_name.endswith("b")]

    reference = [_Scripted("b", "o", _uniform) for _ in range(20)]
    for seed, player in zip(range(20), reference):
        player.seed(derive_seed(seed, "o"))
        moves = moves_of_o(seed)
        expected = ["<step>" if int(player.rng.choice(2)) == 0 else "<roll>" for _ in moves]
        assert moves == expected


def test_chance_outcomes_follow_their_probabilities():
    # never looking and guessing 0 wins exactly when the hidden coin is 0, which chance draws with probability 0.7
    def skip_then_guess_0(player: Player, moves: list[InterfaceAction]) -> InterfaceAction:
        return _number(SKIP)(player, moves) if str(moves[0]) == "look" else _number(0)(player, moves)

    def won(seed: int) -> bool:
        game = _play("plybench_test_noisy_coin", (_Scripted("guess", "i", skip_then_guess_0), _Scripted("none", "o", _never)), seed)
        assert game.ending is not None
        return game.ending.result is GameResults.WIN

    n = 1000
    wins = sum(won(seed) for seed in range(n))
    # 0.7 within about 3.5 standard errors (0.0145 at n = 1000)
    assert abs(wins / n - 0.7) < 0.05


def test_a_single_player_game_asks_only_the_first_seat_and_scores_its_return():
    looker = _Scripted("looker", "i", lambda player, moves: _number(LOOK)(player, moves) if str(moves[0]) == "look" else _uniform(player, moves))
    absent = _Scripted("none", "o", _never)
    game = _play("plybench_test_noisy_coin", (looker, absent), seed=3)

    assert absent.calls == 0
    assert [step.move for step in game.steps][0] == "<look>"
    assert all(step.player_name.endswith("looker") for step in game.steps)
    assert game.ending is not None
    # the return is +0.9 or -1.1: a win or a loss, never a draw; the ending is the player's own information state
    assert game.ending.result in (GameResults.WIN, GameResults.LOSS)
    assert "looked=True signal=" in game.ending.observation


def test_an_unseeded_game_with_chance_plays():
    game = _play("plybench_test_noisy_coin", (_Scripted("skipper", "i", _number(SKIP)), _Scripted("none", "o", _never)), seed=None)
    assert game.ending is not None
    assert game.steps[0].move == "<skip>"
