"""Phase 3 verification: every registered game builds via the registry and plays to a terminal
state with random players, with no illegal moves and legal-move round-tripping intact."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import pytest

from plybench.common.enums import GameResults
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.games.builtins import register_builtin_games
from plybench.player.player import Player, PlayerOutput
from plybench.registry import Registry

registry = Registry()
register_builtin_games(registry)

_VALID_RESULTS = {GameResults.WIN, GameResults.LOSS, GameResults.DRAW}


@dataclass(frozen=True)
class _RandParams(PlayerParams):
    seed: str

    @classmethod
    def from_string(cls, params_string: str) -> "_RandParams":
        return cls(params_string)

    def to_string(self) -> str:
        return self.seed

    @property
    def path_suffix(self) -> str:
        return self.seed


class _RandomPlayer(Player):
    def __init__(self, player_config: PlayerConfig, identifier, seed: int) -> None:
        super().__init__(player_config, identifier)
        self._rng = random.Random(seed)

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        pass

    async def __call__(self, game, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        # exercise the action round-trip on every move: to_openspiel must preserve the number
        choice = self._rng.choice(legal_moves)
        assert choice.to_openspiel().number == choice.number
        return PlayerOutput(action=choice)

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""


def _players() -> tuple[_RandomPlayer, _RandomPlayer]:
    i = PlayerConfig("rand", _RandParams("1"))
    o = PlayerConfig("rand", _RandParams("2"))
    return _RandomPlayer(i, "i", seed=1), _RandomPlayer(o, "o", seed=2)


@pytest.mark.parametrize("game_key", registry.game_keys())
def test_game_plays_to_terminal(game_key: str):
    engine = registry.build_engine(registry.game_config(f"{game_key}:"))
    tracker = asyncio.run(engine.play(_players()))

    assert tracker.ending is not None, f"{game_key} did not reach a terminal state"
    assert tracker.ending.result in _VALID_RESULTS
    assert len(tracker.steps) >= 1
    # random players pick only legal moves, so there should be no FAIL steps
    assert all(not step.move.startswith("FAIL") for step in tracker.steps), f"{game_key} produced an illegal move"
    # both players are bots -> no token accounting
    assert all(step.input_tokens is None for step in tracker.steps)


def test_all_twelve_games_registered():
    assert len(registry.game_keys()) == 12
    for key in registry.game_keys():
        assert registry.resolve_game(key).engine_factory is not None


def test_game_config_round_trips_old_strings():
    # migration guarantee: the registry parses the original serialization unchanged
    for raw in ["tic_tac_toe:", "magic_square:sample=False,magic_constant_add=0", "nim:", "nim:sample=True,num_piles=4,max_pile_size=8,pile_sum=16,nim_start=winning"]:
        assert registry.game_config(raw).to_string() == raw
    assert registry.game_config("magic_square:sample=False,magic_constant_add=0").path == "magic_square_normal_add_0"
    assert registry.game_config("tic_tac_toe:").path == "tic_tac_toe"
    assert registry.solvable("connect_four") is False
    assert registry.solvable("tic_tac_toe") is True
