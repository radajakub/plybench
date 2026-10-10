from __future__ import annotations

from collections.abc import Callable

from plyllm import LLM

from plybench.configs.player_config import PlayerConfig
from plybench.core.game import TurnBasedGame
from plybench.player.llm_player import LLMParams, LLMPlayer, LLMPlayerTracker
from plybench.player.output_strategies import build_output_strategy
from plybench.player.player import Player, PlayerIdentifier
from plybench.player.simple.human_player import HumanParams, HumanPlayer
from plybench.player.simple.mcts_player import MctsParams, MCTSPlayer
from plybench.player.simple.optimal_player import OptimalParams, OptimalPlayer
from plybench.player.simple.random_player import RandomParams, RandomPlayer
from plybench.player.spec import PlayerSpec
from plybench.registry import Registry


def _llm_builder(llm: LLM) -> Callable[[TurnBasedGame, PlayerConfig, PlayerIdentifier], Player]:
    def build(game: TurnBasedGame, player_config: PlayerConfig, identifier: PlayerIdentifier) -> Player:
        params = player_config.params
        assert isinstance(params, LLMParams)
        return LLMPlayer(player_config, build_output_strategy(params.output_strategy), llm, identifier)

    return build


def register_builtin_players(registry: Registry, llm: LLM) -> None:
    registry.register_player(PlayerSpec("human", HumanParams, lambda game, cfg, pid: HumanPlayer(cfg, pid)))
    registry.register_player(PlayerSpec("random", RandomParams, lambda game, cfg, pid: RandomPlayer(cfg, pid)))
    registry.register_player(PlayerSpec("mcts", MctsParams, lambda game, cfg, pid: MCTSPlayer(game, cfg, pid)))
    registry.register_player(PlayerSpec("optimal", OptimalParams, lambda game, cfg, pid: OptimalPlayer(game, cfg, pid)))
    registry.register_player(PlayerSpec("llm", LLMParams, _llm_builder(llm), tracker=LLMPlayerTracker()))
