from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.common.enums import GameResults
from plybench.configs.game_config import GameConfig
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.player_tracker import PlayerTrackerResolver

TransformerT = TypeVar("TransformerT", bound=InterfaceTransformer)
AdapterT = TypeVar("AdapterT", bound=PromptAdapter)


class TurnBasedEngine(ABC, Generic[TransformerT, AdapterT]):
    def __init__(
        self,
        game_config: GameConfig,
        game: TurnBasedGame,
        interface_transformer: TransformerT,
        prompt_adapter: AdapterT,
        action_class: type[InterfaceAction],
        observation_class: type[InterfaceObservation],
    ) -> None:
        self.game_config = game_config
        self.game = game
        self.interface_transformer = interface_transformer
        self.prompt_adapter = prompt_adapter
        self.action_class = action_class
        self.observation_class = observation_class
        # set by Registry.build_engine; used to resolve per-player trackers when recording moves.
        # None when the engine is constructed directly (no per-player `data` recorded). Typed as the
        # narrow resolver Protocol (which Registry satisfies) so the engine does not depend on Registry.
        self.trackers: PlayerTrackerResolver | None = None

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @property
    def game_type(self) -> str:
        return self.game.game_type

    def _get_ending(self, player: int) -> InterfaceObservation:
        # terminal observation from `player`'s POV, recorded in the tracker's ending
        return self.observation_class.from_openspiel(self.game.get_observation(player), self.interface_transformer)

    def _get_game_result(self) -> tuple[GameResults, GameResults]:
        reward1, reward2 = self.game.get_rewards()
        if reward1 > reward2:
            return GameResults.WIN, GameResults.LOSS
        if reward1 < reward2:
            return GameResults.LOSS, GameResults.WIN
        return GameResults.DRAW, GameResults.DRAW

    async def play(self, players: tuple[Player, Player], game_callbacks: GameCallbacks | None = None, game_round: int = 1, verbose: bool = False) -> GameTracker:
        game_callbacks = game_callbacks if game_callbacks is not None else GameCallbacks()

        self.reset()
        for player in players:
            player.initialize_policy(self.game, self.prompt_adapter)

        other_params = self.interface_transformer.get_other_params()
        tracker = GameTracker(
            game_round,
            players[0].player_config,
            players[1].player_config,
            self.game.params,
            [],
            None,
            0,
            other_params,
        )

        game_callbacks.on_game_start(tracker)

        pid = 0
        while not self.game.is_terminal():
            # (1) openspiel observation and legal moves
            pid = self.game.get_player()
            os_observation: OpenSpielObservation = self.game.get_observation(pid)
            os_moves: list[OpenSpielAction] = self.game.get_legal_moves(pid)

            # (2) convert to interface observation and actions
            observation: InterfaceObservation = self.observation_class.from_openspiel(os_observation, self.interface_transformer)
            actions: list[InterfaceAction] = [self.action_class.from_openspiel(move, self.interface_transformer) for move in os_moves]

            # (3) query the player
            player = players[pid]

            # (4) player callback before the move is executed
            game_callbacks.on_before_move(player, observation, actions)

            # (5) get the output from the player
            player_output = await player(self.game, observation, actions)

            # (6) record the move in the game tracker
            tracker.add_move(player.player_config, observation, player_output, self.game.serialize_state(), self.trackers)

            # (7) player callback after the move is executed
            game_callbacks.on_after_move(player, player_output, tracker.steps[-1])

            if verbose:
                print(player.format_output(player_output))

            # (5) handle a failed move (illegal / malformed) and exit early
            if player_output.action is None:
                tracker.add_fail(player.player_config, self._get_ending(1 - pid))
                results = (GameResults.MY_FAIL, GameResults.OPPONENT_FAIL) if pid == 0 else (GameResults.OPPONENT_FAIL, GameResults.MY_FAIL)
                game_callbacks.on_game_end(tracker, results)
                return tracker

            # (6) apply the action
            self.game.step(player_output.action.to_openspiel())

        result1, result2 = self._get_game_result()
        tracker.end_game(result1, self._get_ending(1 - pid))
        game_callbacks.on_game_end(tracker, (result1, result2))

        return tracker
