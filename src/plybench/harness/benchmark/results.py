from __future__ import annotations

from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.trackers.result_tracker import ResultTracker


class BenchmarkResults:
    def __init__(self, game_configs: list[GameConfig], player_configs: list[PlayerConfig], opponent_configs: list[PlayerConfig], trackers: list[ResultTracker]) -> None:
        self.game_configs = game_configs
        self.player_configs = player_configs
        self.opponent_configs = opponent_configs
        self.trackers = trackers
        self._index = {self._key(tracker.game, tracker.i, tracker.o): tracker for tracker in trackers}

    @staticmethod
    def _key(game: GameConfig, player: PlayerConfig, opponent: PlayerConfig) -> tuple[str, str, str]:
        return (game.to_string(), player.hash, opponent.hash)

    def find(self, game_config: GameConfig, player_config: PlayerConfig, opponent_config: PlayerConfig) -> ResultTracker:
        key = self._key(game_config, player_config, opponent_config)
        tracker = self._index.get(key)
        if tracker is None:
            raise ValueError(f"No result for game={game_config.to_string()} player={player_config.to_string()} opponent={opponent_config.to_string()}")
        return tracker

    def for_player(self, player_config: PlayerConfig) -> list[ResultTracker]:
        return [tracker for tracker in self.trackers if tracker.i.hash == player_config.hash]

    def for_opponent(self, opponent_config: PlayerConfig) -> list[ResultTracker]:
        return [tracker for tracker in self.trackers if tracker.o.hash == opponent_config.hash]

    def for_game(self, game_config: GameConfig) -> list[ResultTracker]:
        return [tracker for tracker in self.trackers if tracker.game.to_string() == game_config.to_string()]
