from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from plybench.common.paths import BenchmarkPathBuilder, ExperimentPathBuilder
from plybench.common.serializable import Saveable
from plybench.configs.game_config import GameConfig
from plybench.configs.parser import ConfigParser
from plybench.configs.player_config import PlayerConfig
from plybench.trackers.game_tracker import GameTracker


class ResultTracker(Saveable):
    @staticmethod
    def split_games_by_starting_player(games: list[GameTracker], player: PlayerConfig) -> tuple[list[GameTracker], list[GameTracker]]:
        i_games = [game for game in games if game.i_player.hash == player.hash]
        o_games = [game for game in games if game.o_player.hash == player.hash]
        return i_games, o_games

    @classmethod
    def new(
        cls,
        experiment: str,
        i: PlayerConfig,
        o: PlayerConfig,
        game: GameConfig,
        n: int,
        parser: ConfigParser,
        path_builder: ExperimentPathBuilder | None = None,
        save_on_record: bool = True,
    ) -> ResultTracker:
        path_builder = path_builder if path_builder is not None else BenchmarkPathBuilder()
        return cls(experiment, i, o, game, n, set(), parser, path_builder=path_builder, save_on_record=save_on_record)

    @classmethod
    def from_dict(cls, data: dict[str, Any], parser: ConfigParser, path_builder: ExperimentPathBuilder) -> ResultTracker:
        return cls(
            data["experiment"],
            parser.player_config(data["i_config"]),
            parser.player_config(data["o_config"]),
            parser.game_config(data["game_config"]),
            data["n_games"],
            {int(x) for x in data["completed"]},
            parser,
            path_builder=path_builder,
        )

    @classmethod
    def load_metadata_only(cls, filepath: Path, parser: ConfigParser, path_builder: ExperimentPathBuilder) -> ResultTracker:
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f), parser, path_builder)

    @classmethod
    def load(cls, filepath: Path, parser: ConfigParser, path_builder: ExperimentPathBuilder) -> ResultTracker:
        tracker = cls.load_metadata_only(filepath, parser, path_builder)
        for game_round in tracker.completed:
            tracker.games[game_round - 1] = tracker.load_game(game_round)
        return tracker

    def __init__(
        self,
        experiment: str,
        i: PlayerConfig,
        o: PlayerConfig,
        game: GameConfig,
        n: int,
        completed: set[int],
        parser: ConfigParser,
        path_builder: ExperimentPathBuilder,
        games: list[GameTracker | None] | None = None,
        save_on_record: bool = True,
    ) -> None:
        self.experiment = experiment
        self.i = i
        self.o = o
        self.game = game
        self.n = n
        self.completed = completed
        self.parser = parser
        self.path_builder = path_builder
        self.save_on_record = save_on_record

        self.base_path = self.path_builder.game_base(self.experiment, self.game, self.i, self.o, self.n)
        self.metadata_path = self.path_builder.metadata(self.base_path)
        self.games: list[GameTracker | None] = games if games is not None else [None for _ in range(n)]

    def load_if_exists(self) -> None:
        if self.metadata_path.exists():
            reference = self.load_metadata_only(self.metadata_path, self.parser, self.path_builder)
            self.completed = reference.completed
            self.games = [None for _ in range(self.n)]
            for game_round in self.completed:
                self.games[game_round - 1] = self.load_game(game_round)

    def record_game(self, game_round: int, game_tracker: GameTracker) -> None:
        if self.games[game_round - 1] is not None:
            raise ValueError(f"Game {game_round} already recorded")
        self.games[game_round - 1] = game_tracker
        if self.save_on_record:
            self.save_game(game_round, game_tracker)

    def load_game(self, game_round: int) -> GameTracker | None:
        game_path = self.path_builder.game_file(self.base_path, game_round)
        if not game_path.exists():
            return None
        with open(game_path, "r") as f:
            return GameTracker.from_dict(json.load(f), self.parser)

    def save_game(self, game_round: int, game_tracker: GameTracker) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)
        game_path = self.path_builder.game_file(self.base_path, game_round)
        temp_path = game_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(game_tracker.to_dict(), f, indent=None)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, game_path)
            if game_round not in self.completed:
                self.completed.add(game_round)
                self.save(str(self.metadata_path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def get_completed_games(self) -> list[int]:
        return sorted(self.completed)

    def get_missing_games(self) -> list[int]:
        return sorted(set(range(1, self.n + 1)) - self.completed)

    def is_game_complete(self, game_round: int) -> bool:
        return game_round in self.completed

    def is_complete(self) -> bool:
        return len(self.get_missing_games()) == 0

    def invert(self) -> ResultTracker:
        return ResultTracker(
            self.experiment,
            self.o,
            self.i,
            self.game,
            self.n,
            self.completed,
            self.parser,
            path_builder=self.path_builder,
            games=self.games,
            save_on_record=self.save_on_record,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "i_config": self.i.to_string(),
            "o_config": self.o.to_string(),
            "game_config": self.game.to_string(),
            "n_games": self.n,
            "completed": list(self.completed),
        }
