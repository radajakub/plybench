from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plybench.common.serializable import Serializable
from plybench.configs.toggle_item import ToggleItem

DEFAULT_OPPONENT = "optimal:stochastic=True"
# who moves first in each round; a name from `COLOUR_RULES` in harness/matchup.py
DEFAULT_COLOUR_RULE = "first_starts"


def parse_seed(data: dict[str, Any]) -> int | None:
    # the base seed of an experiment, from which each game's seed is derived with the game's record path; an
    # experiment without one (absent or null) is unseeded
    seed = data.get("seed")
    return None if seed is None else int(seed)


@dataclass(frozen=True, eq=True)
class BenchmarkConfig(Serializable):
    game_configs: list[ToggleItem]
    player_configs: list[ToggleItem]
    opponent_configs: list[ToggleItem]
    num_games: int
    colour_rule: str = DEFAULT_COLOUR_RULE
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        return cls(
            [ToggleItem.from_dict(game_config) for game_config in data["game_configs"]],
            [ToggleItem.from_dict(player_config) for player_config in data["player_configs"]],
            cls._parse_opponents(data),
            int(data["num_games"]),
            str(data.get("colour_rule", DEFAULT_COLOUR_RULE)),
            parse_seed(data),
        )

    @staticmethod
    def _parse_opponents(data: dict[str, Any]) -> list[ToggleItem]:
        if "opponents" in data:
            return [ToggleItem.from_dict(opponent) for opponent in data["opponents"]]

        # legacy single baseline -> one enabled opponent
        return [ToggleItem(data.get("baseline", DEFAULT_OPPONENT), True)]

    def get_game_configs(self) -> list[str]:
        return [item.value for item in self.game_configs if item.enabled]

    def get_player_configs(self) -> list[str]:
        return [item.value for item in self.player_configs if item.enabled]

    def get_opponent_configs(self) -> list[str]:
        return [item.value for item in self.opponent_configs if item.enabled]

    def get_player_labels(self) -> dict[str, str]:
        # the labels of the players and opponents, by the config string as written
        return {**ToggleItem.extract_labels(self.player_configs), **ToggleItem.extract_labels(self.opponent_configs)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_configs": [item.to_dict() for item in self.game_configs],
            "player_configs": [item.to_dict() for item in self.player_configs],
            "opponents": [item.to_dict() for item in self.opponent_configs],
            "num_games": self.num_games,
            "colour_rule": self.colour_rule,
            "seed": self.seed,
        }
