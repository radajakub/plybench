from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plybench.common.serializable import Serializable
from plybench.configs.toggle_item import ToggleItem


@dataclass(frozen=True, eq=True)
class TrainingHarnessConfig(Serializable):
    game_configs: list[ToggleItem]
    player_configs: list[ToggleItem]
    trainer_configs: list[ToggleItem]
    tester_configs: list[ToggleItem]
    training_configs: list[ToggleItem]
    num_replicates: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingHarnessConfig:
        return cls(
            [ToggleItem.from_dict(game_config) for game_config in data["games"]],
            [ToggleItem.from_dict(player_config) for player_config in data["players"]],
            [ToggleItem.from_dict(trainer_config) for trainer_config in data["trainers"]],
            [ToggleItem.from_dict(tester_config) for tester_config in data["testers"]],
            [ToggleItem.from_dict(training_config) for training_config in data["training_configs"]],
            int(data["num_replicates"]),
        )

    def get_game_configs(self) -> list[str]:
        return ToggleItem.extract_values(self.game_configs)

    def get_player_configs(self) -> list[str]:
        return ToggleItem.extract_values(self.player_configs)

    def get_trainer_configs(self) -> list[str]:
        return ToggleItem.extract_values(self.trainer_configs)

    def get_tester_configs(self) -> list[str]:
        return ToggleItem.extract_values(self.tester_configs)

    def get_training_configs(self) -> list[str]:
        return ToggleItem.extract_values(self.training_configs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": [item.to_dict() for item in self.game_configs],
            "players": [item.to_dict() for item in self.player_configs],
            "trainers": [item.to_dict() for item in self.trainer_configs],
            "testers": [item.to_dict() for item in self.tester_configs],
            "training_configs": [item.to_dict() for item in self.training_configs],
            "num_replicates": self.num_replicates,
        }
