from __future__ import annotations

from dataclasses import dataclass

from plybench.configs.game_params import GameParams


@dataclass(frozen=True, eq=True)
class GameConfig:
    key: str
    params: GameParams

    def to_string(self) -> str:
        return f"{self.key}:{self.params.to_string()}"

    def at_epoch(self, epoch: int) -> GameConfig:
        # the game a training epoch plays (GameParams.at_epoch); a run is still named by its own game
        params = self.params.at_epoch(epoch)
        return self if params is self.params else GameConfig(self.key, params)

    @property
    def path(self) -> str:
        suffix = self.params.path_suffix
        return f"{self.key}_{suffix}" if suffix else self.key

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.__str__()
