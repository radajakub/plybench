from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class GameParams(ABC):
    @classmethod
    @abstractmethod
    def from_string(cls, params_string: str) -> GameParams:
        raise NotImplementedError

    @abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def path_suffix(self) -> str:
        """The game-specific part of the result path (may be empty); GameConfig prefixes the key."""
        raise NotImplementedError

    def at_epoch(self, epoch: int) -> GameParams:
        # the params a training epoch plays: the same in every epoch, unless a game changes at an epoch (the bandit's switch)
        return self


@dataclass(frozen=True, eq=True)
class NoGameParams(GameParams):
    @classmethod
    def from_string(cls, params_string: str) -> NoGameParams:
        return cls()

    def to_string(self) -> str:
        return ""

    @property
    def path_suffix(self) -> str:
        return ""
