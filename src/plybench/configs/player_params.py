from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class PlayerParams(ABC):
    @classmethod
    @abstractmethod
    def from_string(cls, params_string: str) -> PlayerParams:
        raise NotImplementedError

    @abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def path_suffix(self) -> str:
        raise NotImplementedError


class CheckpointedParams(PlayerParams):
    @property
    @abstractmethod
    def checkpoint(self) -> Path | None:
        raise NotImplementedError

    @abstractmethod
    def with_checkpoint(self, checkpoint: Path) -> CheckpointedParams:
        raise NotImplementedError
