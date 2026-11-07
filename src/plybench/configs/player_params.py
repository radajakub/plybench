from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from plyllm import ModelConfig


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

    # a short name for plots and tables (e.g. a plugin player naming its parts); None: plybench builds one from the config
    def display_name(self) -> str | None:
        return None

    # every model the player calls, for a check before a run (providers configured, models known) and for total limits; none by default
    def models(self) -> list[ModelConfig]:
        return []


class CheckpointedParams(PlayerParams):
    @property
    @abstractmethod
    def checkpoint(self) -> Path | None:
        raise NotImplementedError

    @abstractmethod
    def with_checkpoint(self, checkpoint: Path) -> CheckpointedParams:
        raise NotImplementedError
