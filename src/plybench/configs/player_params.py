from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from plyllm import ModelConfig

from plybench.common.serializable import record_of


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

    # every param as plain JSON, defaults included (record_of): what a session records of the player, and what a resumed session must match
    def describe(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} is not a dataclass, so it must override describe()")
        return record_of(self)


class CheckpointedParams(PlayerParams):
    @property
    @abstractmethod
    def checkpoint(self) -> Path | None:
        raise NotImplementedError

    @abstractmethod
    def with_checkpoint(self, checkpoint: Path) -> CheckpointedParams:
        raise NotImplementedError
