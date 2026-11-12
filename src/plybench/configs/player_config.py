from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from plybench.configs.player_params import PlayerParams


@dataclass(frozen=True, eq=True)
class PlayerConfig:
    key: str
    params: PlayerParams

    def to_string(self) -> str:
        return f"{self.key}:{self.params.to_string()}"

    @property
    def path(self) -> str:
        suffix = self.params.path_suffix
        return f"{self.key}_{suffix}" if suffix else self.key

    @property
    def hash(self) -> str:
        # short, deterministic id derived from the canonical serialization; used only to match a
        # player to its recorded steps (stable across processes, unlike builtin hash()).
        return hashlib.sha256(self.to_string().encode()).hexdigest()[:12]

    def describe(self) -> dict[str, Any]:
        # the player as a session records it: its key, its config string and every param, defaults included
        return {"key": self.key, "config": self.to_string(), "params": self.params.describe()}

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.__str__()
