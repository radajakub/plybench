from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.player.player import Player, PlayerIdentifier
from plybench.trackers.player_tracker import PlayerTracker

type PlayerBuilder = Callable[[TurnBasedGame, PlayerConfig, PlayerIdentifier], Player]


@dataclass(frozen=True)
class PlayerSpec:
    key: str
    params_cls: type[PlayerParams]
    # builds the player for a game; the LLM (for AI/agents) is captured in this closure at registration
    build: PlayerBuilder
    # optional per-player tracker deciding what extras to persist on each GameStep
    tracker: PlayerTracker | None = None
