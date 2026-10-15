from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from plybench.common.enums import MoveFailure
from plybench.configs.player_config import PlayerConfig
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter

type PlayerIdentifier = Literal["i", "o"]


@dataclass
class PlayerOutput:
    action: InterfaceAction | None = None
    system_message: str = ""
    prompt_message: str = ""
    reasoning_trace: str = ""
    full_output: str = ""
    # None = this player type has no token concept (e.g. a bot); LLM/agent players set real counts
    input_tokens: int | None = None
    output_tokens: int | None = None
    # reasoning_tokens is the reasoning subset of output_tokens (agents/LLM players supply it)
    reasoning_tokens: int | None = None
    # set when action is None; failure_detail is what the move records in brackets after the label
    failure: MoveFailure | None = None
    failure_detail: str = ""
    # freeform per-player extras; the player's registered PlayerTracker decides what to persist
    data: dict[str, Any] | None = None


class Player(ABC):
    @staticmethod
    def _top_border(title: str, max_length: int = 30) -> str:
        remaining = max_length - len(title) - 2
        before = math.ceil(remaining / 2)
        after = remaining - before
        return f"{'=' * before} {title} {'=' * after}"

    @staticmethod
    def _bottom_border(max_length: int = 30) -> str:
        return "=" * max_length

    def __init__(self, player_config: PlayerConfig, identifier: PlayerIdentifier) -> None:
        self.player_config = player_config
        self.identifier = identifier

    @abstractmethod
    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        raise NotImplementedError

    @abstractmethod
    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        raise NotImplementedError

    @abstractmethod
    def format_llm_output(self, player_output: PlayerOutput) -> str:
        raise NotImplementedError

    def format_output(self, player_output: PlayerOutput) -> str:
        selected = player_output.action.to_llm().string if player_output.action is not None else "None"
        parts = [
            Player._top_border(str(self)),
            self.format_llm_output(player_output),
            f"Selected move: {selected}",
            Player._bottom_border(),
        ]
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.player_config.to_string()

    def __repr__(self) -> str:
        return self.__str__()
