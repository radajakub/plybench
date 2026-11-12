from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from plyllm import LLMCallError

from plybench.common.serializable import ThreadSafeAppendable


@dataclass(frozen=True)
class PromptBlock(ThreadSafeAppendable):
    time: str
    player: str
    reason: str
    attempt: int
    max_attempts: int
    input_tokens: int | None
    system_message: str
    prompt_message: str
    # what the step of the blocked move records (GameStep.blocks) to find this entry; "" in entries from before it was kept
    id: str = ""

    @classmethod
    def from_error(cls, error: LLMCallError, player: str, attempt: int, max_attempts: int, system_message: str, prompt_message: str) -> PromptBlock:
        input_tokens = error.response.tokens.input_tokens if error.response is not None else None
        return cls(datetime.now(UTC).isoformat(), player, str(error), attempt, max_attempts, input_tokens, system_message, prompt_message, uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "player": self.player,
            "reason": self.reason,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "input_tokens": self.input_tokens,
            "system_message": self.system_message,
            "prompt_message": self.prompt_message,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptBlock:
        return cls(
            str(data["time"]),
            str(data["player"]),
            str(data["reason"]),
            int(data["attempt"]),
            int(data["max_attempts"]),
            data["input_tokens"],
            str(data["system_message"]),
            str(data["prompt_message"]),
            str(data.get("id", "")),
        )
