from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plybench.player.player import PlayerOutput


@dataclass(frozen=True)
class StepData:
    reasoning_trace: str = ""
    full_output: str = ""
    system_message: str = ""
    prompt_message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_output(cls, player_output: PlayerOutput, **extra: Any) -> StepData:
        return cls(
            player_output.reasoning_trace,
            player_output.full_output,
            player_output.system_message,
            player_output.prompt_message,
            {key: value for key, value in extra.items() if value is not None},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StepData:
        data = data if data is not None else {}
        reserved = {field_name for field_name in cls.__dataclass_fields__ if field_name != "extra"}
        return cls(
            str(data.get("reasoning_trace", "")),
            str(data.get("full_output", "")),
            str(data.get("system_message", "")),
            str(data.get("prompt_message", "")),
            {key: value for key, value in data.items() if key not in reserved},
        )

    @property
    def thinking(self) -> str:
        # models that reason outside a thinking block leave the trace empty but the reasoning in the
        # completion, so a reader wanting "what it thought" needs both, in this order
        return self.reasoning_trace or self.full_output

    def to_dict(self) -> dict[str, Any]:
        data = {
            "reasoning_trace": self.reasoning_trace,
            "full_output": self.full_output,
            "system_message": self.system_message,
            "prompt_message": self.prompt_message,
        }
        return {**{key: value for key, value in data.items() if value}, **self.extra}
