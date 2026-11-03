from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plyllm import LLMMessage, LLMResponse


@dataclass(frozen=True)
class LLMCall:
    # one LLM call, as a player records it when a move takes several, and as a learner records the calls of its propose
    role: str  # what the call was for, named by the agent (e.g. "actor", "critic", "feedback", "apply")
    model: str  # the exact vendor model id (LLMResponse.model_string)
    system_message: str
    prompt_message: str
    reasoning_trace: str
    full_output: str
    input_tokens: int
    output_tokens: int
    # the reasoning subset of output_tokens
    reasoning_tokens: int
    # the seed the provider took; None if the call was unseeded or the provider takes none
    seed: int | None = None
    # the failure kind (plyllm FailureKind value) when the answer arrived but could not be used; None for a usable answer
    failure: str | None = None

    @classmethod
    def from_response(cls, role: str, system: LLMMessage, messages: list[LLMMessage], response: LLMResponse, failure: str | None = None) -> LLMCall:
        return cls(
            role,
            response.model_string,
            system.content,
            "\n".join(message.content for message in messages),
            "\n".join(response.reasoning),
            response.output_text,
            response.tokens.input_tokens,
            response.tokens.output_tokens,
            response.tokens.reasoning_tokens,
            response.seed,
            failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMCall:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
