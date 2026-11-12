from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plyllm import LLMMessage, LLMResponse, LLMTokens, ModelConfig, Provider


@dataclass(frozen=True)
class LLMCall:
    # one LLM call, as a player records it when a move takes several, and as a learner records the calls of its propose
    role: str  # what the call was for, named by the agent (e.g. "actor", "critic", "feedback", "apply")
    provider: str  # the plyllm provider, by value
    model: str  # the model's plyllm alias, which prices it
    model_string: str  # the exact vendor model id that answered (LLMResponse.model_string)
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
    # the options the call was made with (plyllm LLMCallOptions); None for an option left to the provider's default. None in records from before it was kept
    options: dict[str, Any] | None = None
    # why the provider stopped, as its API names it, and whether that is a cut at the output limit (plyllm 1.7); None when it reports none
    finish_reason: str | None = None
    truncated: bool = False
    # the transport retries before the answer arrived, and whether it came through a provider's Batch API
    retries: int = 0
    batch: bool = False
    # what the call was about, named by the agent (e.g. {"steps": [1, 2]} or {"key": ...}); None when it names nothing
    about: dict[str, Any] | None = None

    @classmethod
    def from_response(
        cls, role: str, model: ModelConfig, system: LLMMessage, messages: list[LLMMessage], response: LLMResponse, failure: str | None = None, about: dict[str, Any] | None = None
    ) -> LLMCall:
        return cls(
            role,
            model.provider.value,
            model.model_name,
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
            asdict(model.options),
            response.finish_reason,
            response.truncated,
            response.retries,
            response.batch,
            about,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMCall:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def tokens(self) -> LLMTokens:
        return LLMTokens(input_tokens=self.input_tokens, output_tokens=self.output_tokens, reasoning_tokens=self.reasoning_tokens)

    @property
    def model_config(self) -> ModelConfig | None:
        # enough to price the call (LLM.calculate_cost reads the provider and the alias); None for a provider plyllm no longer knows
        provider = Provider.from_value(self.provider)
        return ModelConfig(provider, self.model) if provider is not None else None
