from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from plybench.common.enums import ObservationType, OutputStrategies
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.output_strategy import OutputStrategy
from plybench.core.prompt_adapter import PromptAdapter
from plybench.llm import LLM, LLMCallOptions, LLMMessage, LLMResponse, ModelConfig, Provider
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.trackers.player_tracker import PlayerTracker
from plybench.trackers.step_data import StepData
from plybench.utils.text import extract_params, to_bool


def _parse_options(options_string: str) -> LLMCallOptions:
    params = extract_params(options_string)
    reasoning_effort = params.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort not in ("minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError(f"Invalid reasoning effort: {reasoning_effort}")
    return LLMCallOptions(
        reasoning_effort=reasoning_effort,
        thinking_enabled=to_bool(params.get("thinking_enabled", False)),
        max_tokens=int(params["max_tokens"]) if "max_tokens" in params else None,
        temperature=float(params["temperature"]) if "temperature" in params else None,
    )


def options_to_string(options: LLMCallOptions) -> str:
    parts: list[str] = []
    if options.thinking_enabled:
        parts.append("thinking_enabled=True")
    if options.reasoning_effort is not None:
        parts.append(f"reasoning_effort={options.reasoning_effort}")
    if options.temperature is not None:
        parts.append(f"temperature={options.temperature}")
    if options.max_tokens is not None:
        parts.append(f"max_tokens={options.max_tokens}")
    return ",".join(parts)


@dataclass(frozen=True, eq=True)
class LLMParams(PlayerParams):
    observation_type: ObservationType
    output_strategy: OutputStrategies
    model: ModelConfig

    @classmethod
    def from_string(cls, params_string: str) -> LLMParams:
        # format: <observation>:<strategy>:<provider>:<model>:<options>
        parts = params_string.split(":")
        if len(parts) < 4:
            raise ValueError(f"Invalid LLM params: {params_string!r} (need observation:strategy:provider:model[:options])")
        observation_type = ObservationType.from_value(parts[0])
        output_strategy = OutputStrategies.from_value(parts[1])
        if observation_type is None or output_strategy is None:
            raise ValueError(f"Invalid observation/strategy in {params_string!r}")
        options = _parse_options(":".join(parts[4:]))
        provider = Provider.from_value(parts[2])
        if provider is None:
            raise ValueError(f"Invalid provider in {params_string!r}")
        model = ModelConfig(provider, parts[3], options)
        return cls(observation_type, output_strategy, model)

    def to_string(self) -> str:
        head = f"{self.observation_type.value}:{self.output_strategy.value}:{self.model.provider.value}:{self.model.model_name}"
        return f"{head}:{options_to_string(self.model.options)}"

    @property
    def path_suffix(self) -> str:
        options = options_to_string(self.model.options).replace("=", "_").replace(",", "_")
        head = f"{self.observation_type.value}_{self.output_strategy.value}_{self.model.provider.value}_{self.model.model_name}"
        return f"{head}_{options}" if options else head


class LLMPlayer(Player):
    def __init__(self, player_config: PlayerConfig, output_strategy: OutputStrategy, llm: LLM, identifier: PlayerIdentifier) -> None:
        super().__init__(player_config, identifier)

        self._params = cast(LLMParams, player_config.params)

        self._output_strategy = output_strategy
        self._llm = llm
        self._prompt_adapter: PromptAdapter | None = None

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        self._prompt_adapter = prompt_adapter_template.for_observation_type(self._params.observation_type)

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        assert self._prompt_adapter is not None, "Prompt adapter not initialized; call initialize_policy first"

        system_message, messages = self._prompt_adapter.build_messages(observation, legal_moves, self._output_strategy)

        response = await self._llm.generate(self._params.model, system_message, messages, output_schema=self._output_strategy.get_output_schema())

        return self._process_response(response, legal_moves, system_message, messages)

    def _process_response(self, response: LLMResponse, legal_moves: list[InterfaceAction], system_message: LLMMessage, messages: list[LLMMessage]) -> PlayerOutput:
        assert self._prompt_adapter is not None
        extracted = self._output_strategy.extract(response.output_text)

        selected_action = None
        failure_reason = None
        if extracted.action is None:
            failure_reason = f"Wrong action format ({response.output_text})"
        else:
            selected_action = self._prompt_adapter.lookup_move(legal_moves, extracted.action)
            if selected_action is None:
                failure_reason = f"Illegal action selected ({extracted.action})"

        return PlayerOutput(
            action=selected_action,
            system_message=system_message.content,
            prompt_message="\n".join(message.content for message in messages),
            reasoning_trace="\n".join(response.reasoning),
            full_output=response.output_text,
            input_tokens=response.tokens.input_tokens,
            output_tokens=response.tokens.output_tokens,
            reasoning_tokens=response.tokens.reasoning_tokens,
            failure_reason=failure_reason,
        )

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return player_output.reasoning_trace


class LLMPlayerTracker(PlayerTracker):
    def record(self, player_output: PlayerOutput) -> dict[str, Any]:
        return StepData.from_output(player_output).to_dict()
