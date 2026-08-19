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
from plybench.llm import LLM, LLMMessage, LLMResponse, ModelConfig, Provider
from plybench.llm.model_config import options_to_string, parse_options
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.trackers.player_tracker import PlayerTracker


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
        options = parse_options(":".join(parts[4:]))
        model = ModelConfig(Provider.from_value(parts[2]), parts[3], options)
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
        data: dict[str, Any] = {}
        if player_output.reasoning_trace:
            data["reasoning_trace"] = player_output.reasoning_trace
        if player_output.full_output:
            data["full_output"] = player_output.full_output
        if player_output.system_message:
            data["system_message"] = player_output.system_message
        if player_output.prompt_message:
            data["prompt_message"] = player_output.prompt_message
        return data
