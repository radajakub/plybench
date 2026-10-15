from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from plyllm import LLM, FailureKind, LLMCallError, LLMMessage, LLMResponse, ModelConfig, Provider, options_to_string, parse_options

from plybench.common.enums import MoveFailure, ObservationType, OutputStrategies
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.output_strategy import OutputStrategy
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.trackers.player_tracker import PlayerTracker
from plybench.trackers.prompt_block import PromptBlock
from plybench.trackers.step_data import StepData

# failures recorded as a failed move rather than raised; see LLMPlayer.__call__. Each is about the answer
# or the prompt, not the transport
_MODEL_FAILURES = {
    FailureKind.UNPARSEABLE: MoveFailure.MALFORMED,
    FailureKind.REFUSAL: MoveFailure.REFUSAL,
    FailureKind.PROMPT_BLOCKED: MoveFailure.PROMPT_BLOCKED,
}
# a blocked prompt is sent again up to this many times: the same request was seen blocked once and
# answered another time, so a block says little about the model
_BLOCK_RETRIES = 3


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
        if options.reasoning_effort is not None and options.reasoning_effort not in ("minimal", "low", "medium", "high", "xhigh", "max"):
            raise ValueError(f"Invalid reasoning effort: {options.reasoning_effort}")
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
    def __init__(self, player_config: PlayerConfig, output_strategy: OutputStrategy, llm: LLM, identifier: PlayerIdentifier, block_log: Path | None = None) -> None:
        super().__init__(player_config, identifier)

        self._params = cast(LLMParams, player_config.params)

        self._output_strategy = output_strategy
        self._llm = llm
        self._prompt_adapter: PromptAdapter | None = None
        # where blocked prompts are recorded, one JSON line per block; None records nothing
        self._block_log = block_log

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        self._prompt_adapter = prompt_adapter_template.for_observation_type(self._params.observation_type)

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        assert self._prompt_adapter is not None, "Prompt adapter not initialized; call initialize_policy first"

        system_message, messages = self._prompt_adapter.build_messages(observation, legal_moves, self._output_strategy)

        try:
            response = await self._generate(system_message, messages)
        except LLMCallError as error:
            # an answer that arrived but cannot be played is the model's move, recorded as a failure like any
            # malformed text. Raising would abort the run, and a resume would replay the game until the model
            # happened to answer, hiding these failures from the results. Transport failures still raise.
            failure = _MODEL_FAILURES.get(error.kind)
            if failure is None or error.response is None:
                raise
            # malformed text is shown as itself, like below; a refusal or a block only has the error to show
            detail = error.response.output_text if error.kind is FailureKind.UNPARSEABLE else str(error)
            return self._player_output(error.response, None, failure, detail, system_message, messages)

        return self._process_response(response, legal_moves, system_message, messages)

    async def _generate(self, system_message: LLMMessage, messages: list[LLMMessage]) -> LLMResponse:
        # a blocked prompt is sent again and each block recorded; the last block raises like any failure
        max_attempts = _BLOCK_RETRIES + 1
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._llm.generate(self._params.model, system_message, messages, output_schema=self._output_strategy.get_output_schema())
            except LLMCallError as error:
                if error.kind is not FailureKind.PROMPT_BLOCKED:
                    raise
                if self._block_log is not None:
                    prompt_message = "\n".join(message.content for message in messages)
                    PromptBlock.from_error(error, self.player_config.to_string(), attempt, max_attempts, system_message.content, prompt_message).append(self._block_log)
                if attempt == max_attempts:
                    raise
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _process_response(self, response: LLMResponse, legal_moves: list[InterfaceAction], system_message: LLMMessage, messages: list[LLMMessage]) -> PlayerOutput:
        assert self._prompt_adapter is not None
        extracted = self._output_strategy.extract(response.output_text)

        if extracted.action is None:
            return self._player_output(response, None, MoveFailure.MALFORMED, response.output_text, system_message, messages)

        selected_action = self._prompt_adapter.lookup_move(legal_moves, extracted.action)
        if selected_action is None:
            return self._player_output(response, None, MoveFailure.ILLEGAL, extracted.action, system_message, messages)

        return self._player_output(response, selected_action, None, "", system_message, messages)

    @staticmethod
    def _player_output(
        response: LLMResponse, action: InterfaceAction | None, failure: MoveFailure | None, failure_detail: str, system_message: LLMMessage, messages: list[LLMMessage]
    ) -> PlayerOutput:
        return PlayerOutput(
            action=action,
            system_message=system_message.content,
            prompt_message="\n".join(message.content for message in messages),
            reasoning_trace="\n".join(response.reasoning),
            full_output=response.output_text,
            input_tokens=response.tokens.input_tokens,
            output_tokens=response.tokens.output_tokens,
            reasoning_tokens=response.tokens.reasoning_tokens,
            failure=failure,
            failure_detail=failure_detail,
        )

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return player_output.reasoning_trace


class LLMPlayerTracker(PlayerTracker):
    def record(self, player_output: PlayerOutput) -> dict[str, Any]:
        return StepData.from_output(player_output).to_dict()
