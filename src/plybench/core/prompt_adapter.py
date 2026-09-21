from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Generic, ParamSpec

from plybench.common.enums import ObservationType
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.llm_interface import LLMAction, LLMObservation
from plybench.core.output_strategy import OutputStrategy
from plybench.llm.message import LLMMessage
from plybench.utils.text import inline_multiline_string

_SYSTEM_PROMPT = """
You are a powerful gaming agent who can make proper decisions to beat the user in gaming tasks.
You are a helpful assistant that strictly follows the user's instructions.
You must answer your questions by choosing one of the legal moves given by the user!
"""


PromptArgs = ParamSpec("PromptArgs")


class PromptAdapter(ABC, Generic[PromptArgs]):
    def __init__(self, head_prompt_template: str, use_partial_state: bool, position_name: str = "positions", order_actions: bool = False) -> None:
        self.system_prompt = inline_multiline_string(_SYSTEM_PROMPT)
        self.head_prompt_template = inline_multiline_string(head_prompt_template)
        self.head_prompt: str | None = None
        self.order_actions = order_actions
        self.observation_type: ObservationType | None = None
        self.position_name = position_name
        self.use_partial_state = use_partial_state

    def for_observation_type(self, observation_type: ObservationType) -> PromptAdapter:
        clone = copy.copy(self)
        clone.observation_type = observation_type
        return clone

    @abstractmethod
    def restart_prompt(self, *args: PromptArgs.args, **kwargs: PromptArgs.kwargs) -> None:
        raise NotImplementedError

    def _i_actions_prompt(self, observation: LLMObservation) -> str:
        if len(observation.i_actions) == 0:
            return "You have not played any actions so far."
        return f"You have played actions: {observation.i_actions_prompt()}."

    def _i_positions_prompt(self, observation: LLMObservation) -> str:
        if len(observation.i_positions) == 0:
            return f"You do not have any {self.position_name} so far."
        return f"You have {self.position_name}: {observation.i_positions_prompt()}."

    def _o_actions_prompt(self, observation: LLMObservation) -> str:
        if len(observation.o_actions) == 0:
            return "Your opponent has not played any actions so far."
        return f"Your opponent has played actions: {observation.o_actions_prompt()}."

    def _o_positions_prompt(self, observation: LLMObservation) -> str:
        if len(observation.o_positions) == 0:
            return f"Your opponent does not have any {self.position_name} so far."
        return f"Your opponent has {self.position_name}: {observation.o_positions_prompt()}."

    def _partial_states_prompt(self, observation: LLMObservation) -> str:
        return f"Currently, {observation.all_partial_states_prompt()}."

    def _observation_prompt_actions(self, observation: LLMObservation) -> str:
        player_position = f"You play as {observation.player_order.value} player."
        i_part = self._i_actions_prompt(observation)
        o_part = self._o_actions_prompt(observation)
        first, second = (i_part, o_part) if observation.player_order.is_first() else (o_part, i_part)
        return " ".join([player_position, first, second]).strip()

    def _observation_prompt_state(self, observation: LLMObservation) -> str:
        player_position = f"You play as {observation.player_order.value} player."
        if self.use_partial_state:
            return " ".join([player_position, self._partial_states_prompt(observation)]).strip()
        i_part = self._i_positions_prompt(observation)
        o_part = self._o_positions_prompt(observation)
        first, second = (i_part, o_part) if observation.player_order.is_first() else (o_part, i_part)
        return " ".join([player_position, first, second]).strip()

    def observation_prompt(self, observation: LLMObservation) -> str:
        match self.observation_type:
            case ObservationType.ACTIONS:
                return self._observation_prompt_actions(observation)
            case ObservationType.STATE:
                return self._observation_prompt_state(observation)
            case _:
                raise ValueError(f"Unsupported observation type: {self.observation_type}")

    @abstractmethod
    def action_format(self) -> str:
        raise NotImplementedError

    def legal_moves_prompt(self, legal_actions: list[LLMAction]) -> str:
        return f"Currently, the legal moves are: {LLMObservation.actions_prompt(legal_actions)}."

    def update_system_prompt(self, new_system_prompt: str) -> None:
        self.system_prompt = inline_multiline_string(new_system_prompt)

    def lookup_move(self, legal_actions: list[InterfaceAction], llm_action: str | None) -> InterfaceAction | None:
        if llm_action is None:
            return None
        for legal_action in legal_actions:
            if legal_action.to_llm().string == llm_action:
                return legal_action
        return None

    @staticmethod
    def _ordered_llm_actions(actions: list[LLMAction]) -> list[LLMAction]:
        return sorted(actions, key=lambda action: action.string)

    def format_observation(self, observation: InterfaceObservation) -> LLMObservation:
        llm_observation = observation.to_llm()
        if self.order_actions:
            llm_observation.i_actions = self._ordered_llm_actions(llm_observation.i_actions)
            llm_observation.o_actions = self._ordered_llm_actions(llm_observation.o_actions)
        return llm_observation

    def format_legal_actions(self, legal_actions: list[InterfaceAction]) -> list[LLMAction]:
        llm_actions = [action.to_llm() for action in legal_actions]
        if self.order_actions:
            llm_actions = self._ordered_llm_actions(llm_actions)
        return llm_actions

    def build_state_content(self, observation: InterfaceObservation) -> str:
        observation_prompt = self.observation_prompt(self.format_observation(observation))
        return f"{self.head_prompt}\n\n{observation_prompt}"

    def build_messages(
        self, observation: InterfaceObservation, legal_actions: list[InterfaceAction], output_strategy: OutputStrategy, context_enrichment: str | None = None
    ) -> tuple[LLMMessage, list[LLMMessage]]:
        system_message = LLMMessage.system(self.system_prompt)

        state_content = self.build_state_content(observation)
        action_format = self.action_format()

        legal_moves_prompt = self.legal_moves_prompt(self.format_legal_actions(legal_actions))
        output_prompt = output_strategy.output_prompt(action_format)

        content = "\n\n".join(x for x in [state_content, context_enrichment, legal_moves_prompt, output_prompt] if x)
        user_message = LLMMessage.user(content)

        return system_message, [user_message]
