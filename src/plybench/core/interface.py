from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, ParamSpec, TypeVar

from plybench.common.enums import PlayerOrder
from plybench.core.game import OpenSpielAction, OpenSpielObservation
from plybench.core.llm_interface import LLMAction, LLMObservation, LLMPartialState, LLMPosition

ActionT = TypeVar("ActionT", bound="InterfaceAction")
ObservationT = TypeVar("ObservationT", bound="InterfaceObservation")
StateArgs = ParamSpec("StateArgs")


class InterfaceTransformer(ABC, Generic[ActionT, ObservationT, StateArgs]):
    @abstractmethod
    def _inner_llm_action(self, action: ActionT) -> str:
        raise NotImplementedError

    def llm_action(self, action: ActionT) -> LLMAction:
        inner_llm_action = self._inner_llm_action(action)
        return LLMAction(f"<{inner_llm_action}>")

    @abstractmethod
    def display_action(self, action: ActionT) -> str:
        raise NotImplementedError

    @abstractmethod
    def _inner_llm_state(self, observation: ObservationT) -> str:
        raise NotImplementedError

    @abstractmethod
    def _inner_llm_partial_states(self, observation: ObservationT) -> list[str]:
        raise NotImplementedError

    def llm_partial_states(self, observation: ObservationT) -> list[LLMPartialState]:
        return [LLMPartialState(partial_state) for partial_state in self._inner_llm_partial_states(observation)]

    @abstractmethod
    def _inner_llm_positions(self, observation: ObservationT) -> tuple[list[str], list[str]]:
        raise NotImplementedError

    def llm_positions(self, observation: ObservationT) -> tuple[list[LLMPosition], list[LLMPosition]]:
        i_positions, o_positions = self._inner_llm_positions(observation)
        return [LLMPosition(pos) for pos in i_positions], [LLMPosition(pos) for pos in o_positions]

    def llm_observation(self, observation: ObservationT) -> LLMObservation:
        state = self._inner_llm_state(observation)
        partial_states = self.llm_partial_states(observation)
        i_actions = [action.to_llm() for action in observation.i_actions]
        o_actions = [action.to_llm() for action in observation.o_actions]
        i_positions, o_positions = self.llm_positions(observation)
        return LLMObservation(state, partial_states, i_actions, o_actions, i_positions, o_positions, observation.player_order)

    @abstractmethod
    def _inner_display_state(self, observation: ObservationT) -> str:
        raise NotImplementedError

    def display_observation(self, observation: ObservationT) -> str:
        state = self._inner_display_state(observation)
        i_actions = ", ".join(f"{action}" for action in observation.i_actions)
        o_actions = ", ".join(f"{action}" for action in observation.o_actions)
        return f"State:\n{state}\nI: {i_actions}\nO: {o_actions}"

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_other_params(self) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, *args: StateArgs.args, **kwargs: StateArgs.kwargs) -> None:
        raise NotImplementedError


class InterfaceAction(ABC):
    @staticmethod
    @abstractmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> InterfaceAction:
        raise NotImplementedError

    def __init__(self, number: int, interface_transformer: InterfaceTransformer) -> None:
        self.number = number
        self.interface_transformer = interface_transformer

    @abstractmethod
    def to_openspiel(self) -> OpenSpielAction:
        raise NotImplementedError

    def to_llm(self) -> LLMAction:
        return self.interface_transformer.llm_action(self)

    def __str__(self) -> str:
        return self.interface_transformer.display_action(self)

    def __repr__(self) -> str:
        return self.__str__()


class InterfaceObservation(ABC):
    @staticmethod
    @abstractmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> InterfaceObservation:
        raise NotImplementedError

    def __init__(
        self, os_observation: OpenSpielObservation, i_actions: Sequence[InterfaceAction], o_actions: Sequence[InterfaceAction], interface_transformer: InterfaceTransformer
    ) -> None:
        # keep the original observation for the MCTS/optimal players to use
        self.os_observation = os_observation
        self.i_actions = i_actions
        self.o_actions = o_actions
        self.player_order = PlayerOrder.from_int(os_observation.player)
        self.interface_transformer = interface_transformer

    def to_llm(self) -> LLMObservation:
        return self.interface_transformer.llm_observation(self)

    def __str__(self) -> str:
        return self.interface_transformer.display_observation(self)

    def __repr__(self) -> str:
        return self.__str__()
