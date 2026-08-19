from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from typing import Any

import pyspiel as sp


class OpenSpielAction:
    def __init__(self, number: int, string: str) -> None:
        self.number = number
        self.string = string

    def __str__(self) -> str:
        return f"OSAction[{self.number}]: {self.string}"

    def __repr__(self) -> str:
        return self.__str__()


class OpenSpielObservation:
    def __init__(self, state: str, i_actions: list[OpenSpielAction], o_actions: list[OpenSpielAction], player: int) -> None:
        self.state = state
        self.i_actions = i_actions
        self.o_actions = o_actions
        self.player = player

    def __str__(self) -> str:
        return f"OSObservation:\n{self.state}\nI: {self.i_actions}\nO: {self.o_actions}"

    def __repr__(self) -> str:
        return self.__str__()


class OpenSpielGame(ABC):
    def __init__(self, game_type: str, game_name: str, params: dict[str, Any] | None = None) -> None:
        # game_type is the registry key of the game variant (e.g. 'tic_tac_toe')
        self.game_type = game_type
        self.game_name = game_name
        self.params = params if params is not None else {}

        self.game = sp.load_game(self.game_name, self.params)
        self.reset()

    def reset(self) -> None:
        self.state = self.game.new_initial_state()

    def __len__(self) -> int:
        return self.game.max_game_length()


class TurnBasedState:
    @staticmethod
    def get_player(state: sp.State) -> int:
        return state.current_player()

    @staticmethod
    def get_legal_moves(state: sp.State, player: int) -> list[OpenSpielAction]:
        return [TurnBasedState._to_openspiel_action(state, player, number) for number in state.legal_actions(player)]

    @staticmethod
    def child(state: sp.State, action: int) -> sp.State:
        successor = state.clone()
        successor.apply_action(action)
        return successor

    @staticmethod
    def children(state: sp.State) -> Iterator[sp.State]:
        return (TurnBasedState.child(state, action) for action in state.legal_actions())

    @staticmethod
    def action_histories(state: sp.State, player: int) -> tuple[list[OpenSpielAction], list[OpenSpielAction]]:
        i, o = player, 1 - player
        histories = state.history()
        i_history = [TurnBasedState._to_openspiel_action(state, i, h) for index, h in enumerate(histories) if index % 2 == i]
        o_history = [TurnBasedState._to_openspiel_action(state, o, h) for index, h in enumerate(histories) if index % 2 == o]
        return i_history, o_history

    @staticmethod
    def get_observation(state: sp.State, player: int) -> OpenSpielObservation:
        string = state.observation_string(player)
        i_history, o_history = TurnBasedState.action_histories(state, player)
        return OpenSpielObservation(string, i_history, o_history, player)

    @staticmethod
    def _to_openspiel_action(state: sp.State, player: int, number: int) -> OpenSpielAction:
        return OpenSpielAction(number, state.action_to_string(player, number))

    @staticmethod
    def get_rewards(state: sp.State) -> tuple[float, float]:
        rewards = state.rewards()
        return rewards[0], rewards[1]

    @staticmethod
    def is_terminal(state: sp.State) -> bool:
        return state.is_terminal()


class TurnBasedGame(OpenSpielGame):
    def get_player(self) -> int:
        return TurnBasedState.get_player(self.state)

    def get_legal_moves(self, player: int) -> list[OpenSpielAction]:
        return TurnBasedState.get_legal_moves(self.state, player)

    def is_terminal(self) -> bool:
        return TurnBasedState.is_terminal(self.state)

    def action_histories(self, player: int) -> tuple[list[OpenSpielAction], list[OpenSpielAction]]:
        return TurnBasedState.action_histories(self.state, player)

    def get_observation(self, player: int) -> OpenSpielObservation:
        return TurnBasedState.get_observation(self.state, player)

    def get_rewards(self) -> tuple[float, float]:
        return TurnBasedState.get_rewards(self.state)

    def get_reward_range(self) -> tuple[float, float]:
        return self.game.min_utility(), self.game.max_utility()

    def step(self, action: OpenSpielAction) -> None:
        self.state.apply_action(action.number)

    def serialize_state(self) -> str:
        return sp.serialize_game_and_state(self.game, self.state)

    def deserialize_state(self, state_str: str) -> None:
        self.game, self.state = sp.deserialize_game_and_state(state_str)
