"""Toy OpenSpiel games for the tests (chance, one player, hidden information) and a minimal engine that plays any OpenSpiel game."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyspiel as sp

from plybench.configs.game_config import GameConfig
from plybench.configs.game_params import NoGameParams
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.prompt_adapter import PromptAdapter

# a hidden coin (0 with probability 0.7); the player may pay 0.1 to see a signal that equals the coin with probability 0.8, then guesses it: +1 right,
# -1 wrong. Hand-computed: no look 0.4; look 0.5 (signal 0: P = 0.62, guess 0; signal 1: P = 0.38, guess 1)
LOOK, SKIP = 0, 1
COIN_TYPE = sp.GameType(
    short_name="plybench_test_noisy_coin",
    long_name="plybench test noisy coin",
    dynamics=sp.GameType.Dynamics.SEQUENTIAL,
    chance_mode=sp.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=sp.GameType.Information.IMPERFECT_INFORMATION,
    utility=sp.GameType.Utility.GENERAL_SUM,
    reward_model=sp.GameType.RewardModel.TERMINAL,
    max_num_players=1,
    min_num_players=1,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
)
COIN_INFO = sp.GameInfo(num_distinct_actions=2, max_chance_outcomes=2, num_players=1, min_utility=-1.1, max_utility=1.0, utility_sum=None, max_game_length=4)


class CoinGame(sp.Game):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(COIN_TYPE, COIN_INFO, params or {})

    def new_initial_state(self) -> CoinState:
        return CoinState(self)

    def make_py_observer(self, iig_obs_type: Any = None, params: Any = None) -> CoinObserver:
        return CoinObserver()


class CoinState(sp.State):
    def __init__(self, game: CoinGame) -> None:
        super().__init__(game)
        self.coin: int | None = None
        self.looked: bool | None = None
        self.signal: int | None = None
        self.guess: int | None = None

    def current_player(self) -> int:
        if self.guess is not None:
            return sp.PlayerId.TERMINAL
        if self.coin is None or (self.looked and self.signal is None):
            return sp.PlayerId.CHANCE
        return 0

    def _legal_actions(self, player: int) -> list[int]:
        return [0, 1]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        if self.coin is None:
            return [(0, 0.7), (1, 0.3)]
        return [(self.coin, 0.8), (1 - self.coin, 0.2)]

    def _apply_action(self, action: int) -> None:
        if self.coin is None:
            self.coin = action
        elif self.looked is None:
            self.looked = action == LOOK
        elif self.looked and self.signal is None:
            self.signal = action
        else:
            self.guess = action

    def _action_to_string(self, player: int, action: int) -> str:
        if player == sp.PlayerId.CHANCE:
            return f"side {action}"
        return ("look", "skip")[action] if self.looked is None else f"guess {action}"

    def is_terminal(self) -> bool:
        return self.guess is not None

    def returns(self) -> list[float]:
        if self.guess is None:
            return [0.0]
        return [(1.0 if self.guess == self.coin else -1.0) - (0.1 if self.looked else 0.0)]


class CoinObserver:
    # what the player knows: its own choices and the signal, never the coin
    def __init__(self) -> None:
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, Any] = {}

    def set_from(self, state: CoinState, player: int) -> None:
        pass

    def string_from(self, state: CoinState, player: int) -> str:
        return f"looked={state.looked} signal={state.signal} guess={state.guess}"


sp.register_game(COIN_TYPE, CoinGame)


# two players race to 3 from 0, alternating; a move steps 1 for sure or rolls 0 or 2 with equal chance; the first to reach 3 wins (+1), a draw after 6 moves.
# The observation is the whole state, move count included, which the solver's key needs (pig's omits its move counter)
STEP, ROLL = 0, 1
RACE_TYPE = sp.GameType(
    short_name="plybench_test_race",
    long_name="plybench test race",
    dynamics=sp.GameType.Dynamics.SEQUENTIAL,
    chance_mode=sp.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=sp.GameType.Information.PERFECT_INFORMATION,
    utility=sp.GameType.Utility.ZERO_SUM,
    reward_model=sp.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
)
RACE_INFO = sp.GameInfo(num_distinct_actions=2, max_chance_outcomes=3, num_players=2, min_utility=-1.0, max_utility=1.0, utility_sum=0.0, max_game_length=12)


class RaceGame(sp.Game):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(RACE_TYPE, RACE_INFO, params or {})

    def new_initial_state(self) -> RaceState:
        return RaceState(self)

    def make_py_observer(self, iig_obs_type: Any = None, params: Any = None) -> RaceObserver:
        return RaceObserver()


class RaceState(sp.State):
    def __init__(self, game: RaceGame) -> None:
        super().__init__(game)
        self.positions = [0, 0]
        self.mover = 0
        self.rolling = False
        self.moves = 0

    def current_player(self) -> int:
        if self.is_terminal():
            return sp.PlayerId.TERMINAL
        return sp.PlayerId.CHANCE if self.rolling else self.mover

    def _legal_actions(self, player: int) -> list[int]:
        return [STEP, ROLL]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        return [(0, 0.5), (2, 0.5)]

    def _apply_action(self, action: int) -> None:
        if self.rolling or action == STEP:
            self.positions[self.mover] += action if self.rolling else 1
            self.rolling = False
            self.mover = 1 - self.mover
            self.moves += 1
        else:
            self.rolling = True

    def _action_to_string(self, player: int, action: int) -> str:
        return f"rolled {action}" if player == sp.PlayerId.CHANCE else ("step", "roll")[action]

    def is_terminal(self) -> bool:
        return max(self.positions) >= 3 or self.moves >= 6

    def returns(self) -> list[float]:
        if max(self.positions) < 3:
            return [0.0, 0.0]
        return [1.0, -1.0] if self.positions[0] >= 3 else [-1.0, 1.0]


class RaceObserver:
    def __init__(self) -> None:
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, Any] = {}

    def set_from(self, state: RaceState, player: int) -> None:
        pass

    def string_from(self, state: RaceState, player: int) -> str:
        return f"positions={state.positions} mover={state.mover} rolling={state.rolling} moves={state.moves}"


sp.register_game(RACE_TYPE, RaceGame)


class ToyGame(TurnBasedGame):
    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(game_type=name, game_name=name, params=params)


class ToyAction(InterfaceAction):
    @staticmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> ToyAction:
        return ToyAction(action.number, action.string, interface_transformer)

    def __init__(self, number: int, string: str, interface_transformer: InterfaceTransformer) -> None:
        super().__init__(number=number, interface_transformer=interface_transformer)
        self.string = string

    def to_openspiel(self) -> OpenSpielAction:
        return OpenSpielAction(self.number, self.string)


class ToyObservation(InterfaceObservation):
    @staticmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> ToyObservation:
        i_actions = [ToyAction.from_openspiel(action, interface_transformer) for action in observation.i_actions]
        o_actions = [ToyAction.from_openspiel(action, interface_transformer) for action in observation.o_actions]
        return ToyObservation(observation, i_actions, o_actions, interface_transformer)


class ToyTransformer(InterfaceTransformer[ToyAction, ToyObservation, []]):
    def _inner_llm_action(self, action: ToyAction) -> str:
        return action.string

    def display_action(self, action: ToyAction) -> str:
        return action.string

    def _inner_llm_state(self, observation: ToyObservation) -> str:
        return observation.os_observation.state

    def _inner_llm_partial_states(self, observation: ToyObservation) -> list[str]:
        return []

    def _inner_llm_positions(self, observation: ToyObservation) -> tuple[list[str], list[str]]:
        return [], []

    def _inner_display_state(self, observation: ToyObservation) -> str:
        return observation.os_observation.state

    def reset(self) -> None:
        pass

    def set_state(self) -> None:
        pass

    def get_other_params(self) -> dict[str, Any] | None:
        return None


class ToyPromptAdapter(PromptAdapter[[]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template="A toy game.", use_partial_state=False, position_name="", order_actions=False)
        self.head_prompt = self.head_prompt_template

    def action_format(self) -> str:
        return "<action>"

    def restart_prompt(self) -> None:
        pass


class ToyEngine(TurnBasedEngine[ToyTransformer, ToyPromptAdapter]):
    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(GameConfig(name, NoGameParams()), ToyGame(name, params), ToyTransformer(), ToyPromptAdapter(), ToyAction, ToyObservation)

    def reset(self) -> None:
        self.game.reset()
