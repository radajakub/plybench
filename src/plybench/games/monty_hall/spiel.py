from __future__ import annotations

from typing import Any

import numpy as np
import pyspiel as sp

# The game show of n closed doors: chance hides a car behind one, the player picks a door, the host opens another one, the player makes a final choice,
# and wins (+1) if that door hides the car, else loses (-1). Params:
# - num_doors: n >= 3
# - final: `switch` (stay with the first door or switch to the other closed one; three doors only) or `pick` (any closed door)
# - host: `standard` (knows the car and opens a goat door, uniformly among those), `ignorant` (opens a door other than the player's uniformly, so it may
#   reveal the car) or `hell` (opens a goat door, uniformly, only when the player's first pick hides the car, else no door; needs `pick`)
# The state holds only ints and None: OpenSpiel serializes a Python state as its history and a pickle of its attributes.

GAME_NAME = "monty_hall"
_DEFAULT_PARAMS = {"num_doors": 3, "final": "switch", "host": "standard"}

_GAME_TYPE = sp.GameType(
    short_name=GAME_NAME,
    long_name="Monty Hall",
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
    parameter_specification=_DEFAULT_PARAMS,
)


def check_params(num_doors: int, final: str, host: str) -> None:
    if num_doors < 3:
        raise ValueError(f"Monty Hall needs at least 3 doors, not {num_doors}")
    if final not in ("switch", "pick"):
        raise ValueError(f"Invalid final choice: {final}")
    if final == "switch" and num_doors != 3:
        raise ValueError(f"The final choice `switch` names one other closed door, so it needs 3 doors, not {num_doors}")
    if host not in ("standard", "ignorant", "hell"):
        raise ValueError(f"Invalid host: {host}")
    if host == "hell" and final != "pick":
        raise ValueError("The host `hell` may open no door, which leaves no single other door to switch to, so it needs the final choice `pick`")


class MontyHallSpielGame(sp.Game):
    def __init__(self, params: dict[str, Any] | None = None) -> None:

        params = {**_DEFAULT_PARAMS, **(params or {})}

        self.num_doors = int(params["num_doors"])
        self.final = str(params["final"])
        self.host = str(params["host"])

        check_params(self.num_doors, self.final, self.host)

        # actions: door d is d, then stay and switch; the host's chance outcome `no door` is num_doors
        self.stay, self.switch = self.num_doors, self.num_doors + 1

        info = sp.GameInfo(
            num_distinct_actions=self.num_doors + 2,
            max_chance_outcomes=self.num_doors + (self.host == "hell"),
            num_players=1,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=None,
            max_game_length=4,
        )

        super().__init__(_GAME_TYPE, info, params)

    def new_initial_state(self) -> MontyHallSpielState:
        return MontyHallSpielState(self)

    def make_py_observer(self, iig_obs_type: Any = None, params: Any = None) -> MontyHallObserver:
        return MontyHallObserver()


class MontyHallSpielState(sp.State):
    def __init__(self, game: MontyHallSpielGame) -> None:
        super().__init__(game)
        self.num_doors = game.num_doors
        self.final_mode = game.final
        self.host = game.host
        self.stay, self.switch = game.stay, game.switch
        self.car: int | None = None
        self.picked: int | None = None
        self.opened: int | None = None
        self.final: int | None = None

    @property
    def no_door(self) -> int:
        # a property, not an attribute: the attributes are pickled into every recorded state, so a new one would change the records of the other variants
        return self.num_doors

    def current_player(self) -> int:
        if self.final is not None:
            return sp.PlayerId.TERMINAL
        if self.car is None or (self.picked is not None and self.opened is None):
            return sp.PlayerId.CHANCE
        return 0

    def _closed(self) -> list[int]:
        return [door for door in range(self.num_doors) if door != self.opened]

    def _legal_actions(self, player: int) -> list[int]:
        if self.picked is None:
            return list(range(self.num_doors))
        if self.final_mode == "switch":
            return [self.stay, self.switch]
        return self._closed()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        if self.car is None:
            return [(door, 1.0 / self.num_doors) for door in range(self.num_doors)]

        if self.host == "hell" and self.picked != self.car:
            return [(self.no_door, 1.0)]

        doors = [door for door in range(self.num_doors) if door != self.picked and (self.host == "ignorant" or door != self.car)]

        return [(door, 1.0 / len(doors)) for door in doors]

    def _apply_action(self, action: int) -> None:
        if self.car is None:
            self.car = action
        elif self.picked is None:
            self.picked = action
        elif self.opened is None:
            self.opened = action
        elif action == self.stay:
            self.final = self.picked
        elif action == self.switch:
            self.final = next(door for door in self._closed() if door != self.picked)
        else:
            self.final = action

    def _action_to_string(self, player: int, action: int) -> str:
        if player == sp.PlayerId.CHANCE:
            if self.car is None:
                return f"car behind door {action + 1}"
            return "host opens no door" if action == self.no_door else f"host opens door {action + 1}"

        if action == self.stay:
            return "stay"

        if action == self.switch:
            return "switch"

        return f"door {action + 1}"

    def is_terminal(self) -> bool:
        return self.final is not None

    def returns(self) -> list[float]:
        if self.final is None:
            return [0.0]
        return [1.0 if self.final == self.car else -1.0]

    def information_state(self) -> str:
        # what the player knows, doors numbered from 1: its picks, the opened door (`none` if the host opened no door) and what it hides, and the car once
        # the game is over
        def door(value: int | None) -> str:
            return "-" if value is None else str(value + 1)

        if self.opened == self.no_door:
            opened, behind = "none", "-"
        else:
            opened, behind = door(self.opened), "-" if self.opened is None else ("car" if self.opened == self.car else "goat")

        car = door(self.car) if self.final is not None else "-"

        return f"picked={door(self.picked)} opened={opened} behind={behind} final={door(self.final)} car={car}"

    def __str__(self) -> str:
        return f"car={self.car} picked={self.picked} opened={self.opened} final={self.final}"


class MontyHallObserver:
    # the information state, which has perfect recall; the observation is the same string
    def __init__(self) -> None:
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, Any] = {}

    def set_from(self, state: MontyHallSpielState, player: int) -> None:
        pass

    def string_from(self, state: MontyHallSpielState, player: int) -> str:
        return state.information_state()


sp.register_game(_GAME_TYPE, MontyHallSpielGame)
