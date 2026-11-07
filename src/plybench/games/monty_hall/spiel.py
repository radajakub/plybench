from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pyspiel as sp

# Monty Hall as a one-player OpenSpiel game with chance and hidden information. There are n closed doors, a car behind one and a goat behind each other.
# A game has four steps:
#   1. chance places the car (param `car`);
#   2. the player picks a door;
#   3. chance, as the host, opens one of the doors the player did not pick, or no door (param `host`);
#   4. the player makes the final choice (param `final`), and the game ends: +1 if the final door hides the prize (param `prize`), else -1.
# The player sees its first pick, the opened door and what it hides, and its final choice; where the car is, only once the game is over.
#
# Params:
# - num_doors: n >= 3.
# - final: `switch` (stay with the first door or switch to the other closed door; three doors only) or `pick` (any closed door, the first pick included).
# - host: how the host opens a door. The host's rule always speaks of the car, also when the prize is a goat:
#   - `standard`: knows where the car is and always opens a goat door, uniformly among the doors the player did not pick that hide a goat.
#   - `ignorant`: does not know where the car is and opens one of the doors the player did not pick, uniformly, so it may reveal the car.
#   - `hell`: knows where the car is and opens a door only when the first pick hides the car (uniformly among the others, all goats); else no door.
#     An opened door then means the first pick hides the car. Needs `pick`: when no door is opened, two or more other doors are closed.
#   - `tempting`: knows where the car is. When the first pick hides the car, always opens a door, uniformly among the others (all goats). When it does
#     not, opens a goat door with probability `offer` (uniformly among the goat doors the player did not pick), else no door. Needs `pick`, as `hell`.
#     An opened door is evidence for the first pick: it hides the car with 1 / (1 + (n - 1) * offer), so staying is the best choice after an opened
#     door when offer < 1 / (n - 1). With 3 doors and the default offer 1/4: after an opened door, staying wins 2/3 and the other closed door 1/3,
#     the classic answer reversed. `hell` is the limit offer = 0, and `standard` with `pick` is offer = 1; neither is a valid offer here.
# - offer: with `tempting` only, the probability above, in (0, 1); a fraction or a decimal (`1/4`, `0.25`); default 1/4.
# - car: where chance places the car. Positive integer weights of the doors, separated by `;` (`3;1;1`: door 1 with 3/5, doors 2 and 3 with 1/5
#   each). Empty or equal weights are uniform. A biased car makes the first pick matter: with `3;1;1`, 3 doors and the standard host, picking door 2
#   or 3 and switching wins 4/5, picking door 1 and switching 2/5, picking door 1 and staying 3/5.
# - prize: `car` (default) or `goat`: the player wins if the final door hides the prize. With `goat`, 3 doors and the standard host, the other closed
#   door hides the car whenever the first pick hides a goat, so staying wins 2/3 and switching 1/3, the classic answer reversed.
#
# Win probabilities of the optimal policy, 3 doors, uniform car (checked by the solver in tests/test_monty_hall.py): standard 2/3 (switch); ignorant
# 1/3 (switching and staying win 1/2 after a goat); hell with `pick` 2/3; tempting with `pick` and offer 1/4 7/12 (stay after an opened door, else
# either other door); standard with prize goat 2/3 (stay).
#
# The state holds only ints and None: OpenSpiel serializes a Python state as its history and a pickle of its attributes. The params `car`, `offer`
# and `prize` are read from the game, never kept in the state, and are passed to OpenSpiel only when they differ from their defaults, so the game
# strings and recorded states of the variants from before they existed do not change.

GAME_NAME = "monty_hall"
_DEFAULT_PARAMS = {"num_doors": 3, "final": "switch", "host": "standard", "car": "", "offer": "", "prize": "car"}

# the params added after the first records: left out of the game string while at their defaults
_LATER_DEFAULTS = {"car": "", "offer": "", "prize": "car"}

DEFAULT_OFFER = Fraction(1, 4)

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


def parse_car(value: str, num_doors: int) -> tuple[int, ...] | None:
    # the car's weights per door, or None when the car is placed uniformly, so that equal weights and no weights are one game
    if not value:
        return None
    weights = tuple(int(weight) for weight in value.split(";"))
    if len(weights) != num_doors:
        raise ValueError(f"The car needs one weight per door, {num_doors}, not {len(weights)}: {value}")
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"The car's weights must be positive: {value}")
    return None if len(set(weights)) == 1 else weights


def parse_offer(value: str, host: str) -> Fraction | None:
    # the probability that a `tempting` host opens a goat door when the first pick does not hide the car; None for the other hosts
    if host != "tempting":
        if value:
            raise ValueError(f"Only the host `tempting` takes an offer, not `{host}`")
        return None
    offer = Fraction(value) if value else DEFAULT_OFFER
    if not 0 < offer < 1:
        raise ValueError(f"The offer must be in (0, 1), not {value}; offer 0 is the host `hell`, offer 1 the host `standard`")
    return offer


def check_params(num_doors: int, final: str, host: str, prize: str = "car") -> None:
    if num_doors < 3:
        raise ValueError(f"Monty Hall needs at least 3 doors, not {num_doors}")
    if final not in ("switch", "pick"):
        raise ValueError(f"Invalid final choice: {final}")
    if final == "switch" and num_doors != 3:
        raise ValueError(f"The final choice `switch` names one other closed door, so it needs 3 doors, not {num_doors}")
    if host not in ("standard", "ignorant", "hell", "tempting"):
        raise ValueError(f"Invalid host: {host}")
    if host in ("hell", "tempting") and final != "pick":
        raise ValueError(f"The host `{host}` may open no door, which leaves no single other door to switch to, so it needs the final choice `pick`")
    if prize not in ("car", "goat"):
        raise ValueError(f"Invalid prize: {prize}")


class MontyHallSpielGame(sp.Game):
    def __init__(self, params: dict[str, Any] | None = None) -> None:

        params = {**_DEFAULT_PARAMS, **(params or {})}

        self.num_doors = int(params["num_doors"])
        self.final = str(params["final"])
        self.host = str(params["host"])

        self.prize = str(params["prize"])

        check_params(self.num_doors, self.final, self.host, self.prize)
        self.car_weights = parse_car(str(params["car"]), self.num_doors)
        self.offer = parse_offer(str(params["offer"]), self.host)

        # actions: door d is d, then stay and switch; the host's chance outcome `no door` is num_doors
        self.stay, self.switch = self.num_doors, self.num_doors + 1

        info = sp.GameInfo(
            num_distinct_actions=self.num_doors + 2,
            max_chance_outcomes=self.num_doors + (self.host in ("hell", "tempting")),
            num_players=1,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=None,
            max_game_length=4,
        )

        # the later params only when set, so the game string, and with it every recorded state, of an earlier variant stays as it was
        super().__init__(_GAME_TYPE, info, {name: value for name, value in params.items() if name not in _LATER_DEFAULTS or value != _LATER_DEFAULTS[name]})

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
            # the weights are read from the game, not kept in the state: the state's attributes are pickled into every recorded state
            weights = cast(MontyHallSpielGame, self.get_game()).car_weights
            if weights is None:
                return [(door, 1.0 / self.num_doors) for door in range(self.num_doors)]
            return [(door, weight / sum(weights)) for door, weight in enumerate(weights)]

        if self.host == "hell" and self.picked != self.car:
            return [(self.no_door, 1.0)]

        if self.host == "tempting" and self.picked != self.car:
            offer = cast(MontyHallSpielGame, self.get_game()).offer
            assert offer is not None
            goats = [door for door in range(self.num_doors) if door not in (self.picked, self.car)]
            return [(door, float(offer / len(goats))) for door in goats] + [(self.no_door, float(1 - offer))]

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
        won = (self.final == self.car) == (cast(MontyHallSpielGame, self.get_game()).prize == "car")
        return [1.0 if won else -1.0]

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
