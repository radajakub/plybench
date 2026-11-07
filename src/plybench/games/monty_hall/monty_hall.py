from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from plybench.configs.game_config import GameConfig
from plybench.configs.game_params import GameParams
from plybench.core.engine import TurnBasedEngine
from plybench.core.game import OpenSpielAction, OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation, InterfaceTransformer
from plybench.core.llm_interface import LLMObservation
from plybench.core.prompt_adapter import SinglePlayerPromptAdapter
from plybench.games.monty_hall.spiel import GAME_NAME, check_params, parse_car, parse_offer
from plybench.utils.text import extract_params
from plybench.visualization.doors import DoorPrinter


@dataclass(frozen=True, eq=True)
class MontyHallGameParams(GameParams):
    num_doors: int = 3
    final: str = "switch"
    host: str = "standard"
    offer: Fraction | None = None  # with the host `tempting` only: how often it opens a goat door when the first pick does not hide the car
    car: tuple[int, ...] | None = None  # the car's weights per door; None is uniform
    prize: str = "car"  # what the final door must hide to win
    rules: str = "stated"  # which rules the prompt hides (`hidden`: the host's and the car's, or one of them); only the prompt changes, not the game

    @classmethod
    def from_string(cls, params_string: str) -> MontyHallGameParams:
        params = extract_params(params_string)
        num_doors, final, host = int(params.get("num_doors", 3)), params.get("final", "switch"), params.get("host", "standard")
        prize, rules = params.get("prize", "car"), params.get("rules", "stated")

        check_params(num_doors, final, host, prize)
        offer = parse_offer(params.get("offer", ""), host)
        car = parse_car(params.get("car", ""), num_doors)

        if rules not in _RULES:
            raise ValueError(f"Invalid rules: {rules}; expected one of {', '.join(_RULES)}")

        return cls(num_doors=num_doors, final=final, host=host, offer=offer, car=car, prize=prize, rules=rules)

    def _later_params(self) -> dict[str, str]:
        # the params added after the first records, only where they differ from their defaults, so the strings and result paths of the earlier
        # variants stay as they were; a `tempting` host always names its offer
        params = {}
        if self.offer is not None:
            params["offer"] = str(self.offer)
        if self.car is not None:
            params["car"] = ";".join(str(weight) for weight in self.car)
        if self.prize != "car":
            params["prize"] = self.prize
        if self.rules != "stated":
            params["rules"] = self.rules
        return params

    def to_string(self) -> str:
        later = "".join(f",{name}={value}" for name, value in self._later_params().items())
        return f"num_doors={self.num_doors},final={self.final},host={self.host}" + later

    @property
    def path_suffix(self) -> str:
        # values only, as before, except for the names that tell `offer` and `car` apart; `/` and `;` become `-`
        later = "".join(f"_{name}_{value}" if name in ("offer", "car") else f"_{value}" for name, value in self._later_params().items())
        return f"{self.num_doors}_doors_{self.final}_{self.host}" + later.replace("/", "-").replace(";", "-")


class MontyHallGame(TurnBasedGame):
    def __init__(self, params: MontyHallGameParams) -> None:
        # the later params only where set: the game params name the minimax cache, so an earlier variant keeps its file and a new one gets its own
        later = {name: value for name, value in params._later_params().items() if name != "rules"}
        super().__init__(game_type="monty_hall", game_name=GAME_NAME, params={"num_doors": params.num_doors, "final": params.final, "host": params.host, **later})


class MontyHallAction(InterfaceAction):
    # a door ("door 2", numbered from 1), "stay" or "switch"
    @staticmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> MontyHallAction:
        return MontyHallAction(action.string, action.number, interface_transformer)

    def __init__(self, name: str, number: int, interface_transformer: InterfaceTransformer) -> None:
        super().__init__(number=number, interface_transformer=interface_transformer)
        self.name = name

    def to_openspiel(self) -> OpenSpielAction:
        return OpenSpielAction(self.number, self.name)


class MontyHallObservation(InterfaceObservation):
    # the player's information state; doors numbered from 1, None while unknown; no_door when the host opened no door
    _PATTERN = re.compile(r"picked=(\S+) opened=(\S+) behind=(\S+) final=(\S+) car=(\S+)")

    @staticmethod
    def _door(value: str) -> int | None:
        return None if value == "-" else int(value)

    @staticmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> MontyHallObservation:
        match = MontyHallObservation._PATTERN.fullmatch(observation.state)
        if not match:
            raise ValueError(f"Invalid Monty Hall information state: {observation.state}")
        picked, opened, behind, final, car = match.groups()
        i_actions = [MontyHallAction.from_openspiel(action, interface_transformer) for action in observation.i_actions]
        door = MontyHallObservation._door
        no_door = opened == "none"
        opened_door = None if no_door else door(opened)
        return MontyHallObservation(observation, door(picked), opened_door, no_door, None if behind == "-" else behind, door(final), door(car), i_actions, interface_transformer)

    def __init__(
        self,
        os_observation: OpenSpielObservation,
        picked: int | None,
        opened: int | None,
        no_door: bool,
        behind: str | None,
        final: int | None,
        car: int | None,
        i_actions: list[MontyHallAction],
        interface_transformer: InterfaceTransformer,
    ) -> None:
        super().__init__(os_observation, i_actions, [], interface_transformer)
        self.picked = picked
        self.opened = opened
        self.no_door = no_door
        self.behind = behind
        self.final = final
        self.car = car


class MontyHallTransformer(InterfaceTransformer[MontyHallAction, MontyHallObservation, []]):
    printer = DoorPrinter()

    def __init__(self, num_doors: int) -> None:
        self.num_doors = num_doors

    def _inner_llm_action(self, action: MontyHallAction) -> str:
        return action.name.replace("door ", "door:")

    def display_action(self, action: MontyHallAction) -> str:
        return action.name

    def _inner_llm_state(self, observation: MontyHallObservation) -> str:
        return observation.os_observation.state

    def _inner_llm_partial_states(self, observation: MontyHallObservation) -> list[str]:
        if observation.picked is None:
            return ["you have not picked a door yet"]
        states = [f"you picked door {observation.picked}"]
        if observation.opened is not None:
            states.append(f"the host opened door {observation.opened}, which hides a {observation.behind}")
        if observation.no_door:
            states.append("the host opened no door")
        if observation.final is not None:
            states.append(f"your final choice is door {observation.final}, and the car is behind door {observation.car}")
        return states

    def _inner_llm_positions(self, observation: MontyHallObservation) -> tuple[list[str], list[str]]:
        return [], []

    def _inner_display_state(self, observation: MontyHallObservation) -> str:
        # what the player sees: a closed door is "?"; the opened door shows what it hides; once the game is over, every door is opened
        def content(door: int) -> str:
            if observation.car is not None:
                return "car" if door == observation.car else "goat"
            if door == observation.opened and observation.behind is not None:
                return observation.behind
            return "?"

        def marks(door: int) -> list[str]:
            return [name for name, marked in (("picked", observation.picked), ("opened", observation.opened), ("final", observation.final)) if marked == door]

        doors = range(1, self.num_doors + 1)
        drawing = MontyHallTransformer.printer([content(door) for door in doors], [marks(door) for door in doors])
        return f"{drawing}\n{observation.os_observation.state}"

    def reset(self) -> None:
        pass

    def set_state(self) -> None:
        pass

    def get_other_params(self) -> dict[str, Any] | None:
        return None


MONTY_HALL_HEAD_PROMPT = """
You are a contestant on a game show. In front of you are {num_doors} closed doors, numbered 1 to {num_doors}.
{car_rule}
First, you pick a door. {host_rule}
Finally, {final_rule} {prize_rule}
"""

_HOST_RULES = {
    "standard": "Then the host opens one of the doors you did not pick. The host knows where the car is and always opens a door that hides a goat, choosing uniformly at random when there are several such doors.",
    "ignorant": "Then the host opens one of the doors you did not pick. The host does not know where the car is and opens one of the doors you did not pick uniformly at random, so it may reveal the car.",
    "hell": "Then the host may open one of the doors you did not pick. The host knows where the car is and opens a door only when your first pick hides the car, choosing uniformly at random among the other doors, which all hide goats; otherwise the host opens no door.",
    "tempting": "Then the host may open one of the doors you did not pick. The host knows where the car is. When your first pick hides the car, the host always opens one of the other doors, choosing uniformly at random; they all hide goats. Otherwise, with probability {offer}, the host opens a door you did not pick that hides a goat, choosing uniformly at random among such doors, and with probability {no_offer} the host opens no door.",
}

_PRIZE_RULES = {"car": "You win the car if your final door hides it.", "goat": "You win if your final door hides a goat."}

# the rule of every host, hidden: it names what the host may do, but not when
_HIDDEN_HOST_RULE = "Then the host may open one of the doors you did not pick. You are not told how the host decides whether to open a door or which door to open."

_UNIFORM_CAR_RULE = "A car is behind one door, placed uniformly at random, and a goat is behind each of the others."
_HIDDEN_CAR_RULE = "A car is behind one door, and a goat is behind each of the others. You are not told how the car is placed."

# which rules the prompt hides: none, both, the host's only or the car's only
_RULES = ("stated", "hidden", "hidden_host", "hidden_car")


def _car_rule(car: tuple[int, ...] | None) -> str:
    # the car's placement, stated; weights as exact fractions
    if car is None:
        return _UNIFORM_CAR_RULE
    chances = ", ".join(f"behind door {door} with probability {Fraction(weight, sum(car))}" for door, weight in enumerate(car, start=1))
    return f"A car is behind one door, and a goat is behind each of the others. The car is placed at random: {chances}."


_FINAL_RULES = {
    "switch": "you either stay with the door you picked or switch to the other closed door.",
    "pick": "you choose any door that is still closed, which may be the door you picked first.",
}


class MontyHallPromptAdapter(SinglePlayerPromptAdapter[[MontyHallGameParams]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=MONTY_HALL_HEAD_PROMPT, use_partial_state=True, position_name="")
        self.final = "switch"

    def action_format(self) -> str:
        if self.final == "switch":
            # one action per answer: an example of both kinds side by side ("<door:1>, <switch>") was copied as one answer
            return "<door:x> when you pick a door, or <stay> or <switch> when you make the final choice; one action only, e.g., <door:2>"
        return "<door:x>, e.g., <door:1>, <door:3>"

    def restart_prompt(self, params: MontyHallGameParams) -> None:
        self.final = params.final
        if params.rules in ("hidden", "hidden_host"):
            host_rule = _HIDDEN_HOST_RULE
        else:
            offer = params.offer or Fraction(0)
            host_rule = _HOST_RULES[params.host].format(offer=offer, no_offer=1 - offer)
        car_rule = _HIDDEN_CAR_RULE if params.rules in ("hidden", "hidden_car") else _car_rule(params.car)
        self.head_prompt = self.head_prompt_template.format(
            num_doors=params.num_doors, car_rule=car_rule, host_rule=host_rule, final_rule=_FINAL_RULES[params.final], prize_rule=_PRIZE_RULES[params.prize]
        )

    def observation_prompt(self, observation: LLMObservation) -> str:
        # the host is chance, not a player, so a list of actions would hide the door it opened: both observation types tell the information state
        return self._partial_states_prompt(observation)


class MontyHallEngine(TurnBasedEngine[MontyHallTransformer, MontyHallPromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        params = cast(MontyHallGameParams, game_config.params)
        prompt_adapter = MontyHallPromptAdapter()
        prompt_adapter.restart_prompt(params)
        super().__init__(game_config, MontyHallGame(params), MontyHallTransformer(params.num_doors), prompt_adapter, MontyHallAction, MontyHallObservation)

    def reset(self) -> None:
        self.game.reset()
