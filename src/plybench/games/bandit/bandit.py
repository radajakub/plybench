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
from plybench.games.bandit.spiel import _DEFAULT_PARAMS, COLOURS, GAME_NAME, check_shift, parse_p, shifted
from plybench.utils.text import extract_params


def _fractions(p: tuple[Fraction, ...]) -> str:
    return ";".join(str(chance) for chance in p)


def _path_fractions(p: tuple[Fraction, ...]) -> str:
    return "_".join(str(chance) for chance in p).replace("/", "-")


@dataclass(frozen=True, eq=True)
class BanditGameParams(GameParams):
    p: tuple[Fraction, ...] = parse_p(str(_DEFAULT_PARAMS["p"]))
    shift: int = 0
    rules: str = "stated"  # `hidden`: the prompt does not give the win probabilities; only the prompt changes, not the game
    # the win probabilities from the training epoch `switch_epoch` on, with the same shift: a training run plays `p` before and `switch` after (at_epoch)
    switch: tuple[Fraction, ...] | None = None
    switch_epoch: int | None = None
    # `omitted`: the prompt does not say that the probabilities never change, which a switch makes false
    stationarity: str = "stated"

    @classmethod
    def from_string(cls, params_string: str) -> BanditGameParams:
        params = extract_params(params_string)
        p = parse_p(params["p"]) if "p" in params else cls.p
        shift, rules, stationarity = int(params.get("shift", 0)), params.get("rules", "stated"), params.get("stationarity", "stated")
        check_shift(shift)
        if rules not in _RULES:
            raise ValueError(f"Invalid rules: {rules}; expected one of {', '.join(_RULES)}")
        if stationarity not in _STATIONARITY:
            raise ValueError(f"Invalid stationarity: {stationarity}; expected one of {', '.join(_STATIONARITY)}")
        if ("switch" in params) != ("switch_epoch" in params):
            raise ValueError("A switch needs both switch (the probabilities after it) and switch_epoch (the first epoch that plays them)")
        switch = parse_p(params["switch"]) if "switch" in params else None
        switch_epoch = int(params["switch_epoch"]) if "switch_epoch" in params else None
        if switch is not None and len(switch) != len(p):
            raise ValueError(f"The switch must give as many probabilities as p: {len(switch)} for {len(p)} buttons")
        # epoch 0 only tests the untrained player, so the first epoch a switch can change is 1
        if switch_epoch is not None and switch_epoch < 1:
            raise ValueError(f"The switch epoch must be at least 1, not {switch_epoch}")
        return cls(p=p, shift=shift, rules=rules, switch=switch, switch_epoch=switch_epoch, stationarity=stationarity)

    def to_string(self) -> str:
        # the probabilities as exact fractions, so a decimal and a fraction are one game; the switch and the stationarity only when set, so the strings
        # of the games before them are unchanged
        switch = f",switch={_fractions(self.switch)},switch_epoch={self.switch_epoch}" if self.switch is not None else ""
        stationarity = f",stationarity={self.stationarity}" if self.stationarity != "stated" else ""
        return f"p={_fractions(self.p)},shift={self.shift},rules={self.rules}{switch}{stationarity}"

    @property
    def path_suffix(self) -> str:
        switch = f"_switch_{_path_fractions(self.switch)}_epoch_{self.switch_epoch}" if self.switch is not None else ""
        stationarity = f"_stationarity_{self.stationarity}" if self.stationarity != "stated" else ""
        return f"{len(self.p)}_buttons_p_{_path_fractions(self.p)}_shift_{self.shift}_{self.rules}{switch}{stationarity}"

    def at_epoch(self, epoch: int) -> BanditGameParams:
        # before the switch the run's own params; from it on a game without a switch that plays the new probabilities
        if self.switch is None or self.switch_epoch is None or epoch < self.switch_epoch:
            return self
        return BanditGameParams(p=self.switch, shift=self.shift, rules=self.rules, stationarity=self.stationarity)

    @property
    def chances(self) -> tuple[Fraction, ...]:
        # the win probability of each button, in the order the buttons are listed
        return shifted(self.p, self.shift)


class BanditGame(TurnBasedGame):
    def __init__(self, params: BanditGameParams) -> None:
        # the rules are left out: they change only the prompt, so a stated and a hidden variant share the game and its minimax cache
        super().__init__(game_type="bandit", game_name=GAME_NAME, params={"p": _fractions(params.p), "shift": params.shift})


class BanditAction(InterfaceAction):
    # a button, "button red"
    @staticmethod
    def from_openspiel(action: OpenSpielAction, interface_transformer: InterfaceTransformer) -> BanditAction:
        return BanditAction(action.string, action.number, interface_transformer)

    def __init__(self, name: str, number: int, interface_transformer: InterfaceTransformer) -> None:
        super().__init__(number=number, interface_transformer=interface_transformer)
        self.name = name

    def to_openspiel(self) -> OpenSpielAction:
        return OpenSpielAction(self.number, self.name)


class BanditObservation(InterfaceObservation):
    # the player's information state: the button it pressed and the reward, None while unknown
    _PATTERN = re.compile(r"pressed=(\S+) reward=(\S+)")

    @staticmethod
    def from_openspiel(observation: OpenSpielObservation, interface_transformer: InterfaceTransformer) -> BanditObservation:
        match = BanditObservation._PATTERN.fullmatch(observation.state)
        if not match:
            raise ValueError(f"Invalid bandit information state: {observation.state}")
        pressed, reward = match.groups()
        i_actions = [BanditAction.from_openspiel(action, interface_transformer) for action in observation.i_actions]
        return BanditObservation(observation, None if pressed == "-" else pressed, None if reward == "-" else int(reward), i_actions, interface_transformer)

    def __init__(
        self, os_observation: OpenSpielObservation, pressed: str | None, reward: int | None, i_actions: list[BanditAction], interface_transformer: InterfaceTransformer
    ) -> None:
        super().__init__(os_observation, i_actions, [], interface_transformer)
        self.pressed = pressed
        self.reward = reward


class BanditTransformer(InterfaceTransformer[BanditAction, BanditObservation, []]):
    def _inner_llm_action(self, action: BanditAction) -> str:
        return action.name.replace("button ", "button:")

    def display_action(self, action: BanditAction) -> str:
        return action.name

    def _inner_llm_state(self, observation: BanditObservation) -> str:
        return observation.os_observation.state

    def _inner_llm_partial_states(self, observation: BanditObservation) -> list[str]:
        if observation.pressed is None:
            return ["you have not pressed a button yet"]
        if observation.reward is None:
            return [f"you pressed the {observation.pressed} button"]
        # a Bernoulli button wins (+1) or loses (-1); a family with more rewards would name the reward instead
        return [f"you pressed the {observation.pressed} button, and it {'won' if observation.reward > 0 else 'lost'}"]

    def _inner_llm_positions(self, observation: BanditObservation) -> tuple[list[str], list[str]]:
        return [], []

    def _inner_display_state(self, observation: BanditObservation) -> str:
        return observation.os_observation.state

    def reset(self) -> None:
        pass

    def set_state(self) -> None:
        pass

    def get_other_params(self) -> dict[str, Any] | None:
        return None


BANDIT_HEAD_PROMPT = """
In front of you are {num_buttons} buttons: {buttons}.
{stationarity} {chance_rule}
You press one button, and the game ends: you win if the button wins, and you lose otherwise.
"""

_HIDDEN_CHANCE_RULE = "You are not told these probabilities."

# which rules the prompt hides: none, or the win probabilities
_RULES = ("stated", "hidden")

# whether the prompt says that the probabilities never change
_STATIONARITY = {"stated": "Each button wins with its own fixed probability, which does not change between games.", "omitted": "Each button wins with its own probability."}


def _chance_rule(chances: tuple[Fraction, ...]) -> str:
    # the probabilities, stated per button in the listed order; exact fractions
    return "The " + ", the ".join(f"{colour} button wins with probability {chance}" for colour, chance in zip(COLOURS, chances)) + "."


class BanditPromptAdapter(SinglePlayerPromptAdapter[[BanditGameParams]]):
    def __init__(self) -> None:
        super().__init__(head_prompt_template=BANDIT_HEAD_PROMPT, use_partial_state=True, position_name="")

    def action_format(self) -> str:
        return "<button:colour>, e.g., <button:red>"

    def restart_prompt(self, params: BanditGameParams) -> None:
        chance_rule = _HIDDEN_CHANCE_RULE if params.rules == "hidden" else _chance_rule(params.chances)
        buttons = ", ".join(COLOURS[: len(params.p)])
        self.head_prompt = self.head_prompt_template.format(num_buttons=len(params.p), buttons=buttons, stationarity=_STATIONARITY[params.stationarity], chance_rule=chance_rule)

    def observation_prompt(self, observation: LLMObservation) -> str:
        # one state, before the press: both observation types tell it, as in Monty Hall
        return self._partial_states_prompt(observation)


class BanditEngine(TurnBasedEngine[BanditTransformer, BanditPromptAdapter]):
    def __init__(self, game_config: GameConfig) -> None:
        params = cast(BanditGameParams, game_config.params)
        prompt_adapter = BanditPromptAdapter()
        prompt_adapter.restart_prompt(params)
        super().__init__(game_config, BanditGame(params), BanditTransformer(), prompt_adapter, BanditAction, BanditObservation)

    def reset(self) -> None:
        self.game.reset()
