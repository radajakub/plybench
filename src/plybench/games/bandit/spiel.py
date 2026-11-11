from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pyspiel as sp

# A stochastic multi-armed bandit as a one-player OpenSpiel game with chance: K buttons, each with its own fixed reward distribution. A game is one
# press:
#   1. the player presses a button;
#   2. chance draws the reward from the button's distribution; the game ends, and the reward is the player's return.
# One game is one round of the bandit; a run of the bandit is a sequence of games, and what the player learns between them is the player's own state.
# The player sees which button it pressed and, once the game is over, the reward.
#
# Every distribution is finite: a list of (reward, probability), so the solver's Q*(a) is the button's exact mean reward. Bernoulli buttons are the
# only family now: the reward is +1 (a win) with the button's probability p, else -1 (a loss), so the mean is 2p - 1. Another family (e.g. a normal
# distribution discretised on an integer grid) is a new way to build the distributions; the rest of the game reads only the (reward, probability) lists.
#
# Params:
# - p: the win probabilities, separated by `;`, as fractions or decimals (`3/4;1/4`, `0.75;0.25`); 2 to 8 buttons, each in [0, 1].
# - shift: which button gets which distribution. The buttons are always listed in the same order (red, blue, green, ...), and the distributions are
#   shifted cyclically over them: button j gets the distribution given at (j - shift) mod K; shift 0 gives button j the j-th one. So the best button
#   moves between runs with a different shift, while the list of buttons, and with it the prompt, stays the same; shifts 0 to K - 1 put each
#   distribution on each button once, so replicates over consecutive shifts balance any preference for a colour or a position.
#
# The state holds only ints and None: OpenSpiel serializes a Python state as its history and a pickle of its attributes. The distributions are read
# from the game, never kept in the state; a chance outcome is the index of its reward in the game's sorted list of rewards.

GAME_NAME = "bandit"
_DEFAULT_PARAMS = {"p": "3/4;1/4;1/4;1/4;1/4", "shift": 0}

# the buttons' names, in the order they are listed; their number bounds the number of buttons
COLOURS = ("red", "blue", "green", "yellow", "purple", "orange", "white", "black")

# the rewards of a Bernoulli button
LOSS, WIN = -1, 1

# a button's reward distribution: (reward, probability) with positive probabilities summing to 1, rewards ascending
type Distribution = tuple[tuple[int, Fraction], ...]

_GAME_TYPE = sp.GameType(
    short_name=GAME_NAME,
    long_name="Bandit",
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


def parse_p(value: str) -> tuple[Fraction, ...]:
    # the win probabilities in the order given, as exact fractions
    p = tuple(Fraction(part) for part in value.split(";"))
    if not 2 <= len(p) <= len(COLOURS):
        raise ValueError(f"A bandit needs 2 to {len(COLOURS)} buttons, not {len(p)}: {value}")
    if any(not 0 <= chance <= 1 for chance in p):
        raise ValueError(f"The win probabilities must be in [0, 1]: {value}")
    return p


def check_shift(shift: int) -> None:
    if shift < 0:
        raise ValueError(f"The shift must be a non-negative integer, not {shift}")


def bernoulli(p: Fraction) -> Distribution:
    # +1 with p, -1 otherwise; an outcome that cannot happen is left out
    return tuple((reward, chance) for reward, chance in ((LOSS, 1 - p), (WIN, p)) if chance > 0)


def shifted[T](values: tuple[T, ...], shift: int) -> tuple[T, ...]:
    # per button, in the order the buttons are listed: button j gets values[(j - shift) mod K]
    return tuple(values[(button - shift) % len(values)] for button in range(len(values)))


def mean(distribution: Distribution) -> Fraction:
    return sum((reward * chance for reward, chance in distribution), Fraction(0))


class BanditSpielGame(sp.Game):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        params = {**_DEFAULT_PARAMS, **(params or {})}

        shift = int(params["shift"])
        check_shift(shift)
        self.distributions: tuple[Distribution, ...] = shifted(tuple(bernoulli(p) for p in parse_p(str(params["p"]))), shift)
        self.num_buttons = len(self.distributions)
        # every reward any button can give, ascending; a chance outcome is an index into it
        self.rewards = tuple(sorted({reward for distribution in self.distributions for reward, _ in distribution}))

        info = sp.GameInfo(
            num_distinct_actions=self.num_buttons,
            max_chance_outcomes=len(self.rewards),
            num_players=1,
            min_utility=float(min(self.rewards)),
            max_utility=float(max(self.rewards)),
            utility_sum=None,
            max_game_length=2,
        )

        super().__init__(_GAME_TYPE, info, params)

    def new_initial_state(self) -> BanditSpielState:
        return BanditSpielState(self)

    def make_py_observer(self, iig_obs_type: Any = None, params: Any = None) -> BanditObserver:
        return BanditObserver()


class BanditSpielState(sp.State):
    def __init__(self, game: BanditSpielGame) -> None:
        super().__init__(game)
        self.num_buttons = game.num_buttons
        self.pressed: int | None = None
        # the index of the reward in the game's rewards, once drawn
        self.outcome: int | None = None

    def _game(self) -> BanditSpielGame:
        return cast(BanditSpielGame, self.get_game())

    def current_player(self) -> int:
        if self.outcome is not None:
            return sp.PlayerId.TERMINAL
        if self.pressed is not None:
            return sp.PlayerId.CHANCE
        return 0

    def _legal_actions(self, player: int) -> list[int]:
        return list(range(self.num_buttons))

    def chance_outcomes(self) -> list[tuple[int, float]]:
        # the distributions are read from the game, not kept in the state: the state's attributes are pickled into every recorded state
        assert self.pressed is not None
        game = self._game()
        return [(game.rewards.index(reward), float(chance)) for reward, chance in game.distributions[self.pressed]]

    def _apply_action(self, action: int) -> None:
        if self.pressed is None:
            self.pressed = action
        else:
            self.outcome = action

    def _action_to_string(self, player: int, action: int) -> str:
        if player == sp.PlayerId.CHANCE:
            return f"reward {self._game().rewards[action]:+d}"
        return f"button {COLOURS[action]}"

    def is_terminal(self) -> bool:
        return self.outcome is not None

    def returns(self) -> list[float]:
        if self.outcome is None:
            return [0.0]
        return [float(self._game().rewards[self.outcome])]

    def information_state(self) -> str:
        # what the player knows: the button it pressed (`-` before) and, once the game is over, the reward
        pressed = "-" if self.pressed is None else COLOURS[self.pressed]
        reward = "-" if self.outcome is None else f"{self._game().rewards[self.outcome]:+d}"
        return f"pressed={pressed} reward={reward}"

    def __str__(self) -> str:
        return f"pressed={self.pressed} outcome={self.outcome}"


class BanditObserver:
    # the information state, which has perfect recall; the observation is the same string
    def __init__(self) -> None:
        self.tensor = np.zeros(0, np.float32)
        self.dict: dict[str, Any] = {}

    def set_from(self, state: BanditSpielState, player: int) -> None:
        pass

    def string_from(self, state: BanditSpielState, player: int) -> str:
        return state.information_state()


sp.register_game(_GAME_TYPE, BanditSpielGame)
