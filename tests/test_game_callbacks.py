"""Tests for GameCallbacks fan-out composition (combine) and per-player scoping (for_player)."""

from __future__ import annotations

from dataclasses import dataclass

from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.common.enums import GameResults
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import OpenSpielObservation, TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.games.tic_tac_toe.tic_tac_toe import TicTacToeObservation, TicTacToeTransformer
from plybench.player.player import Player, PlayerOutput
from plybench.trackers.game_tracker import GameStep, GameTracker


@dataclass(frozen=True)
class _P(PlayerParams):
    label: str

    @classmethod
    def from_string(cls, params_string: str) -> "_P":
        return cls(params_string)

    def to_string(self) -> str:
        return self.label

    @property
    def path_suffix(self) -> str:
        return self.label


CFG_A = PlayerConfig("p", _P("a"))
CFG_B = PlayerConfig("p", _P("b"))
TRACKER = GameTracker(1, CFG_A, CFG_B, {})
OBSERVATION = TicTacToeObservation.from_openspiel(OpenSpielObservation("...\n...\n...", [], [], 0), TicTacToeTransformer())
STEP = GameStep(1, "", "", "", "", "")
OUTPUT = PlayerOutput()


class _FakePlayer(Player):
    def __init__(self, player_config: PlayerConfig) -> None:
        super().__init__(player_config, "i")

    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        pass

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        raise AssertionError("Callback tests do not request a move")

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""


def _recording_bundle(log: list[str], tag: str) -> GameCallbacks:
    return GameCallbacks(
        game_start_callback=lambda tracker: log.append(f"{tag}:start"),
        before_move_callback=lambda player, obs, moves: log.append(f"{tag}:before:{player.player_config.params.to_string()}"),
        after_move_callback=lambda player, out, step: log.append(f"{tag}:after:{player.player_config.params.to_string()}"),
        game_end_callback=lambda tracker, results: log.append(f"{tag}:end"),
    )


def _fire(cb: GameCallbacks) -> None:
    cb.on_game_start(TRACKER)
    cb.on_before_move(_FakePlayer(CFG_A), OBSERVATION, [])
    cb.on_after_move(_FakePlayer(CFG_A), OUTPUT, STEP)
    cb.on_before_move(_FakePlayer(CFG_B), OBSERVATION, [])
    cb.on_after_move(_FakePlayer(CFG_B), OUTPUT, STEP)
    cb.on_game_end(TRACKER, (GameResults.WIN, GameResults.LOSS))


def test_combine_fans_out_to_all_bundles_in_order():
    log: list[str] = []
    combined = GameCallbacks.combine(
        _recording_bundle(log, "log"),
        _recording_bundle(log, "x"),
    )
    combined.on_game_start(TRACKER)
    # both bundles fire, in the order given
    assert log == ["log:start", "x:start"]


def test_combine_mixes_logging_and_scoped_player_bundles():
    logging_log: list[str] = []
    a_log: list[str] = []
    b_log: list[str] = []

    combined = GameCallbacks.combine(
        _recording_bundle(logging_log, "log"),  # non-player: sees everything
        GameCallbacks.for_player(CFG_A, _recording_bundle(a_log, "a")),  # only player A's moves
        GameCallbacks.for_player(CFG_B, _recording_bundle(b_log, "b")),  # only player B's moves
    )
    _fire(combined)

    # logging bundle sees every event
    assert logging_log == [
        "log:start",
        "log:before:a",
        "log:after:a",
        "log:before:b",
        "log:after:b",
        "log:end",
    ]
    # player A bundle: game-level hooks always + only A's move hooks
    assert a_log == ["a:start", "a:before:a", "a:after:a", "a:end"]
    # player B bundle: game-level hooks always + only B's move hooks
    assert b_log == ["b:start", "b:before:b", "b:after:b", "b:end"]


def test_empty_and_none_bundles_are_safe():
    combined = GameCallbacks.combine(None, GameCallbacks())  # no callbacks set
    _fire(combined)  # should not raise


def test_default_callbacks_are_noop():
    _fire(GameCallbacks())  # should not raise
