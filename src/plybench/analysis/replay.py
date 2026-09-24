from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import pyspiel as sp

from plybench.analysis.stats.step_stats import StepStats
from plybench.common.enums import StateClass
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.core.engine import TurnBasedEngine
from plybench.core.interface import InterfaceAction
from plybench.core.minimax import AVQ
from plybench.player.simple.optimal_player import Judgeable
from plybench.registry import Registry
from plybench.trackers.game_tracker import GameStep, GameTracker


@dataclass
class ReplayedStep:
    step: GameStep
    state_class: StateClass
    is_optimal: bool
    regret: float
    n_legal: int  # branching factor at this state
    n_optimal: int  # size of the solver's optimal-action set at this state
    # the move sets as the model saw them, so downstream consumers (e.g. an LLM annotator that must be told what the right answers were) get the ground truth without paying for a second replay
    legal_moves: list[str] = field(default_factory=list)
    optimal_moves: list[str] = field(default_factory=list)

    def to_stats(self) -> StepStats:
        return StepStats(self.step.seq, self.step.input_tokens, self.step.output_tokens, self.state_class, self.is_optimal, self.regret)


class TurnBasedReplayer:
    def __init__(self, engine: TurnBasedEngine, solver: Judgeable) -> None:
        self._engine = engine
        self._solver = solver

    @property
    def reward_range(self) -> tuple[float, float]:
        loss_value, win_value = self._engine.game.get_reward_range()
        return float(loss_value), float(win_value)

    def _solved(self, serialized_state: str) -> tuple[int, list[InterfaceAction], AVQ]:
        game = self._engine.game
        game.deserialize_state(serialized_state)  # deserialize the state from stored string
        pid = game.get_player()  # get the player to move

        # get the observation and legal moves for the player to move and current state
        observation = self._engine.observation_class.from_openspiel(game.get_observation(pid), self._engine.interface_transformer)
        moves = [self._engine.action_class.from_openspiel(move, self._engine.interface_transformer) for move in game.get_legal_moves(pid)]

        # judge the position
        verdict = self._solver.optimal(pid, observation)
        if verdict is None:
            raise ValueError(f"No judge verdict for player {pid} at state {observation.os_observation.state}")

        return pid, moves, verdict

    @staticmethod
    def _selected(moves: list[InterfaceAction], played: str) -> InterfaceAction:
        selected = next((move for move in moves if move.to_llm().string == played), None)
        if selected is None:
            raise ValueError(f"Recorded move not found among legal moves: {played}")
        return selected

    def probe(self, serialized_state: str, played: str) -> tuple[sp.State, int, AVQ]:
        # get the pid, legal moves and verdict for the state
        _, moves, verdict = self._solved(serialized_state)
        # return the state of the game, the selected move and the verdict
        return self._engine.game.state, self._selected(moves, played).number, verdict

    def _iter_replayed_steps(self, game_tracker: GameTracker, player_config: PlayerConfig) -> Iterator[ReplayedStep]:
        loss_value, _ = self.reward_range

        for step in game_tracker.steps:
            # skip steps for other players
            if step.player_hash != player_config.hash:
                continue

            # get the pid, legal moves and verdict for the state
            _, moves, verdict = self._solved(step.serialized_state)

            # classify the state into DECISION, LOST or DONT_CARE
            state_class = verdict.classify_state([move.number for move in moves], loss_value)

            # get legal and optimal actions
            n_legal, n_optimal = len(moves), len(verdict.A())
            legal_llm = {move.number: move.to_llm().string for move in moves}
            optimal_llm = [string for number, string in legal_llm.items() if verdict.check_optimal(number)]

            # check if the move is failed
            if "FAIL" in step.move:
                # a failed (illegal / malformed) move: worst-case regret, definitely not optimal
                is_optimal = False
                regret = float(verdict.V() - loss_value)
            else:
                # extract the move that was selected by the llm player
                selected = self._selected(moves, step.move)
                # compute optimality of the selected move
                is_optimal = bool(verdict.check_optimal(selected.number))
                # compute regret of the selected move
                regret = float(verdict.V() - verdict.Q(selected.number))

            yield ReplayedStep(step, state_class, is_optimal, regret, n_legal, n_optimal, list(legal_llm.values()), optimal_llm)

    def replay_stats(self, game_tracker: GameTracker, player_config: PlayerConfig) -> list[StepStats]:
        return [replayed.to_stats() for replayed in self._iter_replayed_steps(game_tracker, player_config)]

    def replay_steps(self, game_tracker: GameTracker, player_config: PlayerConfig) -> list[ReplayedStep]:
        return list(self._iter_replayed_steps(game_tracker, player_config))


def build_replayer(registry: Registry, game_config: GameConfig) -> TurnBasedReplayer:
    engine = registry.build_engine(game_config)
    engine.reset()  # the constructor's game need not be the played instance; the solver must see the reset one
    solver = registry.build_player(engine.game, registry.player_config("optimal:"), "i")
    solver.initialize_policy(engine.game, engine.prompt_adapter)
    return TurnBasedReplayer(engine, cast(Judgeable, solver))


class ReplayerCache:
    def __init__(self, registry: Registry) -> None:
        self._registry = registry
        self._cache: dict[str, TurnBasedReplayer] = {}

    def __call__(self, game_config: GameConfig) -> TurnBasedReplayer:
        key = game_config.to_string()
        if key not in self._cache:
            self._cache[key] = build_replayer(self._registry, game_config)
        return self._cache[key]


class MemoizedReplay:
    def __init__(self, replayer: TurnBasedReplayer, player_config: PlayerConfig) -> None:
        self._replayer = replayer
        self._player = player_config
        self._cache: dict[int, list[StepStats]] = {}

    def __call__(self, game_tracker: GameTracker) -> list[StepStats]:
        key = id(game_tracker)
        if key not in self._cache:
            self._cache[key] = self._replayer.replay_stats(game_tracker, self._player)
        return self._cache[key]
