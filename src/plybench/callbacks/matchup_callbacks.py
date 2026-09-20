from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Self

from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.player.player import Player, PlayerOutput
from plybench.trackers.game_tracker import GameStep
from plybench.trackers.result_tracker import ResultTracker

# Orchestration-layer hooks for one matchup, the counterpart to the game-loop `GameCallbacks`. A caller
# (or an agent building a more complex evaluation flow) supplies these to observe/act at matchup and
# round boundaries; they never drive gameplay. All slots are optional and null-guarded. Each harness
# extends or produces a bundle of its own: `BenchmarkCallbacks` adds the benchmark's own boundaries,
# `TrainingCallbacks.for_phase` builds one per epoch phase.
MatchupStartCallback = Callable[["ResultTracker", "GameConfig", "PlayerConfig", "PlayerConfig"], None]
MatchupEndCallback = Callable[["ResultTracker"], None]
RoundStartCallback = Callable[["GameConfig", "PlayerConfig", "PlayerConfig", int], None]
RoundCompleteCallback = Callable[["GameConfig", "PlayerConfig", "PlayerConfig", int], None]
# fired for every move of every round; the game-loop `after_move` hook re-tagged with the matchup it belongs to
MoveCompleteCallback = Callable[["GameConfig", "PlayerConfig", "PlayerConfig", int, "GameStep"], None]


@dataclass
class MatchupCallbacks:
    matchup_start_callback: MatchupStartCallback | None = None
    matchup_end_callback: MatchupEndCallback | None = None
    round_start_callback: RoundStartCallback | None = None
    round_complete_callback: RoundCompleteCallback | None = None
    move_complete_callback: MoveCompleteCallback | None = None

    def on_matchup_start(self, result_tracker: ResultTracker, game_config: GameConfig, i: PlayerConfig, o: PlayerConfig) -> None:
        if self.matchup_start_callback is not None:
            self.matchup_start_callback(result_tracker, game_config, i, o)

    def on_matchup_end(self, result_tracker: ResultTracker) -> None:
        if self.matchup_end_callback is not None:
            self.matchup_end_callback(result_tracker)

    def on_round_start(self, game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, game_round: int) -> None:
        if self.round_start_callback is not None:
            self.round_start_callback(game_config, i, o, game_round)

    def on_round_complete(self, game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, game_round: int) -> None:
        if self.round_complete_callback is not None:
            self.round_complete_callback(game_config, i, o, game_round)

    def on_move_complete(self, game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, game_round: int, step: GameStep) -> None:
        if self.move_complete_callback is not None:
            self.move_complete_callback(game_config, i, o, game_round, step)

    def for_round(self, game_config: GameConfig, i: PlayerConfig, o: PlayerConfig, game_round: int, bundle: GameCallbacks | None = None) -> GameCallbacks:
        # bridge to the game loop, whose callbacks only know about players: re-tag its moves with the
        # matchup/round they belong to, on top of whatever game callbacks the caller supplied
        def on_after_move(player: Player, player_output: PlayerOutput, step: GameStep) -> None:
            self.on_move_complete(game_config, i, o, game_round, step)

        return GameCallbacks.combine(bundle, GameCallbacks(after_move_callback=on_after_move))

    @classmethod
    def combine(cls, *bundles: Self | None) -> Self:
        children = tuple(bundle for bundle in bundles if bundle is not None)

        def fan(method_name: str) -> Callable[..., None]:
            def call(*args: object) -> None:
                for child in children:
                    getattr(child, method_name)(*args)

            return call

        # every slot `x_callback` is dispatched by the matching `on_x`, so a subclass's extra hooks fan out too
        return cls(**{field.name: fan(f"on_{field.name.removesuffix('_callback')}") for field in fields(cls)})
