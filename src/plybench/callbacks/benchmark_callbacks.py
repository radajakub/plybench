from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.player.player import Player, PlayerOutput
from plybench.trackers.game_tracker import GameStep
from plybench.trackers.result_tracker import ResultTracker

# Orchestration-layer phase hooks, the counterpart to the game-loop `GameCallbacks`. A caller (or an
# agent building a more complex evaluation flow) supplies these to observe/act at benchmark, matchup
# and round boundaries; they never drive gameplay. All slots are optional and null-guarded.
BenchmarkStartCallback = Callable[[list[str], list[str], list[str]], None]
BenchmarkEndCallback = Callable[["BenchmarkResults"], None]
MatchupStartCallback = Callable[["ResultTracker", "GameConfig", "PlayerConfig", "PlayerConfig"], None]
MatchupEndCallback = Callable[["ResultTracker"], None]
RoundStartCallback = Callable[["GameConfig", "PlayerConfig", "PlayerConfig", int], None]
RoundCompleteCallback = Callable[["GameConfig", "PlayerConfig", "PlayerConfig", int], None]
# fired for every move of every round; the game-loop `after_move` hook re-tagged with the matchup it belongs to
MoveCompleteCallback = Callable[["GameConfig", "PlayerConfig", "PlayerConfig", int, "GameStep"], None]


@dataclass
class BenchmarkCallbacks:
    benchmark_start_callback: BenchmarkStartCallback | None = None
    benchmark_end_callback: BenchmarkEndCallback | None = None
    matchup_start_callback: MatchupStartCallback | None = None
    matchup_end_callback: MatchupEndCallback | None = None
    round_start_callback: RoundStartCallback | None = None
    round_complete_callback: RoundCompleteCallback | None = None
    move_complete_callback: MoveCompleteCallback | None = None

    def on_benchmark_start(self, game_configs: list[str], player_configs: list[str], opponent_configs: list[str]) -> None:
        if self.benchmark_start_callback is not None:
            self.benchmark_start_callback(game_configs, player_configs, opponent_configs)

    def on_benchmark_end(self, results: BenchmarkResults) -> None:
        if self.benchmark_end_callback is not None:
            self.benchmark_end_callback(results)

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
    def combine(cls, *bundles: BenchmarkCallbacks | None) -> BenchmarkCallbacks:
        children = tuple(bundle for bundle in bundles if bundle is not None)

        def fan(method_name: str) -> Callable[..., None]:
            def call(*args: object) -> None:
                for child in children:
                    getattr(child, method_name)(*args)

            return call

        return cls(
            benchmark_start_callback=fan("on_benchmark_start"),
            benchmark_end_callback=fan("on_benchmark_end"),
            matchup_start_callback=fan("on_matchup_start"),
            matchup_end_callback=fan("on_matchup_end"),
            round_start_callback=fan("on_round_start"),
            round_complete_callback=fan("on_round_complete"),
            move_complete_callback=fan("on_move_complete"),
        )
