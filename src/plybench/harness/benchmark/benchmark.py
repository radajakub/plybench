from __future__ import annotations

import asyncio
import json

from plybench.app import PlyBench
from plybench.callbacks.benchmark_callbacks import BenchmarkCallbacks
from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.common.paths import BenchmarkPathBuilder
from plybench.common.progress import track
from plybench.configs.benchmark_config import BenchmarkConfig
from plybench.configs.matchup import Matchup
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.harness.matchup import run_matchup_concurrent
from plybench.trackers.result_tracker import ResultTracker


class Benchmark:
    @classmethod
    def load_experiment(
        cls, op: PlyBench, filename: str, game_override: list[str] | None = None, player_override: list[str] | None = None, opponent_override: list[str] | None = None
    ) -> Benchmark:
        path_builder = BenchmarkPathBuilder()
        path = path_builder.experiment_path(filename)

        with open(path) as f:
            config = BenchmarkConfig.from_dict(json.load(f))

        return cls(
            path.stem,
            op,
            game_override if game_override is not None else config.get_game_configs(),
            player_override if player_override is not None else config.get_player_configs(),
            opponent_override if opponent_override is not None else config.get_opponent_configs(),
            config.num_games,
            path_builder,
        )

    def __init__(
        self,
        experiment: str,
        op: PlyBench,
        game_configs: list[str],
        player_configs: list[str],
        opponent_configs: list[str],
        num_games: int,
        path_builder: BenchmarkPathBuilder | None = None,
    ) -> None:
        self.experiment = experiment
        self.op = op
        self.game_configs = game_configs
        self.player_configs = player_configs
        self.opponent_configs = opponent_configs
        self.num_games = num_games
        self.path_builder = path_builder if path_builder is not None else BenchmarkPathBuilder()

    def _matrix(self) -> list[tuple[str, str, str]]:
        return [(game, player, opponent) for player in self.player_configs for opponent in self.opponent_configs for game in self.game_configs]

    async def _run_sync(self, concurrency: int | None = None, game_callbacks: GameCallbacks | None = None, benchmark_callbacks: BenchmarkCallbacks | None = None):
        for game, player, opponent in self._matrix():
            await self.single_matchup(game, player, opponent, concurrency, game_callbacks, benchmark_callbacks)

    async def _run_async(self, concurrency: int | None = None, game_callbacks: GameCallbacks | None = None, benchmark_callbacks: BenchmarkCallbacks | None = None):
        await asyncio.gather(
            *(asyncio.create_task(self.single_matchup(game, player, opponent, concurrency, game_callbacks, benchmark_callbacks)) for game, player, opponent in self._matrix())
        )

    async def run(
        self, sync: bool = False, concurrency: int | None = None, game_callbacks: GameCallbacks | None = None, benchmark_callbacks: BenchmarkCallbacks | None = None
    ) -> BenchmarkResults:
        benchmark_callbacks = benchmark_callbacks if benchmark_callbacks is not None else BenchmarkCallbacks()
        benchmark_callbacks.on_benchmark_start(self.game_configs, self.player_configs, self.opponent_configs)

        if sync:
            await self._run_sync(concurrency, game_callbacks, benchmark_callbacks)
        else:
            await self._run_async(concurrency, game_callbacks, benchmark_callbacks)

        results = self.get_results()
        benchmark_callbacks.on_benchmark_end(results)
        return results

    async def single_matchup(
        self,
        game_config_str: str,
        player_config_str: str,
        opponent_config_str: str,
        concurrency: int | None = None,
        game_callbacks: GameCallbacks | None = None,
        benchmark_callbacks: BenchmarkCallbacks | None = None,
    ) -> str:
        matchup = Matchup(
            self.op.registry.game_config(game_config_str),
            self.op.registry.player_config(player_config_str),
            self.op.registry.player_config(opponent_config_str),
            self.num_games,
        )

        tracker = await run_matchup_concurrent(
            self.op,
            matchup,
            game_callbacks=game_callbacks,
            matchup_callbacks=benchmark_callbacks,
            path_builder=self.path_builder,
            experiment=self.experiment,
            max_concurrent=concurrency,
        )
        return str(tracker.base_path)

    def get_results(self, progress: bool | None = None) -> BenchmarkResults:
        game_configs = [self.op.registry.game_config(game) for game in self.game_configs]
        player_configs = [self.op.registry.player_config(player) for player in self.player_configs]
        opponent_configs = [self.op.registry.player_config(opponent) for opponent in self.opponent_configs]

        matrix = [(player, opponent, game) for player in player_configs for opponent in opponent_configs for game in game_configs]
        trackers: list[ResultTracker] = []
        for player_config, opponent_config, game_config in track(matrix, "Loading results", len(matrix), progress):
            tracker = ResultTracker.new(
                self.experiment,
                player_config,
                opponent_config,
                game_config,
                self.num_games,
                self.op.registry,
                path_builder=self.path_builder,
            )
            tracker.load_if_exists()
            trackers.append(tracker)

        return BenchmarkResults(game_configs, player_configs, opponent_configs, trackers)
