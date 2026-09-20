from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from plybench.callbacks.matchup_callbacks import MatchupCallbacks
from plybench.harness.benchmark.results import BenchmarkResults

# the benchmark's own boundaries, on top of the matchup and round hooks every harness shares
BenchmarkStartCallback = Callable[[list[str], list[str], list[str]], None]
BenchmarkEndCallback = Callable[["BenchmarkResults"], None]


@dataclass
class BenchmarkCallbacks(MatchupCallbacks):
    benchmark_start_callback: BenchmarkStartCallback | None = None
    benchmark_end_callback: BenchmarkEndCallback | None = None

    def on_benchmark_start(self, game_configs: list[str], player_configs: list[str], opponent_configs: list[str]) -> None:
        if self.benchmark_start_callback is not None:
            self.benchmark_start_callback(game_configs, player_configs, opponent_configs)

    def on_benchmark_end(self, results: BenchmarkResults) -> None:
        if self.benchmark_end_callback is not None:
            self.benchmark_end_callback(results)
