from __future__ import annotations

from collections.abc import Callable

from plybench.analysis.stats.compute import compute_matchup_stats
from plybench.analysis.stats.matchup_stats import MatchupStats
from plybench.analysis.stats.partition import Partitioner, PartitionStats, compute_partition_stats, compute_recognition_split
from plybench.common.progress import track
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.registry import Registry
from plybench.trackers.result_tracker import ResultTracker


class BenchmarkAnalysis:
    """In-memory statistical analysis over a finished benchmark: computes per-(player, opponent, game)
    matchup statistics (rates + score + moves + tokens, each with confidence intervals). When a
    `registry` is supplied, solvable games additionally get optimality/regret (via minimax replay).
    Nothing is persisted — the numbers are cheap to recompute from the recorded result files (the one
    expensive artefact, the solved minimax cache, is already cached on disk by the optimal judge)."""

    def __init__(self, results: BenchmarkResults, registry: Registry | None = None, confidence: float = 0.95, include_fails: bool = False) -> None:
        self.results = results
        self.registry = registry
        self.confidence = confidence
        self.include_fails = include_fails

    def matchup(self, game_config: GameConfig, player_config: PlayerConfig, opponent_config: PlayerConfig) -> MatchupStats:
        tracker = self.results.find(game_config, player_config, opponent_config)
        return compute_matchup_stats(tracker, self.registry, self.confidence, self.include_fails)

    def analyze(self, progress: bool | None = None) -> list[MatchupStats]:
        trackers = track(self.results.trackers, "Computing stats", len(self.results.trackers), progress)
        return [compute_matchup_stats(tracker, self.registry, self.confidence, self.include_fails) for tracker in trackers]

    def _split(self, label: str, compute: Callable[[ResultTracker, Registry], PartitionStats | None], progress: bool | None) -> list[PartitionStats]:
        """Run a per-matchup split over every tracker, dropping the matchups it does not apply to. All
        splits need a registry: they replay the recorded moves against the solved minimax cache."""
        if self.registry is None:
            return []
        trackers = track(self.results.trackers, f"Computing {label} split", len(self.results.trackers), progress)
        stats = [compute(tracker, self.registry) for tracker in trackers]
        return [stat for stat in stats if stat is not None]

    def analyze_partition(self, partitioner: Partitioner, progress: bool | None = None) -> list[PartitionStats]:
        """Per-matchup partition analysis: the analysed player's judged moves grouped by `partitioner`,
        summarised per group on the default move-metrics with a comparison to the baseline group."""
        return self._split(partitioner.name, lambda tracker, registry: compute_partition_stats(tracker, registry, partitioner, confidence=self.confidence), progress)

    def analyze_recognition(self, progress: bool | None = None) -> list[PartitionStats]:
        """Convenience for the recognition split: moves grouped by whether the reasoning trace recognised
        the underlying game. Only defined for recognisable, solvable games with reasoning traces."""
        return self._split("recognition", lambda tracker, registry: compute_recognition_split(tracker, registry, self.confidence), progress)
