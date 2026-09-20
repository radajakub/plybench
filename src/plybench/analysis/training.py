from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plybench.analysis.statistics.bundle import CIBundle
from plybench.analysis.stats.compute import compute_matchup_metrics
from plybench.analysis.stats.matchup_stats import MatchupMetrics, Split
from plybench.common.enums import MetricName
from plybench.common.progress import track
from plybench.configs.player_config import PlayerConfig
from plybench.configs.training_run import TrainingRun
from plybench.harness.training.results import TrainingResults
from plybench.registry import Registry


@dataclass(frozen=True)
class EpochPoint:
    epoch: int
    metrics: Split[MatchupMetrics]

    def bundle(self, metric: MetricName) -> CIBundle:
        return self.metrics.combined.metrics[metric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "metrics": {
                "combined": self.metrics.combined.to_dict(),
                "i_first": self.metrics.i_first.to_dict(),
                "i_second": self.metrics.i_second.to_dict(),
            },
        }


@dataclass(frozen=True)
class LearningCurve:
    run: TrainingRun
    tester: PlayerConfig
    points: list[EpochPoint]

    def series(self, metric: MetricName) -> list[CIBundle]:
        return [point.bundle(metric) for point in self.points]

    def values(self, metric: MetricName) -> list[float]:
        return [bundle.value for bundle in self.series(metric)]

    def delta(self, metric: MetricName) -> float:
        # what the training actually bought, baseline to final epoch
        values = self.values(metric)
        return values[-1] - values[0] if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.run.to_dict(),
            "tester_config": self.tester.to_string(),
            "points": [point.to_dict() for point in self.points],
        }


class TrainingAnalysis:
    def __init__(self, results: TrainingResults, registry: Registry | None = None, confidence: float = 0.95, include_fails: bool = False) -> None:
        self.results = results
        self.registry = registry
        self.confidence = confidence
        self.include_fails = include_fails

    def curve(self, run: TrainingRun, tester: PlayerConfig) -> LearningCurve:
        points = [
            EpochPoint(index, compute_matchup_metrics(tracker, self.registry, self.confidence, self.include_fails)) for index, tracker in enumerate(self.results.curve(run, tester))
        ]
        return LearningCurve(run, tester, points)

    def curves(self, testers: list[PlayerConfig], progress: bool | None = None) -> list[LearningCurve]:
        pairs = [(run, tester) for run in self.results.runs for tester in testers]
        return [self.curve(run, tester) for run, tester in track(pairs, "Computing curves", len(pairs), progress)]
