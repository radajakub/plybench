from __future__ import annotations

from dataclasses import dataclass

from plybench.analysis.pooling import MetricOptions, MetricPool
from plybench.analysis.statistics.bundle import CIBundle
from plybench.analysis.visual.bench.encoder import ColorBy, StyleEncoder, StyleKey
from plybench.analysis.visual.bench.labels import Overrides, effort_key, model_key, model_strength, player_labels, provider_key
from plybench.common.enums import MetricName
from plybench.common.progress import track
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.registry import Registry
from plybench.trackers.result_tracker import ResultTracker


@dataclass(frozen=True)
class SeriesPoint:
    game: GameConfig
    bundle: CIBundle | None


@dataclass(frozen=True)
class Series:
    player: PlayerConfig
    label: str
    style: StyleKey
    points: list[SeriesPoint]

    @property
    def has_data(self) -> bool:
        return any(point.bundle is not None for point in self.points)


def style_key(player: PlayerConfig, color_by: ColorBy) -> StyleKey:
    color = provider_key(player) if color_by is ColorBy.PROVIDER else player.to_string()
    return StyleKey(color, model_key(player), effort_key(player), model_strength(player))


def encoder_keys(players: list[PlayerConfig], color_by: ColorBy) -> list[StyleKey]:
    return [style_key(player, color_by) for player in players]


def indistinguishable(series: list[Series], encoder: StyleEncoder, ignore_markers: bool = False) -> list[list[str]]:
    groups: dict[tuple[str, ...], list[str]] = {}
    for item in series:
        style = encoder.style(item.style)
        key = (style.color, str(style.linestyle)) if ignore_markers else (style.color, str(style.linestyle), style.marker)
        groups.setdefault(key, []).append(item.label)
    return [labels for labels in groups.values() if len(labels) > 1]


@dataclass(frozen=True)
class TrackerIndex:
    by_matchup: dict[tuple[str, str], list[ResultTracker]]

    @classmethod
    def of(cls, results: BenchmarkResults) -> TrackerIndex:
        grouped: dict[tuple[str, str], list[ResultTracker]] = {}
        for tracker in results.trackers:
            grouped.setdefault((tracker.game.to_string(), tracker.i.hash), []).append(tracker)
        return cls(grouped)

    def matchups(self, game: GameConfig, player: PlayerConfig, opponents: list[PlayerConfig]) -> list[ResultTracker]:
        wanted = {opponent.hash for opponent in opponents}
        return [tracker for tracker in self.by_matchup.get((game.to_string(), player.hash), []) if tracker.o.hash in wanted]


@dataclass(frozen=True)
class SeriesRequest:
    metric: MetricName
    opponents: list[PlayerConfig]
    title: str = ""


def build_series_batch(
    results: BenchmarkResults,
    registry: Registry | None,
    requests: list[SeriesRequest],
    games: list[GameConfig],
    players: list[PlayerConfig],
    options: MetricOptions | None = None,
    color_by: ColorBy = ColorBy.PROVIDER,
    label_overrides: Overrides | None = None,
    progress: bool | None = None,
) -> list[list[Series]]:
    index = TrackerIndex.of(results)
    pool = MetricPool(registry, options)
    labelled = list(zip(players, player_labels(players, label_overrides), strict=True))

    work = [(position, player, label) for position in range(len(requests)) for player, label in labelled]
    grouped: list[list[Series]] = [[] for _ in requests]
    for position, player, label in track(work, "Pooling metrics", len(work), progress):
        request = requests[position]
        points = [SeriesPoint(game, pool.bundle(index.matchups(game, player, request.opponents), request.metric)) for game in games]
        grouped[position].append(Series(player, label, style_key(player, color_by), points))
    return grouped


def build_series(
    results: BenchmarkResults,
    registry: Registry | None,
    metric: MetricName,
    games: list[GameConfig],
    players: list[PlayerConfig],
    opponents: list[PlayerConfig],
    options: MetricOptions | None = None,
    color_by: ColorBy = ColorBy.PROVIDER,
    label_overrides: Overrides | None = None,
    progress: bool | None = None,
) -> list[Series]:
    return build_series_batch(results, registry, [SeriesRequest(metric, opponents)], games, players, options, color_by, label_overrides, progress)[0]
