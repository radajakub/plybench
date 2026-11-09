from __future__ import annotations

import math
from typing import Any

from plybench.analysis.extractors.suite import default_suite
from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.intervals import ConfidenceInterval, t_ci, wilson_ci
from plybench.common.enums import CIFamily
from plybench.configs.player_config import PlayerConfig
from plybench.trackers.game_tracker import GameTracker

# A matchup's (or a training phase's) running summary, kept in its metadata.json next to `completed` and updated with every recorded game, so a tool
# reads how a run is doing without loading a game. Per default metric (outcomes, score, moves, tokens; the analysed player's view, fails not counted as
# losses) the moments of its values, n, sum and sum of squares, which add up over games and over trackers, and the value with its interval (Wilson for
# rates, t for means), as `--status` and the analysis compute them. The games stay the source: `--summaries check` compares, `--summaries write` mends.

CONFIDENCE = 0.95

Summary = dict[str, dict[str, Any]]


def _number(value: float) -> float | None:
    # JSON has no NaN or infinity
    return value if math.isfinite(value) else None


def _interval(family: CIFamily, n: int, total: float, squares: float) -> ConfidenceInterval:
    if family == CIFamily.RATIO:
        # a rate's values are 0/1 indicators: their sum is the count, which is all the Wilson interval needs
        return wilson_ci(Distribution([1.0] * round(total) + [0.0] * (n - round(total))), CONFIDENCE)
    # as Distribution.mean: 0 over no values
    mean = total / n if n else 0.0
    if n < 2:
        return ConfidenceInterval(mean, mean, mean)
    # the sample deviation from the moments, then the t interval of the analysis
    variance = max(0.0, (squares - n * mean * mean) / (n - 1))
    return t_ci(_Moments(n, mean, math.sqrt(variance)), CONFIDENCE)


class _Moments(Distribution):
    # a distribution known only by its size, mean and sample deviation: what t_ci reads
    def __init__(self, n: int, mean: float, deviation: float) -> None:
        super().__init__()
        self._n, self._mean, self._deviation = n, mean, deviation

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean

    def std(self, ddof: int = 0) -> float:
        return self._deviation


def _finish(summary: Summary) -> Summary:
    families = {extractor.name.value: extractor.family for extractor in default_suite()}
    for name, entry in summary.items():
        interval = _interval(families[name], entry["n"], entry["sum"], entry["sumsq"])
        entry |= {"value": _number(interval.value), "lower": _number(interval.lower), "upper": _number(interval.upper)}
    return summary


def summarize(games: list[GameTracker], player: PlayerConfig) -> Summary:
    summary: Summary = {}
    for extractor in default_suite():
        items = extractor.extract(games, player).items
        summary[extractor.name.value] = {"n": len(items), "sum": float(sum(items)), "sumsq": float(sum(item * item for item in items))}
    return _finish(summary)


def add_game(summary: Summary | None, game: GameTracker, player: PlayerConfig) -> Summary:
    # the summary with one more game; None (no game yet) starts one
    added = summarize([game], player)
    if summary is None:
        return added
    merged = {name: {key: summary[name][key] + entry[key] for key in ("n", "sum", "sumsq")} for name, entry in added.items()}
    return _finish(merged)


def matches(stored: Summary | None, computed: Summary) -> bool:
    # the same counts (floats compared closely: token sums are large)
    if stored is None or set(stored) != set(computed):
        return False
    return all(
        stored[name]["n"] == entry["n"] and all(math.isclose(stored[name][key], entry[key], rel_tol=1e-9, abs_tol=1e-9) for key in ("sum", "sumsq"))
        for name, entry in computed.items()
    )
