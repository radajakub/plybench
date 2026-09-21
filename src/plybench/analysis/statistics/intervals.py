from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from plybench.analysis.statistics.distribution import Distribution
from plybench.common.serializable import Serializable

_BOOTSTRAP_RESAMPLES = 9999
_BOOTSTRAP_SEED = 0


class ConfidenceInterval(Serializable):
    def __init__(self, value: float, lower: float, upper: float) -> None:
        self.value = value
        self.lower = lower
        self.upper = upper

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfidenceInterval:
        return cls(data["value"], data["lower"], data["upper"])

    def unwrap(self) -> tuple[float, float, float]:
        return (self.value, self.lower, self.upper)

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "lower": self.lower, "upper": self.upper}


def z_score(confidence: float) -> float:
    return float(stats.norm.ppf(1 - (1 - confidence) / 2))


def wilson_ci(distribution: Distribution, confidence: float = 0.95) -> ConfidenceInterval:
    """Wilson score interval for a proportion (the items are 0/1 indicators)."""
    n = distribution.n
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0)

    p = distribution.ratio
    z = z_score(confidence)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ConfidenceInterval(p, max(0.0, center - half), min(1.0, center + half))


def sem_ci(distribution: Distribution, confidence: float = 0.95) -> ConfidenceInterval:
    """Normal (z) interval around the mean using the standard error."""
    n = distribution.n
    mean = distribution.mean
    if n < 2:
        return ConfidenceInterval(mean, mean, mean)

    se = distribution.std(ddof=1) / math.sqrt(n)
    z = z_score(confidence)
    return ConfidenceInterval(mean, mean - z * se, mean + z * se)


def t_ci(distribution: Distribution, confidence: float = 0.95) -> ConfidenceInterval:
    """Student's t interval around the mean (finite-sample correction on the standard error)."""
    n = distribution.n
    mean = distribution.mean
    if n < 2:
        return ConfidenceInterval(mean, mean, mean)

    se = distribution.std(ddof=1) / math.sqrt(n)
    crit = float(stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1))
    return ConfidenceInterval(mean, mean - crit * se, mean + crit * se)


def bootstrap_ci(distribution: Distribution, confidence: float = 0.95) -> ConfidenceInterval:
    """BCa bootstrap interval around the mean. Degenerate (point) interval when the sample has no
    variance or too few observations for a resample to be meaningful."""
    n = distribution.n
    mean = distribution.mean
    if n < 2 or distribution.std(ddof=1) == 0.0:
        return ConfidenceInterval(mean, mean, mean)

    data = np.asarray(distribution.items, dtype=float)
    try:
        result = stats.bootstrap(
            (data,),
            np.mean,
            confidence_level=confidence,
            n_resamples=_BOOTSTRAP_RESAMPLES,
            method="BCa",
            rng=np.random.default_rng(_BOOTSTRAP_SEED),
        )
    except Exception:
        return ConfidenceInterval(mean, mean, mean)

    return ConfidenceInterval(mean, float(result.confidence_interval.low), float(result.confidence_interval.high))
