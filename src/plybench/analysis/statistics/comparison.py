from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scipy import stats

from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.intervals import ConfidenceInterval
from plybench.common.enums import CIFamily


@dataclass(frozen=True)
class GroupComparison:
    """Difference between two groups on one metric (group A minus group B), with a confidence interval
    on the difference and a two-sided significance test. `difference`/`interval`/`se` are None when a
    group is too small to compare; `p_value` is None when the test statistic is undefined (no variance).
    `se` is the unpooled standard error behind `interval` — the one to propagate when combining several
    independent differences, which is why it is carried rather than recomputed by callers."""

    difference: float | None
    interval: ConfidenceInterval | None
    se: float | None
    p_value: float | None
    significant: bool
    test: str
    n_a: int
    n_b: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "difference": self.difference,
            "interval": self.interval.to_dict() if self.interval is not None else None,
            "se": self.se,
            "p_value": self.p_value,
            "significant": self.significant,
            "test": self.test,
            "n_a": self.n_a,
            "n_b": self.n_b,
        }


def _insufficient(test: str, n_a: int, n_b: int) -> GroupComparison:
    return GroupComparison(None, None, None, None, False, test, n_a, n_b)


def two_proportion_test(a: Distribution, b: Distribution, confidence: float = 0.95) -> GroupComparison:
    """Two-sided two-proportion z-test on 0/1 indicator samples. The CI on the difference uses the
    unpooled standard error; the hypothesis test uses the pooled proportion (H0: p_a == p_b)."""
    n_a, n_b = a.n, b.n
    if n_a == 0 or n_b == 0:
        return _insufficient("two_proportion_z", n_a, n_b)

    p_a, p_b = a.ratio, b.ratio
    diff = p_a - p_b
    alpha = 1 - confidence

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    interval = ConfidenceInterval(diff, diff - z_crit * se, diff + z_crit * se)

    p_pool = (a.total + b.total) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se_pool == 0:
        return GroupComparison(diff, interval, se, None, False, "two_proportion_z", n_a, n_b)

    z = diff / se_pool
    p_value = 2.0 * float(stats.norm.sf(abs(z)))
    return GroupComparison(diff, interval, se, p_value, p_value < alpha, "two_proportion_z", n_a, n_b)


def mean_difference_test(a: Distribution, b: Distribution, confidence: float = 0.95) -> GroupComparison:
    """Two-sided Welch's t-test (unequal variances) on the group means, with a matching t interval on
    the difference using the Welch-Satterthwaite degrees of freedom."""
    n_a, n_b = a.n, b.n
    if n_a < 2 or n_b < 2:
        return _insufficient("welch_t", n_a, n_b)

    m_a, m_b = a.mean, b.mean
    diff = m_a - m_b
    alpha = 1 - confidence

    var_a, var_b = a.std(ddof=1) ** 2, b.std(ddof=1) ** 2
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return GroupComparison(diff, ConfidenceInterval(diff, diff, diff), se, None, False, "welch_t", n_a, n_b)

    df = se**4 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    interval = ConfidenceInterval(diff, diff - t_crit * se, diff + t_crit * se)

    t_stat = diff / se
    p_value = 2.0 * float(stats.t.sf(abs(t_stat), df))
    return GroupComparison(diff, interval, se, p_value, p_value < alpha, "welch_t", n_a, n_b)


def compare_for_family(family: CIFamily, a: Distribution, b: Distribution, confidence: float = 0.95) -> GroupComparison:
    """Pick the between-group test appropriate to a metric's family (ratio -> two-proportion z, mean ->
    Welch's t) so any metric can be compared across two groups from its family alone."""
    return two_proportion_test(a, b, confidence) if family == CIFamily.RATIO else mean_difference_test(a, b, confidence)
