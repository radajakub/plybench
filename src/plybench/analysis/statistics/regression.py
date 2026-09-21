from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from scipy import stats


@dataclass(frozen=True)
class LinearFit:
    """An OLS fit of y on a single x: `slope` units of y per unit of x, its standard error and two-sided
    p-value, the correlation `r`, and the sample size. Every field but `n` is None when the sample is too
    small or x has no spread, in which case the slope is not identified."""

    slope: float | None
    intercept: float | None
    stderr: float | None
    r: float | None
    p_value: float | None
    n: int

    @property
    def defined(self) -> bool:
        return self.slope is not None and self.stderr is not None

    def to_dict(self) -> dict[str, Any]:
        return {"slope": self.slope, "intercept": self.intercept, "stderr": self.stderr, "r": self.r, "p_value": self.p_value, "n": self.n}


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> LinearFit:
    if len(xs) != len(ys):
        raise ValueError(f"x and y must be paired observations, got {len(xs)} and {len(ys)}")
    if len(xs) < 3 or len(set(xs)) < 2:
        return LinearFit(None, None, None, None, None, len(xs))

    result = stats.linregress(xs, ys)
    # SciPy builds LinregressResult dynamically; Pyright cannot see its named fields.
    return LinearFit(float(result.slope), float(result.intercept), float(result.stderr), float(result.rvalue), float(result.pvalue), len(xs))  # pyright: ignore[reportAttributeAccessIssue]


@dataclass(frozen=True)
class FitDifference:
    """Difference between two independently estimated slopes (fit A minus fit B), with a two-sided z-test
    treating the regressions as independent (SE = sqrt(se_a^2 + se_b^2)). `delta_intercept` is reported
    alongside but is usually not the quantity of interest: a constant per-observation offset moves the
    intercept while leaving the slope untouched."""

    slope_a: float | None
    slope_b: float | None
    delta_slope: float | None
    se: float | None
    p_value: float | None
    significant: bool
    delta_intercept: float | None
    n_a: int
    n_b: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "slope_a": self.slope_a,
            "slope_b": self.slope_b,
            "delta_slope": self.delta_slope,
            "se": self.se,
            "p_value": self.p_value,
            "significant": self.significant,
            "delta_intercept": self.delta_intercept,
            "n_a": self.n_a,
            "n_b": self.n_b,
        }


def fit_difference(a: LinearFit, b: LinearFit, confidence: float = 0.95) -> FitDifference:
    if a.slope is None or b.slope is None or a.stderr is None or b.stderr is None:
        return FitDifference(a.slope, b.slope, None, None, None, False, None, a.n, b.n)

    delta = a.slope - b.slope
    se = math.sqrt(a.stderr**2 + b.stderr**2)
    delta_intercept = a.intercept - b.intercept if a.intercept is not None and b.intercept is not None else None
    if se == 0:
        return FitDifference(a.slope, b.slope, delta, se, None, False, delta_intercept, a.n, b.n)

    p_value = 2.0 * float(stats.norm.sf(abs(delta / se)))
    return FitDifference(a.slope, b.slope, delta, se, p_value, p_value < (1 - confidence), delta_intercept, a.n, b.n)
