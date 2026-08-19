from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.intervals import ConfidenceInterval, bootstrap_ci, sem_ci, t_ci, wilson_ci
from plybench.common.enums import CIFamily
from plybench.common.serializable import Serializable


class CIBundle(Serializable):
    """A metric's point value + sample size + whichever confidence intervals apply to its family
    (ratios carry a Wilson interval; means carry SEM / t / bootstrap). Absent intervals are omitted."""

    def __init__(
        self,
        value: float,
        n: int,
        wilson: ConfidenceInterval | None = None,
        sem: ConfidenceInterval | None = None,
        t: ConfidenceInterval | None = None,
        bootstrap: ConfidenceInterval | None = None,
    ) -> None:
        self.value = value
        self.n = n
        self.wilson = wilson
        self.sem = sem
        self.t = t
        self.bootstrap = bootstrap

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CIBundle:
        def ci(key: str) -> ConfidenceInterval | None:
            return ConfidenceInterval.from_dict(data[key]) if data.get(key) is not None else None

        return cls(data["value"], data["n"], wilson=ci("wilson"), sem=ci("sem"), t=ci("t"), bootstrap=ci("bootstrap"))

    def fmt(self, width: int = 8, interval: bool = False, count: bool = False) -> str:
        if self.n == 0:
            return f"{'NaN':>{width}s} (n=0)" if count else f"{'NaN':>{width}s}"

        ci = (self.wilson or self.t or self.sem or self.bootstrap) if interval else None
        bounds = f" [{ci.lower:.3f}, {ci.upper:.3f}]" if ci is not None else ""
        suffix = f" (n={self.n})" if count else ""
        return f"{self.value:{width}.4f}{bounds}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"value": self.value, "n": self.n}
        for key, interval in (("wilson", self.wilson), ("sem", self.sem), ("t", self.t), ("bootstrap", self.bootstrap)):
            if interval is not None:
                result[key] = interval.to_dict()
        return result


def ratio_bundle(distribution: Distribution, confidence: float = 0.95) -> CIBundle:
    return CIBundle(distribution.ratio, distribution.n, wilson=wilson_ci(distribution, confidence))


def rate(indicators: Sequence[bool], confidence: float = 0.95) -> CIBundle:
    return ratio_bundle(Distribution.from_values(indicators), confidence)


def mean_bundle(distribution: Distribution, confidence: float = 0.95) -> CIBundle:
    return CIBundle(
        distribution.mean,
        distribution.n,
        sem=sem_ci(distribution, confidence),
        t=t_ci(distribution, confidence),
        bootstrap=bootstrap_ci(distribution, confidence),
    )


def bundle_for_family(family: CIFamily, distribution: Distribution, confidence: float = 0.95) -> CIBundle:
    match family:
        case CIFamily.RATIO:
            return ratio_bundle(distribution, confidence)
        case CIFamily.MEAN:
            return mean_bundle(distribution, confidence)
        case _:
            raise ValueError(f"Unknown CI family: {family}")
