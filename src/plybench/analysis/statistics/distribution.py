from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from plybench.common.serializable import Serializable


class Distribution(Serializable):
    """A flat sample of observations (one value per game or per move). The base container the extractors
    fill and the CI helpers consume; `ratio` is just `mean` read as a proportion of 0/1 indicators."""

    def __init__(self, items: Iterable[float] | None = None) -> None:
        self.items: list[float] = [float(value) for value in items] if items is not None else []

    @classmethod
    def from_values(cls, values: Iterable[float]) -> Distribution:
        return cls([float(value) for value in values])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Distribution:
        return cls([float(value) for value in data["items"]])

    def add(self, value: float) -> None:
        self.items.append(float(value))

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def total(self) -> float:
        return float(sum(self.items))

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def ratio(self) -> float:
        return self.mean

    def std(self, ddof: int = 1) -> float:
        if self.n <= ddof:
            return 0.0
        return float(np.std(self.items, ddof=ddof))

    def to_dict(self) -> dict[str, Any]:
        return {"items": list(self.items)}
