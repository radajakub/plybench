"""How much of the failure space the codebook has seen, from induction's own frequency spectrum.

The saturation curve says the run stopped finding codes. It cannot say whether that is because the
taxonomy is complete or because the batches went quiet. These two estimators answer a different question
-- how much is missing -- from the shape of the counts alone, at no extra LLM call.

Both are diagnostics, not proofs, and both are biased here in a known direction: induction reads a
mistake-enriched sample, so they describe that enriched population rather than the corpus, and batch
coding breaks the independence they assume. Report them beside the uncovered rate on fresh moves, which
is the figure that actually measures completeness, and never instead of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Coverage:
    """The frequency spectrum of coded instances: how many codes were seen exactly once, exactly twice,
    and how many instances there were in total."""

    n_codes: int
    n_instances: int
    f1: int  # codes seen exactly once -- the singletons that carry the whole estimate
    f2: int  # codes seen exactly twice

    @classmethod
    def of(cls, per_code: Mapping[str, int]) -> Coverage:
        counts = [count for count in per_code.values() if count > 0]
        return cls(len(counts), sum(counts), counts.count(1), counts.count(2))

    @property
    def unseen_mass(self) -> float | None:
        """Good-Turing: the share of error instances belonging to codes the sample never produced. This is
        mass, not types, and mass is the right target -- a code occurring once in 177 663 moves being
        absent changes no reported rate, while a missing common one changes every one of them."""
        return self.f1 / self.n_instances if self.n_instances else None

    @property
    def chao1(self) -> float | None:
        """A lower bound on the number of code types that exist, seen and unseen. The bias-corrected form
        is used when no code was seen exactly twice, where the classic estimator divides by zero."""
        if not self.n_codes:
            return None
        if self.f2:
            return self.n_codes + self.f1**2 / (2 * self.f2)
        return self.n_codes + self.f1 * (self.f1 - 1) / 2

    @property
    def missing_codes(self) -> float | None:
        """Chao1 minus what we have: how many types the spectrum says are still out there."""
        return None if self.chao1 is None else self.chao1 - self.n_codes

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_codes": self.n_codes,
            "n_instances": self.n_instances,
            "f1": self.f1,
            "f2": self.f2,
            "unseen_mass": self.unseen_mass,
            "chao1": self.chao1,
            "missing_codes": self.missing_codes,
        }

    def summary(self) -> str:
        if not self.n_instances:
            return "no coded instances, so nothing can be estimated"
        assert self.unseen_mass is not None and self.chao1 is not None and self.missing_codes is not None
        return (
            f"{self.n_codes} code(s) over {self.n_instances} instance(s); "
            f"singletons {self.f1}, doubletons {self.f2}; "
            f"Good-Turing unseen mass {self.unseen_mass:.1%}; "
            f"Chao1 {self.chao1:.1f} (about {self.missing_codes:.1f} more type(s) unseen)"
        )
