"""What a report is about, and how that is printed. Every stage reports under the same heading, so the
identity lives here rather than in one stage's reporter: the tables stack up under identical headers."""

from __future__ import annotations

from dataclasses import dataclass

from plybench.analysis.errors.facets import CELL_FACETS, facet_values
from plybench.analysis.errors.funnel.result import FunnelResult

# printed in a header and used as the sort key; the rest of a cell's facets stay in `to_dict` for the JSON
HEADLINE_FACETS: tuple[str, ...] = ("presentation", "model", "effort")


@dataclass(frozen=True, slots=True)
class Scope:
    """The coordinates a report is about, as an ordered facet -> value mapping. A cell names every facet;
    a pooled row names only the ones it grouped on, and the rest are simply absent rather than blank --
    which is what lets one set of report objects describe a single cell and a whole model's worth."""

    items: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, funnel: FunnelResult) -> Scope:
        return cls(facet_values(funnel))

    @property
    def experiment(self) -> str:
        return self.get("experiment")

    def get(self, facet: str) -> str:
        return next((value for name, value in self.items if name == facet), "")

    def only(self, *facets: str) -> Scope:
        """The scope of a pooled row: the facets it grouped on, in the order they were asked for. The
        experiment is always kept -- two experiments are two corpora, never one pooled row."""
        kept = ("experiment", *(facet for facet in facets if facet != "experiment"))
        return Scope(tuple((name, self.get(name)) for name in kept))

    def __str__(self) -> str:
        shown = [(name, value) for name, value in self.items if name in HEADLINE_FACETS] or list(self.items[1:])
        return "  |  ".join(f"{value}" for _, value in shown) or "all cells"

    def to_dict(self) -> dict[str, str]:
        return dict(self.items)


def cell(funnel: FunnelResult) -> str:
    return str(Scope.of(funnel))


def header(scope: Scope, n_moves: int) -> str:
    return f"\n{scope}  |  moves={n_moves}"


def describe(facets: tuple[str, ...]) -> str:
    return " x ".join(facet for facet in facets if facet in CELL_FACETS)
