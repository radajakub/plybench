from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.typing import LegendLocType

from plybench.analysis.visual.core.style import SeriesStyle, Style

Placement = Literal["below", "right", "inside"]


@dataclass(frozen=True)
class LegendEntry:
    label: str
    style: SeriesStyle
    kind: Literal["line", "patch"] = "line"


@dataclass(frozen=True)
class LegendSpec:
    placement: Placement = "below"
    columns: int = 3
    frame: bool = False
    show_markers: bool = True


def dedupe(entries: Iterable[LegendEntry]) -> list[LegendEntry]:
    seen: dict[str, LegendEntry] = {}
    for entry in entries:
        seen.setdefault(entry.label, entry)
    return list(seen.values())


def proxy(entry: LegendEntry, spec: LegendSpec) -> Artist:
    style = entry.style
    if entry.kind == "patch":
        return Patch(facecolor=style.color, alpha=style.alpha, linewidth=0)
    return Line2D(
        [],
        [],
        color=style.color,
        linestyle=style.linestyle,
        marker=style.marker if spec.show_markers else "none",
        markersize=style.markersize,
        linewidth=style.linewidth,
    )


def _anchor(spec: LegendSpec) -> tuple[LegendLocType, tuple[float, float] | None, int]:
    if spec.placement == "below":
        return ("upper center", (0.5, -0.02), spec.columns)
    if spec.placement == "right":
        return ("center left", (1.02, 0.5), 1)
    return ("best", None, spec.columns)


def draw_figure_legend(figure: MplFigure, entries: list[LegendEntry], spec: LegendSpec, style: Style) -> None:
    if not entries:
        return
    location, anchor, columns = _anchor(spec)
    figure.legend(
        [proxy(entry, spec) for entry in entries],
        [entry.label for entry in entries],
        loc=location,
        bbox_to_anchor=anchor,
        ncol=columns,
        frameon=spec.frame,
        labelcolor=style.text_primary,
    )


def draw_panel_legend(ax: Axes, entries: list[LegendEntry], spec: LegendSpec, style: Style) -> None:
    if not entries:
        return
    location, anchor, columns = _anchor(spec)
    ax.legend(
        [proxy(entry, spec) for entry in entries],
        [entry.label for entry in entries],
        loc=location,
        bbox_to_anchor=anchor,
        ncol=columns,
        frameon=spec.frame,
        labelcolor=style.text_primary,
    )
