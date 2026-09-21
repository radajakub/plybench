from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib.axes import Axes

from plybench.analysis.visual.core.layers.base import DrawContext
from plybench.analysis.visual.core.layout import Bounds
from plybench.analysis.visual.core.legend import LegendEntry
from plybench.analysis.visual.core.style import StyleOverride

Interval = tuple[float, float]
Run = list[tuple[float, Interval]]


@dataclass(frozen=True)
class LineLayer:
    x: tuple[float, ...]
    y: tuple[float | None, ...]
    label: str = ""
    band: tuple[Interval | None, ...] | None = None
    show_markers: bool = True
    x_offset: float = 0.0
    style: StyleOverride = field(default_factory=StyleOverride)

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(f"{len(self.x)} x values for {len(self.y)} y values")
        if self.band is not None and len(self.band) != len(self.y):
            raise ValueError(f"{len(self.band)} band entries for {len(self.y)} y values")

    @property
    def style_overrides(self) -> tuple[StyleOverride, ...]:
        return (self.style,)

    def positions(self) -> tuple[float, ...]:
        return tuple(position + self.x_offset for position in self.x)

    def bounds(self) -> Bounds | None:
        return Bounds.of(self.y, self.band)

    def x_bounds(self) -> Bounds | None:
        return Bounds.of(self.positions())

    def legend_entries(self, ctx: DrawContext) -> list[LegendEntry]:
        return [LegendEntry(self.label, ctx.styles[0])] if self.label else []

    def draw(self, ax: Axes, ctx: DrawContext) -> None:
        style = ctx.styles[0]
        ax.plot(
            list(self.positions()),
            [float("nan") if value is None else value for value in self.y],
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker if self.show_markers else "none",
            markersize=style.markersize,
            linewidth=style.linewidth,
            alpha=style.alpha,
            label=self.label,
            zorder=3,
        )
        for run in self._runs():
            if len(run) == 1:
                self._draw_whisker(ax, ctx, run[0])
                continue
            ax.fill_between(
                [position for position, _ in run],
                [interval[0] for _, interval in run],
                [interval[1] for _, interval in run],
                color=style.color,
                alpha=style.fill_alpha,
                linewidth=0,
                zorder=2,
            )

    def _draw_whisker(self, ax: Axes, ctx: DrawContext, point: tuple[float, Interval]) -> None:
        style = ctx.styles[0]
        position, (low, high) = point
        ax.errorbar(
            position,
            (low + high) / 2,
            yerr=(high - low) / 2,
            color=style.color,
            alpha=style.alpha,
            elinewidth=style.linewidth,
            capsize=style.markersize / 2,
            fmt="none",
            zorder=2,
        )

    def _runs(self) -> list[Run]:
        if self.band is None:
            return []
        runs: list[Run] = []
        current: Run = []
        for position, interval in zip(self.positions(), self.band, strict=True):
            if interval is None:
                if current:
                    runs.append(current)
                    current = []
                continue
            current.append((position, interval))
        return [*runs, current] if current else runs
