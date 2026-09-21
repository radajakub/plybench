from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

from matplotlib.typing import LineStyleType, RcKeyType

GRID_COLOR = "#d9d8d4"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

Linestyle = LineStyleType


@dataclass(frozen=True)
class SeriesStyle:
    color: str
    linestyle: Linestyle = "-"
    marker: str = "o"
    linewidth: float = 1.2
    markersize: float = 4.0
    alpha: float = 1.0
    fill_alpha: float = 0.12


NEUTRAL = SeriesStyle(color=TEXT_SECONDARY)


@dataclass(frozen=True)
class StyleOverride:
    color: str | None = None
    linestyle: Linestyle | None = None
    marker: str | None = None
    linewidth: float | None = None
    markersize: float | None = None
    alpha: float | None = None
    fill_alpha: float | None = None

    def over(self, base: SeriesStyle) -> SeriesStyle:
        set_fields = {field.name: getattr(self, field.name) for field in fields(self) if getattr(self, field.name) is not None}
        return replace(base, **set_fields)


@dataclass(frozen=True)
class Style:
    font_size: float = 10.0
    title_size: float = 13.0
    dpi: int = 200
    grid: bool = True
    grid_axis: Literal["both", "x", "y"] = "y"
    grid_width: float = 0.8
    hide_spines: tuple[str, ...] = ("top", "right")
    grid_color: str = GRID_COLOR
    text_primary: str = TEXT_PRIMARY
    text_secondary: str = TEXT_SECONDARY
    surface: str = SURFACE

    def rc(self) -> dict[RcKeyType, Any]:
        return {
            "font.size": self.font_size,
            "axes.titlesize": self.font_size + 1,
            "axes.labelsize": self.font_size,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
            "legend.fontsize": self.font_size - 1,
            "figure.facecolor": self.surface,
            "axes.facecolor": self.surface,
            "savefig.facecolor": self.surface,
        }
