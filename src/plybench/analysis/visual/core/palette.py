from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from plybench.analysis.visual.core.style import Linestyle, SeriesStyle

# Validated categorical palette (light surface, adjacent pairlist -- the pairlist that applies to
# lines): worst adjacent CVD dE 9.1, worst normal-vision dE 19.6. The ORDER is the colour-blindness
# safety mechanism, not cosmetic, so slots are taken from the front and never reordered or cycled.
CATEGORICAL: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# Ordered from solid to sparse. Dotted sits third so that the common case of three plotted groups
# reads as solid / dashed / dotted.
LINESTYLES: tuple[Linestyle, ...] = ("-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (7, 2)))

MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
ChannelT = TypeVar("ChannelT")


@dataclass(frozen=True)
class Palette:
    colors: tuple[str, ...] = CATEGORICAL
    linestyles: tuple[Linestyle, ...] = LINESTYLES
    markers: tuple[str, ...] = MARKERS
    vary_linestyle: bool = False
    vary_marker: bool = False

    def slot(self, index: int) -> SeriesStyle:
        if index >= len(self.colors):
            raise ValueError(f"palette slot {index} exceeds the {len(self.colors)} available colours; supply explicit styles or a larger palette")
        return SeriesStyle(
            color=self.colors[index],
            linestyle=self._pick(self.linestyles, index, self.vary_linestyle, "-"),
            marker=self._pick(self.markers, index, self.vary_marker, "o"),
        )

    @staticmethod
    def _pick(channel: tuple[ChannelT, ...], index: int, vary: bool, default: ChannelT) -> ChannelT:
        if not vary:
            return default
        if index >= len(channel):
            raise ValueError(f"palette slot {index} exceeds the {len(channel)} available entries in a varying channel")
        return channel[index]
