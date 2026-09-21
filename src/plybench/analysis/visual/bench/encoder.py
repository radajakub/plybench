from __future__ import annotations

from dataclasses import dataclass

from plybench.analysis.visual.core.palette import CATEGORICAL, LINESTYLES
from plybench.analysis.visual.core.style import Linestyle, SeriesStyle
from plybench.llm.options import ReasoningEffort
from plybench.utils.enums import ExtendedEnum

# Reasoning effort rides on the marker rather than the line, keeping the linestyle free to mean
# "tier" alone. The shapes are ordered so a downward triangle reads as less effort than an upward one.
EFFORT_MARKERS: tuple[tuple[ReasoningEffort | None, str], ...] = ((None, "o"), ("minimal", "."), ("low", "v"), ("medium", "s"), ("high", "^"), ("xhigh", "D"), ("max", "*"))


class ColorBy(ExtendedEnum):
    PROVIDER = "provider"
    PLAYER = "player"


@dataclass(frozen=True)
class StyleKey:
    color: str
    line: str
    effort: ReasoningEffort | None
    strength: tuple[int, tuple[float, ...], str]


def marker_for(effort: str | None) -> str:
    for known, marker in EFFORT_MARKERS:
        if known == effort:
            return marker
    raise ValueError(f"no marker defined for reasoning effort {effort!r}; expected one of {', '.join(str(known) for known, _ in EFFORT_MARKERS)}")


class StyleEncoder:
    def __init__(self, keys: list[StyleKey], color_keys: list[StyleKey] | None = None) -> None:
        groups: dict[str, dict[str, tuple[int, tuple[float, ...], str]]] = {}
        for key in keys:
            groups.setdefault(key.color, {})[key.line] = key.strength
        colors = list(dict.fromkeys(key.color for key in (color_keys if color_keys is not None else keys)))
        if len(colors) > len(CATEGORICAL):
            raise ValueError(
                f"{len(colors)} colour groups exceeds the {len(CATEGORICAL)}-slot categorical palette ({', '.join(colors)}); filter the selection or set color_by=provider"
            )
        for color, lines in groups.items():
            if len(lines) > len(LINESTYLES):
                raise ValueError(
                    f"colour group {color!r} has {len(lines)} models, more than the {len(LINESTYLES)} available line styles ({', '.join(lines)}); filter the selection"
                )
        self._colors = {color: CATEGORICAL[index] for index, color in enumerate(colors)}
        # a drawn key absent from the roster would fail with a bare KeyError at draw time, one series in
        unknown = sorted({key.color for key in keys} - set(self._colors))
        if unknown:
            raise ValueError(f"colour groups {', '.join(unknown)} are drawn but missing from the colour roster; pass every drawn group in color_keys")
        # strongest first, so the flagship of every provider gets the solid line
        self._lines: dict[tuple[str, str], Linestyle] = {
            (color, line): LINESTYLES[index] for color, lines in groups.items() for index, line in enumerate(sorted(lines, key=lambda name: lines[name], reverse=True))
        }

    def style(self, key: StyleKey) -> SeriesStyle:
        return SeriesStyle(self._colors[key.color], self._lines[(key.color, key.line)], marker_for(key.effort))
