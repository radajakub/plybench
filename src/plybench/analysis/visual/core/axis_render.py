from __future__ import annotations

from matplotlib.axes import Axes

from plybench.analysis.visual.core.axis import Axis, resolve_limits
from plybench.analysis.visual.core.layout import Bounds
from plybench.analysis.visual.core.style import Style
from plybench.analysis.visual.core.ticks import Which


def apply_axis(ax: Axes, axis: Axis, which: Which, bounds: Bounds | None, style: Style) -> bool:
    _apply_side(ax, axis, which)
    if axis.scale != "linear":
        ax.set_xscale(axis.scale) if which == "x" else ax.set_yscale(axis.scale)

    limits = resolve_limits(axis, bounds)
    if limits is not None:
        ax.set_xlim(*limits) if which == "x" else ax.set_ylim(*limits)

    if axis.ticks is not None:
        axis.ticks.apply(ax, which)
    _apply_rotation(ax, axis, which)
    return _apply_sci(ax, axis, which, style)


def _apply_side(ax: Axes, axis: Axis, which: Which) -> None:
    if axis.side is None:
        return
    if which == "y" and axis.side in ("left", "right"):
        ax.yaxis.set_label_position(axis.side)
        ax.yaxis.set_ticks_position(axis.side)
    elif which == "x" and axis.side in ("bottom", "top"):
        ax.xaxis.set_label_position(axis.side)
        ax.xaxis.set_ticks_position(axis.side)


def _apply_rotation(ax: Axes, axis: Axis, which: Which) -> None:
    if not axis.rotation:
        return
    for label in ax.get_xticklabels() if which == "x" else ax.get_yticklabels():
        label.set_rotation(axis.rotation)
        label.set_horizontalalignment("right")


def _apply_sci(ax: Axes, axis: Axis, which: Which, style: Style) -> bool:
    if axis.sci is None or axis.scale != "linear":
        return False
    ax.ticklabel_format(axis=which, style="sci", scilimits=axis.sci)
    mpl_axis = ax.xaxis if which == "x" else ax.yaxis
    offset = mpl_axis.get_offset_text()
    offset.set_fontsize(style.font_size - 1)
    offset.set_color(style.text_secondary)
    # whether an offset shows can be learned without a canvas draw, but only after the locator has run
    formatter = mpl_axis.get_major_formatter()
    formatter.set_locs(list(mpl_axis.get_ticklocs()))
    return bool(formatter.get_offset())
