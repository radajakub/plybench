"""Benchmark glue: turns recorded results into the core's layers and panels.

Everything that knows about players, games and metrics lives here, so a new chart type can be added
to the core without touching the benchmark and vice versa.
"""

from plybench.analysis.visual.bench.encoder import EFFORT_MARKERS, ColorBy, StyleEncoder, StyleKey, marker_for
from plybench.analysis.visual.bench.labels import (
    Overrides,
    effort_key,
    game_label,
    game_labels,
    game_params,
    game_title,
    metric_label,
    model_key,
    model_name,
    model_strength,
    player_label,
    player_labels,
    provider_key,
    tier_rank,
)
from plybench.analysis.visual.bench.presets import LineOptions, dodges, interval_of, is_ratio, metric_axis, metric_panel, series_layer
from plybench.analysis.visual.bench.series import Series, SeriesPoint, SeriesRequest, TrackerIndex, build_series, build_series_batch, encoder_keys, indistinguishable, style_key

__all__ = [
    "EFFORT_MARKERS",
    "ColorBy",
    "LineOptions",
    "Overrides",
    "Series",
    "SeriesPoint",
    "SeriesRequest",
    "StyleEncoder",
    "StyleKey",
    "TrackerIndex",
    "build_series",
    "build_series_batch",
    "dodges",
    "effort_key",
    "encoder_keys",
    "game_label",
    "game_labels",
    "game_title",
    "game_params",
    "indistinguishable",
    "interval_of",
    "is_ratio",
    "marker_for",
    "metric_axis",
    "metric_label",
    "metric_panel",
    "model_key",
    "model_name",
    "model_strength",
    "player_label",
    "player_labels",
    "provider_key",
    "series_layer",
    "style_key",
    "tier_rank",
]
