from __future__ import annotations

import json
import math
import os
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.intervals import t_ci
from plybench.configs.names import item_names

# A job's report: what it plays, how far it got and its results so far, as one JSON file the running process rewrites while it works, so a tool
# (plyservice, then plyweb) shows a run by reading one file, never by starting a process or reading the results. plybench-run and plybench-train write
# one with `--job-report <path>`; a script writes its own with the same classes. The shape, version 1:
#
#   {version, updated, finished, unit,
#    items:    [{key, name, names: {axis: name}, configs: {axis: config}, planned, done, group?}],   the work this process does, nothing else
#    shared:   {axis: [what every item of the axis shares, left out of the names]},
#    progress: {done, total, unit},
#    blocks:   [grid | table | line | bars]}                                                       drawn in order
#
#   grid   {type, title, note?, columns: [label], rows: [{item, cells: [{done, total}]}]}
#          a group row instead: {group, cells, rows: [item row]}, with summed progress and its replicates for expansion
#   table  {type, title, note?, columns: [{key, label, rate?, digits?, name?}], rows: [{item?, <key>: value}]}
#          a value is a string, a number, a boolean, null, or {value, lower, upper} (an estimate with its interval); `rate`: values in [0, 1]; `name`:
#          the column holds an item key, shown as the item's name
#   line   {type, title, group?, measure?, note?, x: {label}, y: {label, rate}, series: [{item, points: [{x, value, lower?, upper?}]}],
#           references: [{label, value, games?}]}   games: the items' game configs a reference applies to (all without)
#          blocks with one `group` and different `measure`s are one panel with a choice of measure; series of items with one `group` are replicates.
#          A series may stand for the replicates of a group instead ({group, replicates, points}, replicate_series): each point the mean over the
#          replicates with a point there (n), its t interval when n >= 2, and their spread (q10, q90: the 10th and 90th percentiles)
#   bars   {type, title, note?, stack: [{key, label, role}], rows: [{item, values: {key: value}}]}
#          a role is how a part reads: good, good_other, neutral, bad, bad_other
#
# Numbers are finite or null (JSON has no NaN). A key is the item's results folder relative to the experiment's, unique within a report.

VERSION = 1
# at most one write in this many seconds while the process works (a script may pass its own); a change within them is written when they end
MIN_INTERVAL = 5.0


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_finite(entry) for entry in value]
    return value


def estimate(value: float | None, lower: float | None = None, upper: float | None = None) -> dict[str, float | None]:
    return {"value": value, "lower": lower, "upper": upper}


def _percentile(values: list[float], share: float) -> float:
    # linear between the closest ranks, as numpy's default
    ordered = sorted(values)
    at = share * (len(ordered) - 1)
    low = math.floor(at)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (at - low)


def _pooled_moves(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # the moves of the replicates' points as one count, most frequent first; `optimal` only where every point has it
    counts: dict[str, list[int]] = {}
    rated = bool(points)
    for point in points:
        for move in point.get("moves", []):
            entry = counts.setdefault(move["move"], [0, 0])
            entry[0] += move["count"]
            entry[1] += move.get("optimal", 0)
            rated = rated and "optimal" in move
    ordered = sorted(counts.items(), key=lambda item: (-item[1][0], item[0]))
    return [{"move": move, "count": count, **({"optimal": optimal} if rated else {})} for move, (count, optimal) in ordered]


def replicate_series(series: Sequence[dict[str, Any]], groups: dict[str, str | None], rate: bool, confidence: float = 0.95) -> list[dict[str, Any]]:
    # the series of a line block with the replicates of each group as one: per x the mean of the replicates with a point there, its t interval over
    # them (the variation between runs; a rate's bounds kept in [0, 1]) and their spread (q10, q90), with the moves pooled; a group of one item keeps
    # its own series and interval. groups: each item's group (Item.group). In the order of the groups' first items
    members: dict[str, list[dict[str, Any]]] = {}
    for entry in series:
        members.setdefault(groups.get(entry["item"]) or entry["item"], []).append(entry)
    out = []
    for group, entries in members.items():
        if len(entries) == 1:
            out.append(entries[0])
            continue
        by_x: dict[float, list[dict[str, Any]]] = {}
        for entry in entries:
            for point in entry["points"]:
                if point.get("value") is not None and math.isfinite(point["value"]):
                    by_x.setdefault(point["x"], []).append(point)
        points = []
        for x in sorted(by_x):
            values = [point["value"] for point in by_x[x]]
            interval = t_ci(Distribution(values), confidence)
            lower, upper = (max(0.0, interval.lower), min(1.0, interval.upper)) if rate else (interval.lower, interval.upper)
            point = {"x": x, "value": interval.value, "n": len(values), "q10": _percentile(values, 0.1), "q90": _percentile(values, 0.9)}
            if len(values) > 1:
                point |= {"lower": lower, "upper": upper}
            moves = _pooled_moves(by_x[x])
            points.append(point | ({"moves": moves} if moves else {}))
        out.append({"group": group, "replicates": len(entries), "points": points})
    return out


@dataclass
class Item:
    key: str
    configs: dict[str, str]
    planned: int
    done: int = 0
    group: str | None = None


@dataclass
class Report:
    unit: str = "games"
    items: list[Item] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    # the experiment file's labels by config string, which name the items in place of the short form
    labels: dict[str, str] = field(default_factory=dict)
    # an item's axis shown with a prefix in its name (e.g. "vs " before an opponent)
    prefixes: dict[str, str] = field(default_factory=dict)
    finished: bool = False

    def add_grid(self, title: str, columns: Sequence[str], rows: Sequence[dict[str, Any]], note: str | None = None) -> None:
        self.blocks.append({"type": "grid", "title": title, **({"note": note} if note else {}), "columns": list(columns), "rows": list(rows)})

    def add_table(self, title: str, columns: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], note: str | None = None) -> None:
        self.blocks.append({"type": "table", "title": title, **({"note": note} if note else {}), "columns": list(columns), "rows": list(rows)})

    def add_line(
        self,
        title: str,
        x: str,
        y: str,
        rate: bool,
        series: Sequence[dict[str, Any]],
        references: Sequence[dict[str, Any]] = (),
        group: str | None = None,
        measure: str | None = None,
        note: str | None = None,
    ) -> None:
        block: dict[str, Any] = {"type": "line", "title": title}
        block |= {"group": group} if group is not None else {}
        block |= {"measure": measure} if measure is not None else {}
        block |= {"note": note} if note else {}
        self.blocks.append(block | {"x": {"label": x}, "y": {"label": y, "rate": rate}, "series": list(series), "references": list(references)})

    def add_bars(self, title: str, stack: Sequence[dict[str, str]], rows: Sequence[dict[str, Any]], note: str | None = None) -> None:
        self.blocks.append({"type": "bars", "title": title, **({"note": note} if note else {}), "stack": list(stack), "rows": list(rows)})

    def to_dict(self) -> dict[str, Any]:
        names, joined, shared = item_names([item.configs for item in self.items], self.labels, self.prefixes)
        items = [
            {"key": item.key, "name": name, "names": axes, "configs": item.configs, "planned": item.planned, "done": item.done, **({"group": item.group} if item.group else {})}
            for item, axes, name in zip(self.items, names, joined, strict=True)
        ]
        progress = {"done": sum(item.done for item in self.items), "total": sum(item.planned for item in self.items), "unit": self.unit}
        updated = datetime.now(UTC).isoformat(timespec="seconds")
        return _finite(
            {"version": VERSION, "updated": updated, "finished": self.finished, "unit": self.unit, "items": items, "shared": shared, "progress": progress, "blocks": self.blocks}
        )

    def save(self, path: Path) -> None:
        # written to a temporary file, then renamed over the report, so a reader never sees half of one
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), allow_nan=False))
        os.replace(temporary, path)


class ReportFile:
    # a report rewritten while a process works: `changed` builds and saves it at most every min_interval seconds, `write` and `close` at once. A change
    # within the interval is not dropped: a timer writes it when the interval ends (one write for all changes within it), so the file is at most one
    # interval behind. The timer builds the report on its own thread while the process may change what that reads; a build that fails there is a
    # warning, and the next change or `close` writes again. A failure to build or write is a warning, never an error: a report must not stop a run
    def __init__(self, path: Path, build: Callable[[], Report], min_interval: float = MIN_INTERVAL) -> None:
        self.path = path
        self.build = build
        self.min_interval = min_interval
        self._last = -math.inf
        self._timer: threading.Timer | None = None
        # one write at a time: the timer's and the process's would share the temporary file
        self._lock = threading.Lock()

    def changed(self) -> None:
        with self._lock:
            if self._timer is not None:
                return
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                self._timer = threading.Timer(wait, self._pending)
                # a daemon: a timer still waiting never keeps the process alive
                self._timer.daemon = True
                self._timer.start()
                return
        self.write()

    def _pending(self) -> None:
        with self._lock:
            self._timer = None
        self.write()

    def write(self, finished: bool = False) -> None:
        with self._lock:
            # this write has the newest state, so a change waiting for the timer is written by it
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._last = time.monotonic()
            try:
                report = self.build()
                report.finished = finished
                report.save(self.path)
            except Exception as error:
                # any error: the run goes on, and the next change tries again
                warnings.warn(f"the report {self.path} was not written: {type(error).__name__}: {error}", stacklevel=2)

    def close(self, finished: bool) -> None:
        self.write(finished)
