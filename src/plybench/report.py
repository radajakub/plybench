from __future__ import annotations

import json
import math
import os
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
#   table  {type, title, note?, columns: [{key, label, rate?, digits?, name?}], rows: [{item?, <key>: value}]}
#          a value is a string, a number, a boolean, null, or {value, lower, upper} (an estimate with its interval); `rate`: values in [0, 1]; `name`:
#          the column holds an item key, shown as the item's name
#   line   {type, title, group?, measure?, note?, x: {label}, y: {label, rate}, series: [{item, points: [{x, value, lower?, upper?}]}],
#           references: [{label, value}]}
#          blocks with one `group` and different `measure`s are one panel with a choice of measure; series of items with one `group` are replicates
#   bars   {type, title, note?, stack: [{key, label, role}], rows: [{item, values: {key: value}}]}
#          a role is how a part reads: good, good_other, neutral, bad, bad_other
#
# Numbers are finite or null (JSON has no NaN). A key is the item's results folder relative to the experiment's, unique within a report.

VERSION = 1
# at most one write in this many seconds while the process works; the last one is always written
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
    # a report rewritten while a process works: `changed` builds and saves it at most every MIN_INTERVAL seconds, `close` always. A failure to build or
    # write is a warning, never an error: a report must not stop a run
    def __init__(self, path: Path, build: Callable[[], Report], min_interval: float = MIN_INTERVAL) -> None:
        self.path = path
        self.build = build
        self.min_interval = min_interval
        self._last = -math.inf

    def changed(self) -> None:
        if time.monotonic() - self._last >= self.min_interval:
            self.write()

    def write(self, finished: bool = False) -> None:
        self._last = time.monotonic()
        try:
            report = self.build()
            report.finished = finished
            report.save(self.path)
        except Exception as error:
            # any error: the run goes on, and the next game tries again
            warnings.warn(f"the report {self.path} was not written: {type(error).__name__}: {error}", stacklevel=2)

    def close(self, finished: bool) -> None:
        self.write(finished)
