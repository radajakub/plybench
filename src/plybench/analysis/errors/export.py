"""One line per graded move, keyed by the same move uid the judge's own stores use. Stage 2 keeps nothing
on disk because it is free to recompute, but that also means its per-move numbers can only ever be joined
to anything inside one process. This is the join key made durable: with it, "were the moves the judge
called self-corrected refuted more shallowly?" is a question a notebook can ask. Aggregates cannot answer
it -- only the rows can."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from plybench.analysis.errors.analysis import Analysis
from plybench.analysis.errors.moves import MatchupId, TracedMove
from plybench.analysis.errors.procedural.detection import MoveDiagnosis
from plybench.analysis.errors.split import SplitConfig, split_for

CellIndex = dict[MatchupId, int]


def _row(move: TracedMove, diagnosis: MoveDiagnosis, cell: int, split_config: SplitConfig) -> dict[str, Any]:
    row: dict[str, Any] = {
        "uid": move.uid,
        "cell": cell,
        "round": move.game_round,
        "seq": move.seq,
        "stage": move.stage.value,
        "analysis_split": split_for(move, config=split_config).value,
        "output_failure": move.output_failure.value if move.output_failure is not None else None,
        "position": move.record.state_class.value,
        "optimal": move.record.is_optimal,
        "regret": move.record.regret,
        "labels": sorted(label.value for label in diagnosis.labels),
    }
    # absent rather than null on the moves the solver agreed with, which are most of the file: there is no
    # refutation to record for a move that needed none, and a null would invite it being read as a zero
    if diagnosis.refutation is not None:
        row.update(diagnosis.refutation.to_dict())
    return row


def move_rows(analysis: Analysis, cells: CellIndex) -> Iterator[dict[str, Any]]:
    return (_row(move, diagnosis, cells.setdefault(move.matchup, len(cells)), analysis.funnel.split_config) for move, diagnosis in analysis.graded)


def _index(cells: CellIndex) -> list[dict[str, Any]]:
    ordered = sorted(cells.items(), key=lambda item: item[1])
    return [{"cell": cell, "experiment": m.experiment, "game": m.game, "player": m.player, "opponent": m.opponent} for m, cell in ordered]


def write_moves(path: Path, analyses: Iterable[Analysis]) -> tuple[int, Path]:
    """Rows to `path`, their cell table beside it. A cell's identity is four config strings that repeat on
    every one of its moves and would otherwise be half the file, so the rows carry an integer and the table
    carries the strings once. The rows stay uniform, which is what keeps the file a single read."""
    cells: CellIndex = {}
    written = 0
    with path.open("w") as handle:
        for analysis in analyses:
            for row in move_rows(analysis, cells):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                written += 1

    index = path.with_suffix(".cells.json")
    index.write_text(json.dumps(_index(cells), indent=2))
    return written, index
