from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol


class MoveVerdict(Protocol):
    """What every per-move LLM verdict has in common: it is about one move, it was produced by one named
    annotator, and it serialises to a dict. Keyed by that pair so the same move can be judged twice by
    different annotators for inter-rater reliability without either overwriting the other."""

    @property
    def key(self) -> tuple[str, str]: ...

    def to_dict(self) -> dict[str, Any]: ...


class VerdictStore[T: MoveVerdict]:
    """Append-only JSONL of per-move verdicts. This is research data, not cache: LLM judgement is
    non-deterministic, so a deleted store does not rebuild to the same verdicts. Appending rather than
    rewriting means a crashed run keeps everything it had already paid for."""

    def __init__(self, path: Path, parse: Callable[[dict[str, Any]], T]) -> None:
        self.path = path
        self._parse = parse
        self._records: dict[tuple[str, str], T] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            if line.strip():
                record = self._parse(json.loads(line))
                self._records[record.key] = record  # a later line supersedes an earlier one

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[T]:
        return iter(self._records.values())

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._records

    def annotated_by(self, annotator: str) -> set[str]:
        """Move uids this annotator has already covered — what a resumed run should skip."""
        return {uid for uid, name in self._records if name == annotator}

    def by_move(self, annotator: str) -> dict[str, T]:
        """One annotator's verdicts keyed by move uid — the shape every join against the funnel needs."""
        return {uid: record for (uid, name), record in self._records.items() if name == annotator}

    def annotators(self) -> list[str]:
        return sorted({name for _, name in self._records})

    def add(self, record: T) -> None:
        self._records[record.key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")

    def extend(self, records: list[T]) -> None:
        for record in records:
            self.add(record)

    def double_annotated(self) -> list[str]:
        """Move uids carrying more than one annotator's verdict — the reliability subsample."""
        counts: dict[str, int] = {}
        for uid, _ in self._records:
            counts[uid] = counts.get(uid, 0) + 1
        return sorted(uid for uid, count in counts.items() if count > 1)
