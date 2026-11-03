from __future__ import annotations

import fcntl
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, ParamSpec, Self, TypeVar

DeserializeArgs = ParamSpec("DeserializeArgs")
LoadArgs = ParamSpec("LoadArgs")
LoadPath = TypeVar("LoadPath", str, Path)


def require_json_round_trip(value: Any, what: str) -> None:
    # a value kept as JSON must read back equal, or what a run uses now differs from what a resumed run reads (tuples come back
    # as lists, non-string keys as strings, numpy integers do not serialise); refused when it is made, not found when it is read
    try:
        restored = json.loads(json.dumps(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{what} is not JSON: {error}") from error
    if restored != value:
        raise ValueError(f"{what} does not read back from JSON unchanged; keep to dicts with string keys, lists, strings, numbers, booleans and None")


class Serializable(ABC, Generic[DeserializeArgs]):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any], *args: DeserializeArgs.args, **kwargs: DeserializeArgs.kwargs) -> Self:
        raise NotImplementedError


class Saveable(Serializable[DeserializeArgs], Generic[DeserializeArgs, LoadPath, LoadArgs]):
    def save(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=None)

    @classmethod
    @abstractmethod
    def load(cls, filepath: LoadPath, *args: LoadArgs.args, **kwargs: LoadArgs.kwargs) -> Self:
        raise NotImplementedError


class ThreadSafeSaveable(Serializable[[]]):
    def save(self, filepath: str) -> None:
        lock_path = Path(filepath).with_suffix(".lock")
        tmp_path = Path(filepath).with_suffix(".tmp")

        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            # write to a temporary file first to prevent corruption, then atomically rename
            with open(tmp_path, "w") as tmp:
                json.dump(self.to_dict(), tmp, allow_nan=True, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())

            os.replace(tmp_path, filepath)

            fcntl.flock(lock, fcntl.LOCK_UN)

    def cleanup(self, filepath: str, part_filepath: str) -> None:
        Path(filepath).with_suffix(".lock").unlink(missing_ok=True)
        Path(filepath).with_suffix(".tmp").unlink(missing_ok=True)
        Path(part_filepath).unlink(missing_ok=True)
        Path(part_filepath).with_suffix(".lock").unlink(missing_ok=True)


class ThreadSafeAppendable(Serializable[[]]):
    def append(self, filepath: Path) -> None:
        # adds one JSON line. The file is locked while it is written, so lines from processes appending at
        # the same time are never interleaved
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
            file.flush()
            fcntl.flock(file, fcntl.LOCK_UN)
