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
