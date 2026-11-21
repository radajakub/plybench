from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import posixpath
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

# Where results are kept. Every result a run writes or reads back (games, matchup metadata, run manifests, learner updates, states and checkpoints, the
# llm blocks log) goes through one Store, by key: the relative path the file has always had ("results/benchmarks/.../game_1.json"). By default the
# keys are files under the working directory, as before; another store (a database) is plugged in by naming an entry point of the plybench.stores
# group in PLYBENCH_STORE, so plybench never depends on one. A key ending in .json holds a JSON value, .jsonl an append-only log, anything else text.

STORES = "plybench.stores"
STORE_ENV = "PLYBENCH_STORE"


class RecordExists(Exception):
    # a write-once record is there already (put without replace, put_many)
    pass


class Store(Protocol):
    def put(self, key: str, value: Any, *, replace: bool = False) -> None:
        # write-once unless replace (RecordExists)
        ...

    def put_many(self, values: Mapping[str, Any]) -> None:
        # several new records, all or none (a learner's update or state); refuses all if one exists
        ...

    def get(self, key: str) -> Any | None:
        # the record, None when there is none
        ...

    def get_many(self, keys: Sequence[str]) -> dict[str, Any]:
        # the records of those keys that exist, in one call (a whole matchup for a report); missing keys are left out
        ...

    def exists(self, key: str) -> bool: ...

    def keys(self, prefix: str = "") -> list[str]:
        # every key under the prefix (by whole parts), sorted
        ...

    def append(self, key: str, record: Any) -> None:
        # one JSON record at the end of a .jsonl log; appends from processes at the same time never interleave
        ...

    def read_log(self, key: str) -> list[Any]:
        # the records of a log in order; [] when there is none
        ...


def _encode(key: str, value: Any) -> str:
    # compact JSON without NaN or infinities (a database cannot hold them, and they are not JSON); text as it is
    if key.endswith(".json") or key.endswith(".jsonl"):
        try:
            return json.dumps(value, allow_nan=False, ensure_ascii=key.endswith(".json"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key}: not storable JSON ({error})") from error
    if not isinstance(value, str):
        raise ValueError(f"{key} is text: give a str, not {type(value).__name__}")
    return value


def _decode(key: str, text: str) -> Any:
    return json.loads(text) if key.endswith(".json") else text


def _check(key: str) -> str:
    if not key or key.startswith("/") or "" in key.split("/") or any(part.startswith(".") for part in key.split("/")) or key.endswith((".tmp", ".lock")):
        raise ValueError(f"invalid key {key!r}: a relative name of parts joined by '/', none empty or starting with '.', not ending in .tmp or .lock")
    return key


class FileStore:
    # each record a file under a root folder: a temporary file, fsync, then a rename (a reader never sees half a record); logs appended under flock
    _STAGING = ".plybench-staging"

    def __init__(self, root: Path = Path(".")) -> None:
        self.root = root

    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def put(self, key: str, value: Any, *, replace: bool = False) -> None:
        path = self.root / _check(key)
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        self._write(temporary, _encode(key, value))
        try:
            if replace:
                os.replace(temporary, path)
            else:
                # a hard link fails on an existing name: write-once without a window between the check and the write
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    raise RecordExists(key) from None
        finally:
            temporary.unlink(missing_ok=True)

    def put_many(self, values: Mapping[str, Any]) -> None:
        texts = {_check(key): _encode(key, value) for key, value in values.items()}
        if not texts:
            return
        if there := [key for key in texts if (self.root / key).exists()]:
            raise RecordExists(", ".join(there))
        staging = self.root / self._STAGING / uuid.uuid4().hex
        try:
            for key, text in texts.items():
                self._write(staging / key, text)
            folders = [posixpath.dirname(key) for key in texts]
            folder = "" if "" in folders else posixpath.commonpath(folders)
            target = self.root / folder
            # a new folder holding all of them appears at once by one rename; otherwise each is linked in, and taken out again if one is there
            if folder and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.rename(staging / folder, target)
                    return
                except OSError:
                    pass
            linked: list[Path] = []
            try:
                for key in texts:
                    path = self.root / key
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.link(staging / key, path)
                    linked.append(path)
            except FileExistsError:
                for path in linked:
                    path.unlink(missing_ok=True)
                raise RecordExists(", ".join(key for key in texts if (self.root / key).exists())) from None
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            try:
                os.rmdir(self.root / self._STAGING)
            except OSError:
                pass

    def get(self, key: str) -> Any | None:
        path = self.root / _check(key)
        return _decode(key, path.read_text(encoding="utf-8")) if path.is_file() else None

    def get_many(self, keys: Sequence[str]) -> dict[str, Any]:
        return {key: value for key in keys if (value := self.get(key)) is not None}

    def exists(self, key: str) -> bool:
        return (self.root / _check(key)).is_file()

    def keys(self, prefix: str = "") -> list[str]:
        base = self.root / _check(prefix) if prefix else self.root
        if base.is_file():
            return [prefix]
        found = []
        for folder, subfolders, files in os.walk(base):
            subfolders[:] = [name for name in subfolders if not name.startswith(".")]
            for name in files:
                if not name.startswith(".") and not name.endswith((".tmp", ".lock")):
                    found.append((Path(folder) / name).relative_to(self.root).as_posix())
        return sorted(found)

    def append(self, key: str, record: Any) -> None:
        line = _encode(_check(key), record)
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(line + "\n")
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)

    def read_log(self, key: str) -> list[Any]:
        path = self.root / _check(key)
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.is_file() else []


def open_store() -> Store:
    # files under the working directory, or the store the entry point named in PLYBENCH_STORE builds for it
    name = os.environ.get(STORE_ENV)
    if not name:
        return FileStore()
    found = importlib.metadata.entry_points(group=STORES, name=name)
    if not found:
        raise ValueError(
            f"{STORE_ENV}={name}: no such store is installed (entry points of {STORES}: {', '.join(sorted(point.name for point in importlib.metadata.entry_points(group=STORES)))})"
        )
    return next(iter(found)).load()(Path("."))


_store: Store | None = None


def store() -> Store:
    # the store of this process, opened on first use
    global _store
    if _store is None:
        _store = open_store()
    return _store


def use_store(replacement: Store | None) -> None:
    # another store for this process (a test, a script); None opens the configured one again on next use
    global _store
    _store = replacement
