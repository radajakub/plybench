from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

# always recorded; the packages of the players are added by the caller
_CORE_PACKAGES = ("plybench", "plyllm")


def _git(directory: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# once per process: a run opens many matchups, and the code does not change under it
@cache
def _commit(directory: Path) -> dict[str, Any] | None:
    # the checkout a directory belongs to, and whether it has uncommitted changes to tracked files; None outside a checkout
    commit = _git(directory, "rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git(directory, "status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status)}


@cache
def _package(name: str) -> dict[str, Any]:
    try:
        version: str | None = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    spec = importlib.util.find_spec(name)
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    # an installed copy lives in site-packages, inside whatever checkout holds the environment; its commit is not that one's
    source = Path(locations[0]) if locations and "site-packages" not in Path(locations[0]).parts else None
    return {"version": version, "source": _commit(source) if source is not None else None}


def provenance(packages: Iterable[str] = ()) -> dict[str, Any]:
    # what produced a session's results: package versions, the commit of each package run from a checkout and of the working
    # directory (where the experiment configs live), and the command
    names = sorted({*_CORE_PACKAGES, *packages})
    return {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "packages": {name: _package(name) for name in names},
        "working_directory": _commit(Path.cwd()),
        "command": sys.argv,
        "python": platform.python_version(),
    }


def _differences(old: Any, new: Any, path: str) -> list[str]:
    # the leaves where two JSON values differ, each as "path: old -> new"
    if isinstance(old, dict) and isinstance(new, dict):
        return [line for key in sorted(old.keys() | new.keys()) for line in _differences(old.get(key), new.get(key), f"{path}.{key}" if path else key)]
    return [] if old == new else [f"{path}: {old!r} -> {new!r}"]


def session_changes(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    # what makes a session other than the one it continues: another player (every param, defaults included) or another version of plybench,
    # plyllm or a player's package (its version, and its commit when it runs from a checkout). Empty when the two may share results
    if "players" not in previous:
        return ["the earlier session recorded no player descriptions (plybench before 4.12)"]
    changes = _differences(previous["players"], current["players"], "players")
    old = previous.get("packages", {})
    changes += [line for name, package in current["packages"].items() for line in _differences(old.get(name), package, f"packages.{name}")]
    return changes


def package_of(obj: object) -> str:
    # the top-level package that defines a class or function, e.g. "plyagents" for the tac player's params
    module = obj.__module__ if isinstance(obj, type) else type(obj).__module__
    return module.split(".")[0]
