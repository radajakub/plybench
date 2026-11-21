"""List, show and save experiments in plybench's store (PLYBENCH_STORE; by default the files under experiments/).

Examples:
    uv run plybench-experiment list                                  # every experiment, as <kind>/<name>
    uv run plybench-experiment list --json                           # every experiment with its config, as JSON
    uv run plybench-experiment show benchmarks/ttt                   # one experiment, as JSON
    uv run plybench-experiment put benchmarks/ttt ttt.json           # a new experiment, or a new version of one (- reads standard input)
    uv run plybench-experiment edit benchmarks/ttt                   # the current version in $EDITOR, saved when the editor closes

A new version is checked as the harness would load it. It cannot remove an item of an experiment that has results: disable it (enabled: false).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from plybench.harness.experiments import KINDS, experiment_key, list_experiments, load_experiment, save_experiment


def _named(value: str) -> tuple[str, str]:
    kind, _, name = value.partition("/")
    try:
        experiment_key(kind, name)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value}: {error} (expected <kind>/<name>, e.g. benchmarks/ttt)") from error
    return kind, name


def _read(source: str) -> Any:
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="every experiment")
    listing.add_argument("--kind", choices=KINDS, help="only this kind")
    listing.add_argument("--json", action="store_true", help="with their configs, as JSON by kind and name")
    commands.add_parser("show", help="one experiment as JSON").add_argument("experiment", type=_named, help="<kind>/<name>")
    put = commands.add_parser("put", help="save a new experiment or a new version of one")
    put.add_argument("experiment", type=_named, help="<kind>/<name>")
    put.add_argument("file", help="the JSON file, - for standard input")
    commands.add_parser("edit", help="edit the current version in $EDITOR and save it").add_argument("experiment", type=_named, help="<kind>/<name>")
    args = parser.parse_args()

    if args.command == "list":
        found = {kind: list_experiments(kind) for kind in KINDS if args.kind in (None, kind)}
        if args.json:
            print(json.dumps(found, ensure_ascii=False))
        else:
            print("\n".join(f"{kind}/{name}" for kind, experiments in found.items() for name in experiments))
        return
    kind, name = args.experiment
    try:
        if args.command == "show":
            print(json.dumps(load_experiment(kind, name), ensure_ascii=False, indent=2))
        elif args.command == "put":
            save_experiment(kind, name, _read(args.file))
            print(f"saved {experiment_key(kind, name)}")
        else:
            _edit(kind, name)
    except FileNotFoundError as error:
        parser.exit(2, f"{error}: does not exist\n")
    except ValueError as error:
        parser.exit(1, f"{error}\n")


def _edit(kind: str, name: str) -> None:
    # the editor works on a copy; a version the store refuses stays in the copy, named, so the edit is not lost
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix=f"{name}.", delete=False, encoding="utf-8") as handle:
        handle.write(json.dumps(load_experiment(kind, name), ensure_ascii=False, indent=2) + "\n")
        path = Path(handle.name)
    before = path.read_text(encoding="utf-8")
    editor = shlex.split(os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi")
    if subprocess.run([*editor, str(path)]).returncode != 0:
        raise ValueError(f"the editor failed; the edit is in {path}")
    if path.read_text(encoding="utf-8") == before:
        path.unlink()
        print("no change")
        return
    try:
        save_experiment(kind, name, _read(str(path)))
    except ValueError as error:
        raise ValueError(f"{error}\nnot saved; the edit is in {path} (save it with: plybench-experiment put {kind}/{name} {path})") from error
    path.unlink()
    print(f"saved {experiment_key(kind, name)}")


if __name__ == "__main__":
    main()
