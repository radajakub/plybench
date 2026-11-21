"""Print one saved game as a trace: each player's components with their full config, every move and the learning from the game, each step with
what went into each component, where it came from and what came out (prompts, answers and calls whole), and the memories as the game started.

Examples:
    uv run plybench-trace results/training/tac_bandit/.../epoch_2/train/game_3.json
    uv run plybench-trace <game.json> --player textrl --sections moves learning --max-chars 200
    uv run plybench-trace <game.json> --json > game_3.trace.json
    uv run plybench-trace results/training/tac_bandit/.../epoch_2/train --steps    # every game of the folder as played, as JSON
    uv run plybench-trace results/training/tac_bandit --list                        # the saved games under a folder, as JSON

The games are read from plybench's store (PLYBENCH_STORE): by default files, found from anywhere by the path's `results` folder; in another store by
their key (the path from `results` on).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.common.store import STORE_ENV, FileStore, store, use_store
from plybench.trace.load import game_files, load_game
from plybench.trace.text import SECTIONS, render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game", type=Path, help="a saved game_<N>.json, or with --steps a folder of them")
    parser.add_argument("--player", nargs="+", help="the players to show, by seat (i, o) or key (e.g. textrl); default both")
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS), help="what to print per player (default all)")
    parser.add_argument("--max-chars", type=int, help="cut every text to one line of this length (default: whole texts)")
    parser.add_argument("--all-entries", action="store_true", help="memory: print the text of the entries the game did not touch too")
    parser.add_argument("--json", action="store_true", help="print the trace as JSON instead")
    parser.add_argument("--steps", action="store_true", help="print the game, or every game of the folder, as played (moves, end, params) as JSON, without the players")
    parser.add_argument("--list", action="store_true", help="print the keys of the saved games under a folder (and its subfolders) as JSON; none when there is no such folder")
    args = parser.parse_args()
    root, game = locate(args.game)
    if not os.environ.get(STORE_ENV):
        use_store(FileStore(root))
    if args.list:
        print(json.dumps({"games": [path.as_posix() for path in game_files(game, nested=True)]}))
        return
    if not store().keys(game.as_posix()):
        parser.error(f"{args.game} does not exist")

    # reading needs no provider; the plugins' players (and their readers) come with PlyBench
    registry = PlyBench(LLMConfig()).registry
    if args.steps:
        files = [game] if store().exists(game.as_posix()) else game_files(game)
        print(json.dumps({"games": [load_game(file, registry, players=False).to_dict() for file in files]}, ensure_ascii=False, indent=1))
        return
    trace = load_game(game, registry)
    if args.json:
        print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=1))
        return
    print(render(trace, args.player, tuple(args.sections), args.max_chars, args.all_entries))


def locate(path: Path) -> tuple[Path, Path]:
    # the folder that holds a game's (or a folder's) results, and its key under it (from `results` on, as runs record them), so a path given from
    # elsewhere ("../plydata/plybench/results/...") reads as one under the working directory does
    if not path.is_absolute() and ".." not in path.parts:
        return Path("."), path
    if "results" not in path.parts:
        raise SystemExit(f"{path} is not under a results folder")
    index = path.parts.index("results")
    return Path(*path.parts[:index]), Path(*path.parts[index:])


if __name__ == "__main__":
    main()
