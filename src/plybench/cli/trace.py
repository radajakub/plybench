"""Print one saved game as a trace: each player's components with their full config, every move and the learning from the game, each step with
what went into each component, where it came from and what came out (prompts, answers and calls whole), and the memories as the game started.

Examples:
    uv run plybench-trace results/training/tac_bandit/.../epoch_2/train/game_3.json
    uv run plybench-trace <game.json> --player textrl --sections moves learning --max-chars 200
    uv run plybench-trace <game.json> --json > game_3.trace.json
    uv run plybench-trace results/training/tac_bandit/.../epoch_2/train --steps    # every game of the folder as played, as JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plyllm import LLMConfig

from plybench.app import PlyBench
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
    args = parser.parse_args()
    if not args.game.exists():
        parser.error(f"{args.game} does not exist")

    # reading needs no provider; the plugins' players (and their readers) come with PlyBench
    registry = PlyBench(LLMConfig()).registry
    if args.steps:
        files = game_files(args.game) if args.game.is_dir() else [args.game]
        print(json.dumps({"games": [load_game(file, registry, players=False).to_dict() for file in files]}, ensure_ascii=False, indent=1))
        return
    trace = load_game(args.game, registry)
    if args.json:
        print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=1))
        return
    print(render(trace, args.player, tuple(args.sections), args.max_chars, args.all_entries))


if __name__ == "__main__":
    main()
