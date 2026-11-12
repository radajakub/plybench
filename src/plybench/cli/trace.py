"""Print one saved game as a trace: each player's components with their full config, every move and the learning from the game, each step with
what went into each component, where it came from and what came out (prompts, answers and calls whole), and the memories as the game started.

Examples:
    uv run plybench-trace results/training/tac_bandit/.../epoch_2/train/game_3.json
    uv run plybench-trace <game.json> --player tac --sections moves learning --max-chars 200
    uv run plybench-trace <game.json> --json > game_3.trace.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.trace.load import load_game
from plybench.trace.text import SECTIONS, render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game", type=Path, help="a saved game_<N>.json")
    parser.add_argument("--player", nargs="+", help="the players to show, by seat (i, o) or key (e.g. tac); default both")
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS), help="what to print per player (default all)")
    parser.add_argument("--max-chars", type=int, help="cut every text to one line of this length (default: whole texts)")
    parser.add_argument("--all-entries", action="store_true", help="memory: print the text of the entries the game did not touch too")
    parser.add_argument("--json", action="store_true", help="print the trace as JSON instead")
    args = parser.parse_args()
    if not args.game.exists():
        parser.error(f"{args.game} does not exist")

    # reading needs no provider; the plugins' players (and their readers) come with PlyBench
    trace = load_game(args.game, PlyBench(LLMConfig()).registry)
    if args.json:
        print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=1))
        return
    print(render(trace, args.player, tuple(args.sections), args.max_chars, args.all_entries))


if __name__ == "__main__":
    main()
