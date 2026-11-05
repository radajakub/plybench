"""Play a single game interactively in the terminal — any mix of human and bot/LLM players.

Examples:
    uv run plybench-play                              # human vs optimal on tic-tac-toe
    uv run plybench-play --i human: --o human:        # two humans
    uv run plybench-play --game nim: --i optimal: --o mcts:max_simulations=200,rollout_count=1,uct_c=2.0
Enter moves for a human player using the printed move strings (e.g. C2R2).
"""

from __future__ import annotations

import argparse
import asyncio

from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.cli.shared import build_op
from plybench.common.enums import GameResults
from plybench.configs.matchup import Matchup
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.harness.matchup import check_seats
from plybench.player.player import Player, PlayerOutput
from plybench.player.simple.human_player import HumanPlayer
from plybench.trackers.game_tracker import GameStep, GameTracker


def _console_callbacks() -> GameCallbacks:
    def before(player: Player, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> None:
        print(f"\n=== {player.player_config.to_string()} to move ===")
        if not isinstance(player, HumanPlayer):
            print(observation)

    def after(player: Player, output: PlayerOutput, step: GameStep) -> None:
        if output.reasoning_trace:
            print("<reasoning_trace>")
            print(output.reasoning_trace)
            print("<reasoning_trace/>")
        print(f"  -> {step.move}")

    def end(tracker: GameTracker, results: tuple[GameResults, GameResults]) -> None:
        assert tracker.ending is not None
        print("\n=== game over ===")
        print(tracker.ending.observation)
        print(f"i-player result: {results[0].value} ({results[0].name})")

    return GameCallbacks(before_move_callback=before, after_move_callback=after, game_end_callback=end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", default="tic_tac_toe:", help="game config string (default tic_tac_toe:)")
    parser.add_argument("--i", default="human:", help="first-player config (default human:)")
    parser.add_argument("--o", default="optimal:stochastic=True", help="second-player config (default optimal:stochastic=True)")
    args = parser.parse_args()

    op = build_op()
    game_config, i_config, o_config = op.registry.game_config(args.game), op.registry.player_config(args.i), op.registry.player_config(args.o)
    engine = op.registry.build_engine(game_config)
    # the seats the harness checks: a single-player game needs `--o none:`
    try:
        check_seats(engine.game, Matchup(game_config, i_config, o_config, 1))
    except ValueError as error:
        parser.error(str(error))
    players = (
        op.registry.build_player(engine.game, i_config, "i"),
        op.registry.build_player(engine.game, o_config, "o"),
    )

    asyncio.run(engine.play(players, game_callbacks=_console_callbacks()))


if __name__ == "__main__":
    main()
