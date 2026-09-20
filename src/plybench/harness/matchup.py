from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

from plybench.app import PlyBench
from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.callbacks.matchup_callbacks import MatchupCallbacks
from plybench.common.paths import BenchmarkPathBuilder, ExperimentPathBuilder
from plybench.configs.game_config import GameConfig
from plybench.configs.matchup import Matchup
from plybench.configs.player_config import PlayerConfig
from plybench.player.player import Player
from plybench.player.spec import PlayerBuilder
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.result_tracker import ResultTracker

type ColourRule = Callable[[int, int], bool]
type AfterGame = Callable[[GameTracker], Awaitable[None]]


def first_starts(game_round: int, num_games: int) -> bool:
    return game_round <= (num_games // 2)


def first_starts_by_parity(game_round: int, num_games: int) -> bool:
    # order-independent colour balancing: round 1 is the i-player's, and a phase extended from 20 games
    # to 50 keeps an even split, where the half-split rule would leave 15/35
    return game_round % 2 == 1


def order_players_for_game(i: Player, o: Player, game_round: int, num_games: int, colour_rule: ColourRule = first_starts) -> tuple[Player, Player]:
    # colour balancing: the first half of the rounds the i-player moves first, the second half the o-player does
    return (i, o) if colour_rule(game_round, num_games) else (o, i)


async def play_round(
    op: PlyBench,
    matchup: Matchup,
    game_round: int,
    result_tracker: ResultTracker,
    game_callbacks: GameCallbacks | None,
    matchup_callbacks: MatchupCallbacks,
    player_builder: PlayerBuilder | None = None,
    colour_rule: ColourRule = first_starts,
) -> GameTracker | None:
    if result_tracker.is_game_complete(game_round):
        matchup_callbacks.on_round_complete(matchup.game, matchup.i, matchup.o, game_round)
        return result_tracker.games[game_round - 1]

    matchup_callbacks.on_round_start(matchup.game, matchup.i, matchup.o, game_round)

    engine = op.registry.build_engine(matchup.game)
    # a builder may hand back a shared, stateful player -- but only under `run_matchup_sequential`;
    # concurrent rounds would interleave its per-game scratch, so there it must build a fresh one
    build_player = player_builder if player_builder is not None else op.registry.build_player
    player_i = build_player(engine.game, matchup.i, "i")
    player_o = build_player(engine.game, matchup.o, "o")

    player_pair = order_players_for_game(player_i, player_o, game_round, matchup.num_games, colour_rule)
    round_callbacks = matchup_callbacks.for_round(matchup.game, matchup.i, matchup.o, game_round, game_callbacks)
    game_tracker = await engine.play(player_pair, game_callbacks=round_callbacks, game_round=game_round)

    result_tracker.record_game(game_round, game_tracker)
    matchup_callbacks.on_round_complete(matchup.game, matchup.i, matchup.o, game_round)
    return game_tracker


def _open_matchup(
    op: PlyBench,
    matchup: Matchup,
    matchup_callbacks: MatchupCallbacks | None,
    path_builder: ExperimentPathBuilder | None,
    experiment: str | None,
    save_on_record: bool,
) -> tuple[ResultTracker, MatchupCallbacks]:
    matchup_callbacks = matchup_callbacks if matchup_callbacks is not None else MatchupCallbacks()
    experiment = experiment if experiment is not None else f"run_{uuid.uuid4().hex}"
    path_builder = path_builder if path_builder is not None else BenchmarkPathBuilder()

    result_tracker = ResultTracker.new(
        experiment,
        matchup.i,
        matchup.o,
        matchup.game,
        matchup.num_games,
        op.registry,
        path_builder=path_builder,
        save_on_record=save_on_record,
    )
    result_tracker.load_if_exists()
    matchup_callbacks.on_matchup_start(result_tracker, matchup.game, matchup.i, matchup.o)
    return result_tracker, matchup_callbacks


async def run_matchup_concurrent(
    op: PlyBench,
    matchup: Matchup,
    game_callbacks: GameCallbacks | None = None,
    matchup_callbacks: MatchupCallbacks | None = None,
    path_builder: ExperimentPathBuilder | None = None,
    experiment: str | None = None,
    save_on_record: bool = True,
    player_builder: PlayerBuilder | None = None,
    max_concurrent: int | None = None,
    colour_rule: ColourRule = first_starts,
) -> ResultTracker:
    result_tracker, matchup_callbacks = _open_matchup(op, matchup, matchup_callbacks, path_builder, experiment, save_on_record)

    if result_tracker.is_complete():
        matchup_callbacks.on_matchup_end(result_tracker)
        return result_tracker

    missing_games = result_tracker.get_missing_games()
    semaphore = asyncio.Semaphore(max_concurrent if max_concurrent is not None else len(missing_games))

    async def _run_round(game_round: int) -> None:
        async with semaphore:
            await play_round(op, matchup, game_round, result_tracker, game_callbacks, matchup_callbacks, player_builder, colour_rule)

    await asyncio.gather(*(asyncio.create_task(_run_round(game_round)) for game_round in missing_games))

    matchup_callbacks.on_matchup_end(result_tracker)
    return result_tracker


async def run_matchup_sequential(
    op: PlyBench,
    matchup: Matchup,
    game_callbacks: GameCallbacks | None = None,
    matchup_callbacks: MatchupCallbacks | None = None,
    path_builder: ExperimentPathBuilder | None = None,
    experiment: str | None = None,
    save_on_record: bool = True,
    player_builder: PlayerBuilder | None = None,
    after_game: AfterGame | None = None,
    colour_rule: ColourRule = first_starts,
) -> ResultTracker:
    result_tracker, matchup_callbacks = _open_matchup(op, matchup, matchup_callbacks, path_builder, experiment, save_on_record)

    for game_round in range(1, matchup.num_games + 1):
        game = await play_round(op, matchup, game_round, result_tracker, game_callbacks, matchup_callbacks, player_builder, colour_rule)

        if game is None:
            raise ValueError(f"Round {game_round} of {result_tracker.experiment} is recorded as complete but its record is missing")

        if after_game is not None:
            await after_game(game)

    matchup_callbacks.on_matchup_end(result_tracker)
    return result_tracker


async def single_game(
    op: PlyBench,
    game_config: GameConfig,
    i_config: PlayerConfig,
    o_config: PlayerConfig,
    game_callbacks: GameCallbacks | None = None,
    matchup_callbacks: MatchupCallbacks | None = None,
) -> ResultTracker:
    return await run_matchup_concurrent(
        op,
        Matchup(game_config, i_config, o_config, 1),
        game_callbacks=game_callbacks,
        matchup_callbacks=matchup_callbacks,
        save_on_record=False,
    )
