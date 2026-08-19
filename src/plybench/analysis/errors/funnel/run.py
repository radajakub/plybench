from __future__ import annotations

from collections.abc import Iterator

from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.moves import MatchupId, TracedMove
from plybench.analysis.errors.split import SplitConfig
from plybench.analysis.replay import ReplayerCache, TurnBasedReplayer, build_replayer
from plybench.common.progress import track
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.harness.results import BenchmarkResults
from plybench.registry import Registry
from plybench.trackers.result_tracker import ResultTracker


def _trackers_for(results: BenchmarkResults, game: GameConfig, player: PlayerConfig) -> list[ResultTracker]:
    return [tracker for tracker in results.trackers if tracker.game.to_string() == game.to_string() and tracker.i.hash == player.hash]


def _tracker_moves(tracker: ResultTracker, replayer: TurnBasedReplayer) -> list[TracedMove]:
    matchup = MatchupId.from_tracker(tracker)  # built once, then shared by every move of this matchup
    moves: list[TracedMove] = []
    for game_tracker in tracker.games:
        if game_tracker is None:
            continue
        moves.extend(TracedMove.from_replayed(replayed, matchup, tracker.game.key, game_tracker.game_round) for replayed in replayer.replay_steps(game_tracker, tracker.i))
    return moves


def build_funnel(
    results: BenchmarkResults,
    game: GameConfig,
    player: PlayerConfig,
    registry: Registry,
    replayer: TurnBasedReplayer | None = None,
    progress: bool | None = None,
    split_config: SplitConfig | None = None,
) -> FunnelResult:
    if not registry.solvable(game.key):
        raise ValueError(f"Funnel needs the minimax replay, which {game.key} does not support")

    replayer = replayer if replayer is not None else build_replayer(registry, game)
    trackers = _trackers_for(results, game, player)
    experiment = trackers[0].experiment if trackers else ""

    moves: list[TracedMove] = []
    for tracker in track(trackers, f"Funnelling {game.key}", len(trackers), progress):
        moves.extend(_tracker_moves(tracker, replayer))
    return FunnelResult(experiment, game, player, moves, split_config)


def unsupported_games(results: BenchmarkResults, registry: Registry) -> list[str]:
    return [game.to_string() for game in results.game_configs if not registry.solvable(game.key)]


def build_funnels(
    results: BenchmarkResults,
    registry: Registry,
    replayers: ReplayerCache,
    progress: bool | None = None,
    split_config: SplitConfig | None = None,
) -> Iterator[FunnelResult]:
    games = [game for game in results.game_configs if registry.solvable(game.key)]
    cells = [(game, player) for game in games for player in results.player_configs]
    for game, player in track(cells, "Funnelling", len(cells), progress):
        yield build_funnel(results, game, player, registry, replayers(game), progress=False, split_config=split_config)
