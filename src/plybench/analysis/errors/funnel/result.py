"""The (game, player) cell every stage reads: the recorded moves, pooled over opponents, and the slices
each later stage runs on. Stage 1 produces this and stages 2-4 consume it, so it holds the moves and their
routing only -- the funnel's own numbers about them are `funnel/stats.py`, like every other stage."""

from __future__ import annotations

from plybench.analysis.errors.moves import FunnelStage, TracedMove, by_stage
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig


class FunnelResult:
    def __init__(self, experiment: str, game: GameConfig, player: PlayerConfig, moves: list[TracedMove]) -> None:
        self.experiment = experiment
        self.game = game
        self.player = player
        self.moves = moves
        grouped = by_stage(moves)
        # every leaf is kept, empty ones included: a bucket nothing landed in is a result, not a missing row
        self._by_stage: dict[FunnelStage, list[TracedMove]] = {stage: grouped.get(stage, []) for stage in FunnelStage}

    def stage(self, *stages: FunnelStage) -> list[TracedMove]:
        return [move for stage in stages for move in self._by_stage[stage]]

    @property
    def analyzable(self) -> list[TracedMove]:
        # every move that has a trace and produced a legal move. Forced positions are in: a trace can be
        # wrong, or contradict itself, wherever it was written, and whether the position offered a real
        # choice is a separate axis the reports cross-tabulate against
        # todo: should we use fails as well?
        return self.stage(FunnelStage.OPTIMAL, FunnelStage.SUBOPTIMAL, FunnelStage.NON_DECISION)
