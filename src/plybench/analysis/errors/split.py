"""Stable discovery/evaluation partitioning for reasoning-error analysis.

The unit of assignment is a complete recorded game.  Keeping all of a game's moves together prevents
near-identical adjacent positions from leaking from taxonomy induction into evaluation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from plybench.analysis.errors.moves import TracedMove


class AnalysisSplit(StrEnum):
    DISCOVERY = "discovery"
    EVALUATION = "evaluation"


DEFAULT_EVALUATION_FRACTION = 0.8
DEFAULT_SPLIT_SEED = "reasoning-evaluation-v1"


@dataclass(frozen=True, slots=True)
class SplitConfig:
    discovery_fraction: float = 0.2
    seed: str = DEFAULT_SPLIT_SEED

    @property
    def evaluation_fraction(self) -> float:
        return 1.0 - self.discovery_fraction

    def __post_init__(self) -> None:
        if not 0.0 < self.discovery_fraction < 1.0:
            raise ValueError("discovery_fraction must be strictly between 0 and 1")


def split_for(
    move: TracedMove,
    evaluation_fraction: float = DEFAULT_EVALUATION_FRACTION,
    seed: str = DEFAULT_SPLIT_SEED,
    config: SplitConfig | None = None,
) -> AnalysisSplit:
    if config is not None:
        evaluation_fraction, seed = config.evaluation_fraction, config.seed
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be strictly between 0 and 1")
    matchup = move.matchup
    game_id = "\x00".join((seed, matchup.experiment, matchup.game, matchup.player, matchup.opponent, str(move.game_round)))
    # Use 64 deterministic bits and a continuous threshold so changing corpus order cannot move a game.
    draw = int.from_bytes(hashlib.sha256(game_id.encode()).digest()[:8], "big") / 2**64
    return AnalysisSplit.EVALUATION if draw < evaluation_fraction else AnalysisSplit.DISCOVERY


def in_split(
    moves: list[TracedMove],
    split: AnalysisSplit,
    evaluation_fraction: float = DEFAULT_EVALUATION_FRACTION,
    seed: str = DEFAULT_SPLIT_SEED,
    config: SplitConfig | None = None,
) -> list[TracedMove]:
    return [move for move in moves if split_for(move, evaluation_fraction, seed, config) == split]
