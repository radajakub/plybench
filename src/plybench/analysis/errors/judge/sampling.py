from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from plybench.analysis.errors.moves import FunnelStage, TracedMove

Stratum = tuple[str, ...]
StratumKey = Callable[[TracedMove], Stratum]


def by_game_player_outcome(move: TracedMove) -> Stratum:
    """The default strata. Suboptimal moves are rare and concentrated in a few models, so sampling them
    uniformly would produce a codebook describing whichever model fails most, not the population."""
    return (move.matchup.game, move.matchup.player, move.decision.value)


def by_matchup_outcome(move: TracedMove) -> Stratum:
    """The discovery strata. Opponents alter which positions a player reaches, so they belong in
    taxonomy discovery even though prevalence is estimated from the natural, unstratified population."""
    matchup = move.matchup
    return (matchup.game, matchup.player, matchup.opponent, move.decision.value)


def position_of(move: TracedMove) -> tuple[str, ...]:
    """The cell and the board the model was looking at. `serialized_state` is the solver's own
    representation and is the right key; the rendered observation stands in when a run predates it."""
    matchup = move.matchup
    return (matchup.experiment, matchup.game, matchup.player, matchup.opponent, move.serialized_state or move.observation)


def cap_per_position(moves: Sequence[TracedMove], cap: int, seed: str = "") -> list[TracedMove]:
    """At most `cap` moves from any one (cell, position), in the order they came in.

    60% of traced moves sit in a position their own model reached more than once in the same
    presentation, so an uncapped discovery sample spends much of its budget re-reading the same board.
    This is a sampling parameter and deliberately not a filtering stage: it changes which moves are drawn,
    never which moves exist, and it introduces no similarity threshold -- positions are equal or they are
    not. Exact duplicate traces remain a descriptive result worth reporting, not something to act on."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    groups: dict[tuple[str, ...], list[TracedMove]] = {}
    for move in moves:
        groups.setdefault(position_of(move), []).append(move)
    kept = {move.uid for group in groups.values() for move in sorted(group, key=lambda move: _rank(move, seed))[:cap]}
    return [move for move in moves if move.uid in kept]


@dataclass(frozen=True, slots=True)
class DiscoveryPlan:
    """A diversity sample and the prefix that must be read before saturation is allowed."""

    moves: tuple[TracedMove, ...]
    coverage_moves: int
    candidate_strata: int
    n_candidates: int = 0  # analysable moves offered to the plan
    n_after_position_cap: int = 0  # what survived the per-position cap, before the outcome caps


def _discovery_cap(move: TracedMove, suboptimal_cap: int, optimal_cap: int, non_decision_cap: int) -> int:
    return {
        FunnelStage.SUBOPTIMAL: suboptimal_cap,
        FunnelStage.OPTIMAL: optimal_cap,
        FunnelStage.NON_DECISION: non_decision_cap,
    }[move.decision]


def discovery_plan(
    moves: Sequence[TracedMove],
    *,
    suboptimal_cap: int = 32,
    optimal_cap: int = 4,
    non_decision_cap: int = 2,
    coverage_per_stratum: int = 2,
    state_cap: int = 2,
    seed: str = "",
) -> DiscoveryPlan:
    """Build a mistake-enriched, deterministic taxonomy-discovery sequence.

    Every small stratum is retained whole. Large suboptimal strata receive much more capacity than
    controls, preventing weak players from swamping the taxonomy while preserving rare failures from
    strong players. The first ``coverage_per_stratum`` selected moves from every represented matchup and
    outcome form a coverage prefix; the induction loop may only start its saturation countdown after it.

    ``state_cap`` bounds how many moves may come from any one (cell, position) before any of that happens.
    """
    if suboptimal_cap <= 0:
        raise ValueError("suboptimal_cap must be positive")
    if optimal_cap < 0 or non_decision_cap < 0:
        raise ValueError("control caps cannot be negative")
    if coverage_per_stratum <= 0:
        raise ValueError("coverage_per_stratum must be positive")

    # capped before stratification, so a stratum whose positions repeat yields fewer moves rather than the
    # same number of near-identical ones
    n_candidates = len(moves)
    moves = cap_per_position(moves, state_cap, seed)

    groups: dict[Stratum, list[TracedMove]] = {}
    for move in moves:
        groups.setdefault(by_matchup_outcome(move), []).append(move)

    selected: dict[Stratum, list[TracedMove]] = {}
    for stratum, group in groups.items():
        cap = _discovery_cap(group[0], suboptimal_cap, optimal_cap, non_decision_cap)
        if cap:
            selected[stratum] = sorted(group, key=lambda move: _rank(move, seed))[:cap]

    # A hashed stratum order prevents game names from deciding which domains share early batches.
    stratum_order = sorted(selected, key=lambda stratum: hashlib.sha256(f"{seed}|{stratum}".encode()).hexdigest())
    prefix = [selected[stratum][index] for index in range(coverage_per_stratum) for stratum in stratum_order if index < len(selected[stratum])]
    covered = {move.uid for move in prefix}
    remainder = interleave(
        [move for group in selected.values() for move in group if move.uid not in covered],
        key=by_matchup_outcome,
        seed=seed,
    )
    return DiscoveryPlan(tuple((*prefix, *remainder)), len(prefix), len(groups), n_candidates, len(moves))


def _rank(move: TracedMove, seed: str) -> str:
    # ordering by a hash of the move's own id rather than by an RNG: the same seed selects the same moves
    # on any machine and in any order, so a re-run extends an annotation set instead of replacing it
    return hashlib.sha256(f"{seed}|{move.uid}".encode()).hexdigest()


def sample(moves: Sequence[TracedMove], per_stratum: int, seed: str = "", key: StratumKey = by_game_player_outcome) -> list[TracedMove]:
    """Up to `per_stratum` moves from each stratum, chosen deterministically. Strata smaller than the cap
    are taken whole, which is what makes the rare buckets survive the sample."""
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")

    groups: dict[Stratum, list[TracedMove]] = {}
    for move in moves:
        groups.setdefault(key(move), []).append(move)

    sampled: list[TracedMove] = []
    for stratum in sorted(groups):
        sampled.extend(sorted(groups[stratum], key=lambda move: _rank(move, seed))[:per_stratum])
    return sampled


def interleave(moves: Sequence[TracedMove], key: StratumKey = by_game_player_outcome, seed: str = "") -> list[TracedMove]:
    """Spread every stratum evenly across the whole sequence. Induction consumes contiguous slices of this
    list as batches and stops once `patience` of them propose nothing new, so a list that runs cell by cell
    -- or, after sampling, all the optimal moves before any suboptimal one -- makes it declare saturation
    on a prefix. Placing each move at its fractional rank within its own stratum, rather than round-robin,
    keeps the spread even when the strata differ wildly in size.

    The rank alone is not enough. Every stratum contributes about one move per rank, so whatever breaks
    the tie decides what a batch is made of -- and ordering by the stratum itself sorts by the game config
    string, which put twelve consecutive all-magic_square batches at the front of a live run and induced
    the whole codebook off one presentation. The tiebreak is therefore a hash of the stratum: still fully
    deterministic given the seed, but with no relationship to how a game happens to be named."""
    groups: dict[Stratum, list[TracedMove]] = {}
    for move in moves:
        groups.setdefault(key(move), []).append(move)

    shuffled = {stratum: hashlib.sha256(f"{seed}|{stratum}".encode()).hexdigest() for stratum in groups}
    ranked = [(index / len(groups[stratum]), shuffled[stratum], move) for stratum in sorted(groups) for index, move in enumerate(groups[stratum])]
    return [move for _, _, move in sorted(ranked, key=lambda row: row[:2])]


def strata(moves: Sequence[TracedMove], key: StratumKey = by_game_player_outcome) -> dict[Stratum, int]:
    counts: dict[Stratum, int] = {}
    for move in moves:
        counts[key(move)] = counts.get(key(move), 0) + 1
    return counts
