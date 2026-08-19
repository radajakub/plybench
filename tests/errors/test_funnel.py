"""The reasoning funnel: how a recorded move is routed to exactly one analysis bucket, and the move
identity the (expensive, non-reproducible) LLM annotations are keyed by."""

from __future__ import annotations

import asyncio
import json

from plybench.analysis.errors.format import Scope
from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.funnel.run import build_funnel
from plybench.analysis.errors.funnel.stats import funnel_report, metrics
from plybench.analysis.errors.moves import FunnelStage, MatchupId, TracedMove, group_by
from plybench.analysis.stats.moves import MoveRecord
from plybench.app import PlyBench
from plybench.common.enums import MetricName, StateClass
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.llm import LLMConfig

op = PlyBench(LLMConfig())


def _traced(trace: str | None = "reasoning...", move="<A1>", state_class=StateClass.DECISION, is_optimal=True, regret=0.0, opponent="random:", game_round=1, seq=1):
    record = MoveRecord(state_class, is_optimal, regret, None, 100, None, 3, 1)
    matchup = MatchupId("exp", "tic_tac_toe:", "llm:player", opponent)
    return TracedMove(matchup, game_round, seq, record, trace, "board", move, ("<A1>", "<A2>", "<A3>"), ("<A1>",))


def _funnel(moves):
    return FunnelResult("exp", op.registry.game_config("tic_tac_toe:"), op.registry.player_config("random:distribution=uniform"), moves)


# --- leaf assignment ---------------------------------------------------------------------------
def test_each_move_lands_in_exactly_one_leaf_taken_in_funnel_order():
    # a trace-less move is filtered out first, even though it also failed; a failed move with a trace is
    # filtered next, whatever position it was played in
    assert _traced(trace=None, move="FAIL: illegal").stage == FunnelStage.NON_REASONING
    assert _traced(move="FAIL: illegal", state_class=StateClass.DONT_CARE).stage == FunnelStage.FAILED
    assert _traced(state_class=StateClass.DONT_CARE).stage == FunnelStage.NON_DECISION
    assert _traced(state_class=StateClass.LOST).stage == FunnelStage.NON_DECISION
    assert _traced(is_optimal=True).stage == FunnelStage.OPTIMAL
    assert _traced(is_optimal=False, regret=2.0).stage == FunnelStage.SUBOPTIMAL


def _report(funnel):
    return funnel_report(Scope.of(funnel), funnel.moves)


def test_the_leaves_partition_the_moves_so_proportions_are_a_count_over_them():
    funnel = _funnel([_traced(trace=None), _traced(move="FAIL: x"), _traced(state_class=StateClass.LOST), _traced(), _traced(is_optimal=False)])
    summary = _report(funnel).stages
    assert sum(stage.n_moves for stage in summary.values()) == len(funnel.moves) == 5
    assert sum(stage.share for stage in summary.values()) == 1.0
    assert summary[FunnelStage.SUBOPTIMAL].share == 0.2


def test_analyzable_keeps_the_optimal_moves_as_the_baseline_for_mistake_rates():
    optimal, suboptimal, forced = _traced(is_optimal=True), _traced(is_optimal=False), _traced(state_class=StateClass.LOST)
    funnel = _funnel([optimal, suboptimal, forced, _traced(trace=None), _traced(move="FAIL: x")])
    # right-move-wrong-reasoning is only visible if the optimal decisions are annotated too, and a forced
    # position can still be reasoned about wrongly -- only the two moves with nothing to read are dropped
    assert funnel.analyzable == [optimal, suboptimal, forced]


def test_lost_share_separates_already_lost_positions_from_genuinely_forced_ones():
    # the two are not comparable across players: forced positions come from the game tree, lost ones from
    # the player's own earlier blunders, so a weak player inflates its own non-decision bucket
    assert _report(_funnel([_traced(state_class=StateClass.LOST), _traced(state_class=StateClass.DONT_CARE)])).lost_share == 0.5
    assert _report(_funnel([_traced()])).lost_share is None  # undefined with an empty bucket, not zero


# --- move identity -----------------------------------------------------------------------------
def test_move_uid_is_derived_from_recorded_data_alone():
    # equal coordinates -> equal uid, in any process and at any load order: a rebuilt annotation cache
    # re-keys onto the same moves, and the codebook keeps pointing at what it pointed at
    assert _traced(game_round=3, seq=7).uid == _traced(game_round=3, seq=7, trace="different text", is_optimal=False).uid

    base = _traced(game_round=3, seq=7)
    others = [
        _traced(game_round=4, seq=7),
        _traced(game_round=3, seq=8),
        _traced(game_round=3, seq=7, opponent="mcts:"),  # pooled opponents stay distinguishable
        TracedMove(MatchupId("other", "tic_tac_toe:", "llm:player", "random:"), 3, 7, base.record, None, "", "", (), ()),
        TracedMove(MatchupId("exp", "nim:", "llm:player", "random:"), 3, 7, base.record, None, "", "", (), ()),
        TracedMove(MatchupId("exp", "tic_tac_toe:", "llm:other", "random:"), 3, 7, base.record, None, "", "", (), ()),
    ]
    assert len({base.uid, *(other.uid for other in others)}) == len(others) + 1


# --- slicing and summarising -------------------------------------------------------------------
def test_any_slice_summarises_through_the_shared_move_metrics():
    funnel = _funnel(
        [_traced(is_optimal=True, opponent="random:"), _traced(is_optimal=False, regret=2.0, opponent="mcts:"), _traced(is_optimal=False, regret=1.0, opponent="mcts:")]
    )
    assert {label: len(moves) for label, moves in group_by(funnel.moves, lambda move: move.matchup.opponent).items()} == {"random:": 1, "mcts:": 2}
    assert len([move for move in funnel.moves if move.record.regret > 0]) == 2

    bundles = metrics(funnel.analyzable)
    assert bundles[MetricName.OPTIMALITY_RATE].value == 1 / 3
    assert bundles[MetricName.REGRET].value == 1.0
    json.dumps(_report(funnel).to_dict())


# --- the real load path, through the minimax replay ---------------------------------------------
def test_funnel_pools_opponents_and_walks_each_game_in_recorded_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["random:distribution=uniform"], ["random:distribution=normal", "optimal:stochastic=True"], 4)
    results = asyncio.run(benchmark.run(sync=True, concurrency=1))

    game, player = results.game_configs[0], results.player_configs[0]
    funnel = build_funnel(results, game, player, op.registry, progress=False)

    # both opponents are pooled into one result, each move still carrying which one it was played against
    assert set(group_by(funnel.moves, lambda move: move.matchup.opponent)) == {"random:distribution=normal", "optimal:stochastic=True"}
    assert funnel.experiment == "exp"

    uids = [move.uid for move in funnel.moves]
    assert len(set(uids)) == len(uids)  # every recorded move is addressed exactly once
    assert uids == [move.uid for move in build_funnel(results, game, player, op.registry, progress=False).moves]

    # the matchup coordinates are held once and shared, not re-serialized per move
    assert len({id(move.matchup) for move in funnel.moves}) == 2  # one per opponent

    # a bot carries no trace, so the whole population is filtered out at the first funnel step
    assert len(funnel.stage(FunnelStage.NON_REASONING)) == len(funnel.moves) > 0
