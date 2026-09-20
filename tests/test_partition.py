"""The move-level primitives every study composes from: move metrics, move features, and the partition
engine that groups a matchup's judged moves and compares the groups."""

from __future__ import annotations

import asyncio
import json

import pytest

from plybench.analysis.stats.move_features import branching, decision_moves, sharpness
from plybench.analysis.stats.move_metrics import DEFAULT_MOVE_METRICS
from plybench.analysis.stats.moves import MoveRecord
from plybench.analysis.stats.partition import BinaryPartitioner, QuantilePartitioner, by_recognition, compute_partition_stats
from plybench.app import PlyBench
from plybench.common.enums import CIFamily, MetricName, StateClass
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.llm import LLMConfig

op = PlyBench(LLMConfig())


def _move(is_optimal=True, regret=0.0, tokens=None, recognized=None, state_class=StateClass.DECISION, n_legal=0, n_optimal=0):
    return MoveRecord(state_class, is_optimal, regret, None, tokens, recognized, n_legal, n_optimal)


# --- move metrics ------------------------------------------------------------------------------
def test_move_metric_drops_not_applicable_moves():
    metrics = {m.name: m for m in DEFAULT_MOVE_METRICS}
    decision, forced = _move(state_class=StateClass.DECISION, tokens=100), _move(state_class=StateClass.DONT_CARE, tokens=None)
    # opt-nt is undefined off a DECISION state; output-tokens is undefined when the move has no token count
    assert metrics[MetricName.OPTIMALITY_RATE_NON_TRIVIAL].distribution([decision, forced]).n == 1
    assert metrics[MetricName.OUTPUT_TOKENS_PER_MOVE].distribution([decision, forced]).n == 1
    assert metrics[MetricName.OPTIMALITY_RATE].distribution([decision, forced]).n == 2  # defined for every move


def test_move_metric_bundles_and_compares_from_its_family_alone():
    metrics = {m.name: m for m in DEFAULT_MOVE_METRICS}
    rate, regret = metrics[MetricName.OPTIMALITY_RATE], metrics[MetricName.REGRET]
    assert rate.family == CIFamily.RATIO and regret.family == CIFamily.MEAN

    hits, misses = [_move(is_optimal=True)] * 4, [_move(is_optimal=False)] * 4
    assert rate.bundle(hits).wilson is not None and rate.bundle(hits).value == 1.0
    assert rate.compare(hits, misses).test == "two_proportion_z"

    spread = [_move(regret=r) for r in (1.0, 2.0, 3.0)]
    assert regret.bundle(spread).sem is not None
    assert regret.compare(spread, [_move(regret=0.0)] * 3).test == "welch_t"


# --- move features -----------------------------------------------------------------------------
def test_sharpness_grades_difficulty_and_branching_reads_the_action_count():
    trivial = _move(n_legal=4, n_optimal=4)  # every legal move optimal
    sharp = _move(n_legal=9, n_optimal=1)  # one optimal move among nine
    assert sharpness(trivial) == 0.0 and sharpness(sharp) == pytest.approx(8 / 9)
    assert branching(trivial) == 4.0 and branching(sharp) == 9.0
    assert sharpness(_move(n_legal=0, n_optimal=0)) == 0.0  # terminal state: no spread to speak of


def test_decision_moves_keeps_only_judged_moves_carrying_tokens():
    keep = _move(state_class=StateClass.DECISION, tokens=50)
    no_tokens = _move(state_class=StateClass.DECISION, tokens=None)
    forced = _move(state_class=StateClass.DONT_CARE, tokens=50)
    assert decision_moves([keep, no_tokens, forced]) == [keep]


# --- partitioners ------------------------------------------------------------------------------
def test_binary_partitioner_orders_baseline_first_and_drops_undefined_moves():
    groups = by_recognition().partition([_move(recognized=True), _move(recognized=False), _move(recognized=None)])
    # negative (baseline) group first, positive second; the None (trace-less) move is dropped
    assert [g.label for g in groups] == ["not_recognized", "recognized"]
    assert (len(groups[0].moves), len(groups[1].moves)) == (1, 1)


def test_quantile_partitioner_bins_low_to_high_and_covers_every_move():
    moves = [_move(n_legal=9, n_optimal=k) for k in (1, 1, 3, 3, 6, 6, 9, 9)]
    groups = QuantilePartitioner("sharpness", sharpness, 3).partition(moves)

    assert len(groups) > 1 and sum(len(g.moves) for g in groups) == len(moves)
    # the baseline (first) bin holds the least sharp moves, so comparisons read as harder-minus-easiest
    means = [sum(sharpness(m) for m in g.moves) / len(g.moves) for g in groups]
    assert means == sorted(means)


def test_quantile_partitioner_handles_degenerate_and_empty_input():
    flat = QuantilePartitioner("sharpness", sharpness, 3).partition([_move(n_legal=4, n_optimal=2)] * 5)
    assert len(flat) == 1 and len(flat[0].moves) == 5  # a single distinct value -> one bin
    assert QuantilePartitioner("sharpness", sharpness, 3).partition([]) == []


# --- the partition engine, gated through the real replay path ----------------------------------
def test_general_partition_engine_splits_by_an_arbitrary_predicate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # random player makes both optimal and suboptimal moves over a few games
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["random:distribution=uniform"], ["random:distribution=normal"], 6)
    results = asyncio.run(benchmark.run(sync=True, concurrency=1))

    by_optimality = BinaryPartitioner("optimality", "optimal", "suboptimal", lambda m: m.is_optimal)
    stats = compute_partition_stats(results.trackers[0], op.registry, by_optimality)
    assert stats is not None

    # baseline (negative) group first, and the comparison is keyed by the non-baseline group's label
    assert [g.label for g in stats.groups] == ["suboptimal", "optimal"]
    assert set(stats.comparisons.keys()) == {"optimal"}

    optimal = next(g for g in stats.groups if g.label == "optimal")
    # optimal moves are optimal by construction and carry zero regret
    assert optimal.metrics[MetricName.OPTIMALITY_RATE].value == 1.0
    assert optimal.metrics[MetricName.REGRET].value == 0.0
    json.dumps(stats.to_dict())  # the general partition record serialises to plain JSON


def test_partition_engine_accepts_a_quantile_split_of_the_same_moves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["random:distribution=uniform"], ["random:distribution=normal"], 6)
    results = asyncio.run(benchmark.run(sync=True, concurrency=1))

    stats = compute_partition_stats(results.trackers[0], op.registry, QuantilePartitioner("sharpness", sharpness, 3))
    assert stats is not None and stats.partitioner == "sharpness"
    # every non-baseline bin is compared against the first, and the record still serialises
    assert set(stats.comparisons.keys()) == {g.label for g in stats.groups[1:]}
    json.dumps(stats.to_dict())
