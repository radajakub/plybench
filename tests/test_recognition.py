"""Recognition analysis: detecting the underlying game name in a reasoning trace, the per-matchup
recognition rate, and the recognised/not-recognised move split."""

from __future__ import annotations

import asyncio

from plybench.analysis import BenchmarkAnalysis
from plybench.analysis.extractors.recognition import recognition_extractors
from plybench.analysis.recognition import original_game_name, recognizable, step_recognized, trace_mentions_original_game
from plybench.analysis.stats.partition import compute_recognition_split
from plybench.app import PlyBench
from plybench.common.enums import MetricName
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.llm import LLMConfig
from plybench.trackers.game_tracker import GameStep, GameTracker

op = PlyBench(LLMConfig())
registry = op.registry


def _run_benchmark(tmp_path, monkeypatch, num_games=2):
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["random:distribution=uniform"], ["random:distribution=normal"], num_games)
    return asyncio.run(benchmark.run(sync=True, concurrency=1))


# --- recognising the underlying game in a reasoning trace ------------------------------------
def test_obfuscated_variants_map_back_to_their_original_game():
    assert original_game_name("magic_square") == "tic tac toe"
    assert original_game_name("inverse_nim") == "nim"
    assert recognizable("connect_four") and not recognizable("some_unknown_game")


def test_trace_detection_matches_aliases_on_word_boundaries():
    # aliases with any separator, case-insensitive
    assert trace_mentions_original_game("Honestly this is just Tic-Tac-Toe.", "magic_square")
    assert trace_mentions_original_game("the classic game of NIM", "story_nim")
    assert trace_mentions_original_game("looks like connect 4 to me", "connect_four")
    # a bare mention that is only a substring must not match (nim inside "minimum")
    assert not trace_mentions_original_game("we minimise the minimum loss", "story_nim")
    # empty trace and unrecognisable game are both negative
    assert not trace_mentions_original_game("", "magic_square")
    assert not trace_mentions_original_game("clearly tic tac toe", "some_unknown_game")


def test_step_recognised_reads_the_reasoning_trace_from_step_data():
    with_trace = GameStep(0, "p", "h", "", "obs", "a1", data={"reasoning_trace": "this is nim"})
    without_trace = GameStep(1, "p", "h", "", "obs", "a1", data={})
    assert step_recognized(with_trace, "modified_nim")
    assert not step_recognized(without_trace, "modified_nim")


# --- the recognition rate as a matchup metric ------------------------------------------------
def _game_with_traces(i, o, traces):
    steps = [GameStep(seq, i.to_string(), i.hash, "", "obs", "a1", data={"reasoning_trace": trace} if trace else {}) for seq, trace in enumerate(traces)]
    # one opponent move, which must never enter the player's sample even though it names the game
    steps.append(GameStep(len(steps), o.to_string(), o.hash, "", "obs", "a1", data={"reasoning_trace": "tic tac toe"}))
    return GameTracker(1, i, o, {}, steps, None, 1, {})


def test_recognition_rate_counts_only_the_players_reasoning_bearing_moves():
    i, o = registry.player_config("llm:actions:structured:openai:gpt-5.4:"), registry.player_config("random:distribution=uniform")
    game = _game_with_traces(i, o, ["this is just tic-tac-toe", "let me take the centre", None])

    (extractor,) = recognition_extractors("magic_square")
    distribution = extractor.extract([game], i)

    # the trace-less move is excluded rather than counted as a miss, and the opponent's move is ignored
    assert distribution.n == 2 and distribution.mean == 0.5


def test_recognition_extractor_is_absent_for_unrecognisable_games():
    assert recognition_extractors("some_unknown_game") == []


def test_recognition_rate_is_a_first_class_matchup_metric(tmp_path, monkeypatch):
    results = _run_benchmark(tmp_path, monkeypatch, num_games=2)  # tic_tac_toe is recognisable
    # present even without a registry (recognition needs no minimax), with a Wilson interval like any ratio
    metrics = (
        BenchmarkAnalysis(results)
        .matchup(
            registry.game_config("tic_tac_toe:"),
            registry.player_config("random:distribution=uniform"),
            registry.player_config("random:distribution=normal"),
        )
        .metrics.combined.metrics
    )
    assert MetricName.RECOGNITION_RATE in metrics and metrics[MetricName.RECOGNITION_RATE].wilson is not None


# --- the per-matchup split, gated through the real replay path -------------------------------
def test_split_is_dropped_when_no_reasoning_traces_exist(tmp_path, monkeypatch):
    # bots on tic_tac_toe: recognisable + solvable, but no traces to judge
    results = _run_benchmark(tmp_path, monkeypatch, num_games=2)

    analysis = BenchmarkAnalysis(results, op.registry)
    assert analysis.analyze_recognition() == []
    assert compute_recognition_split(results.trackers[0], op.registry) is None
    # and without a registry the split cannot be computed at all
    assert BenchmarkAnalysis(results).analyze_recognition() == []
