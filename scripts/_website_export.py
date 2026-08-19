"""Translate a finished benchmark (new plybench schema) into the .tar.gz layout the plybench-website
ingestion endpoint expects, and pack it.

The website mirrors the ORIGINAL plybench `to_dict()` shapes, which differ from this refactored package
in three places — handled here so the website stays untouched:
  1. GameStep: the site reads FLAT `full_model_output` / `reasoning_trace` / `system_message` /
     `prompt_message`; the new schema nests them under `data` (with `full_output`). We unfold them.
  2. analysis metadata: the site wants each split as a flat object of metric keys (plus
     `total_input_tokens` / `total_output_tokens`); the new schema keeps a generic `metrics` map. We
     flatten it and rename `moves_per_game` to `player_moves_per_game`.
  3. experiment.json: the site reads a single `baseline` string (informational); we join the opponents.

Archive layout (single wrapper dir, which the site strips):
    <name>/experiment.json
    <name>/results/benchmarks/<experiment>/<game>_<n>/<i>_<o>/{metadata.json, game_<r>.json}
    <name>/analysis/benchmarks/<experiment>/<game>_<n>/<i>_<o>/{metadata.json, game_<r>.json}
"""

from __future__ import annotations

import io
import json
import tarfile
from typing import Any

from plybench.analysis.replay import TurnBasedReplayer, build_replayer
from plybench.analysis.statistics.bundle import mean_bundle, ratio_bundle
from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.stats.compute import compute_matchup_stats
from plybench.analysis.stats.matchup_stats import MatchupMetrics, MatchupStats
from plybench.app import PlyBench
from plybench.common.enums import GameResults, MetricName, StateClass
from plybench.configs.player_config import PlayerConfig
from plybench.harness.benchmark import Benchmark
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.result_tracker import ResultTracker

# Every metric is exported under its own enum value, so a newly added metric reaches the website
# without touching this file; the website ignores keys it does not know and treats everything beyond
# the required set as optional. Only genuine name mismatches are listed here.
_METRIC_KEY_OVERRIDES = {MetricName.MOVES_PER_GAME: "player_moves_per_game"}


def experiment_json(benchmark: Benchmark) -> dict[str, Any]:
    return {
        "game_configs": [{"value": game, "enabled": True} for game in benchmark.game_configs],
        "player_configs": [{"value": player, "enabled": True} for player in benchmark.player_configs],
        "baseline": ", ".join(benchmark.opponent_configs),
        "num_games": benchmark.num_games,
    }


def _step_json(step: dict[str, Any]) -> dict[str, Any]:
    data = step.get("data") or {}
    out: dict[str, Any] = {key: step[key] for key in ("seq", "player_name", "player_hash", "serialized_state", "observation", "move")}
    if step.get("input_tokens") is not None:
        out["input_tokens"] = step["input_tokens"]
    if step.get("output_tokens") is not None:
        out["output_tokens"] = step["output_tokens"]
    # unfold the nested LLM extras into the flat fields the website reads
    for data_key, flat_key in (
        ("full_output", "full_model_output"),
        ("system_message", "system_message"),
        ("prompt_message", "prompt_message"),
        ("reasoning_trace", "reasoning_trace"),
    ):
        if data.get(data_key):
            out[flat_key] = data[data_key]
    return out


def game_tracker_json(game: GameTracker) -> dict[str, Any]:
    payload = game.to_dict()
    payload["steps"] = [_step_json(step) for step in payload["steps"]]
    return payload


def _total_tokens(bundle_per_game) -> int:
    # per_game bundle value is the mean over `n` games, so mean * n is the integer token total
    return round(bundle_per_game.value * bundle_per_game.n)


def _metrics_json(metrics: MatchupMetrics) -> dict[str, Any]:
    by_name = metrics.metrics
    out: dict[str, Any] = {"n_games": metrics.n_games}
    for name, bundle in by_name.items():
        out[_METRIC_KEY_OVERRIDES.get(name, name.value)] = bundle.to_dict()
    out["total_input_tokens"] = _total_tokens(by_name[MetricName.INPUT_TOKENS_PER_GAME])
    out["total_output_tokens"] = _total_tokens(by_name[MetricName.OUTPUT_TOKENS_PER_GAME])
    return out


def analysis_metadata_json(stats: MatchupStats) -> dict[str, Any]:
    return {
        "experiment": stats.experiment,
        "i_config": stats.i.to_string(),
        "o_config": stats.o.to_string(),
        "game_config": stats.game.to_string(),
        "n_games": stats.n_games,
        "completed": stats.completed,
        "metrics": {
            "combined": _metrics_json(stats.metrics.combined),
            "i_first": _metrics_json(stats.metrics.i_first),
            "i_second": _metrics_json(stats.metrics.i_second),
        },
    }


def game_stats_json(game: GameTracker, player: PlayerConfig, replayer: TurnBasedReplayer | None) -> dict[str, Any]:
    player_steps = [step for step in game.steps if step.player_hash == player.hash]
    input_dist = Distribution([step.input_tokens for step in player_steps if step.input_tokens is not None])
    output_dist = Distribution([step.output_tokens for step in player_steps if step.output_tokens is not None])

    result = game.get_result(player).value if game.ending is not None else GameResults.DRAW.value
    out: dict[str, Any] = {
        "game_round": game.game_round,
        "result": result,
        "total_moves": len(player_steps),
        "total_input_tokens": int(input_dist.total),
        "total_output_tokens": int(output_dist.total),
        "input_tokens_per_move": mean_bundle(input_dist).to_dict(),
        "output_tokens_per_move": mean_bundle(output_dist).to_dict(),
        "steps": [],
    }

    if replayer is None:
        out["steps"] = [_raw_step_stats(step) for step in player_steps]
        return out

    step_stats = replayer.replay_stats(game, player)
    optimality = Distribution([1.0 if s.is_optimal else 0.0 for s in step_stats])
    non_trivial = Distribution([1.0 if s.is_optimal else 0.0 for s in step_stats if s.state_class == StateClass.DECISION])
    regret = Distribution([s.regret for s in step_stats])
    out["optimality_rate"] = ratio_bundle(optimality).to_dict()
    out["optimality_rate_non_trivial"] = ratio_bundle(non_trivial).to_dict()
    out["regret"] = mean_bundle(regret).to_dict()
    out["steps"] = [
        {"seq": s.seq, "input_tokens": s.input_tokens, "output_tokens": s.output_tokens, "is_trivial": s.is_trivial, "is_optimal": s.is_optimal, "regret": s.regret}
        for s in step_stats
    ]
    return out


def _raw_step_stats(step) -> dict[str, Any]:
    out: dict[str, Any] = {"seq": step.seq}
    if step.input_tokens is not None:
        out["input_tokens"] = step.input_tokens
    if step.output_tokens is not None:
        out["output_tokens"] = step.output_tokens
    return out


def _relative_dir(kind: str, tracker: ResultTracker) -> str:
    game_dir = f"{tracker.game.path}_{tracker.n}"
    matchup_dir = f"{tracker.i.path}_{tracker.o.path}"
    return f"{kind}/benchmarks/{tracker.experiment}/{game_dir}/{matchup_dir}"


def build_archive(op: PlyBench, benchmark: Benchmark, out_path: str, name: str, confidence: float = 0.95, include_fails: bool = False) -> dict[str, int]:
    results = benchmark.get_results()
    counts = {"matchups": 0, "games": 0, "game_stats": 0}

    with tarfile.open(out_path, "w:gz") as tar:
        _add_json(tar, f"{name}/experiment.json", experiment_json(benchmark))

        for tracker in results.trackers:
            games = [game for game in tracker.games if game is not None]
            if not games:
                continue
            counts["matchups"] += 1

            results_dir = _relative_dir("results", tracker)
            _add_json(tar, f"{name}/{results_dir}/metadata.json", tracker.to_dict())
            for game in games:
                _add_json(tar, f"{name}/{results_dir}/game_{game.game_round}.json", game_tracker_json(game))
                counts["games"] += 1

            analysis_dir = _relative_dir("analysis", tracker)
            stats = compute_matchup_stats(tracker, op.registry, confidence, include_fails)
            _add_json(tar, f"{name}/{analysis_dir}/metadata.json", analysis_metadata_json(stats))

            replayer = build_replayer(op.registry, tracker.game) if op.registry.solvable(tracker.game.key) else None
            for game in games:
                _add_json(tar, f"{name}/{analysis_dir}/game_{game.game_round}.json", game_stats_json(game, tracker.i, replayer))
                counts["game_stats"] += 1

    return counts


def _add_json(tar: tarfile.TarFile, path: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    info = tarfile.TarInfo(path)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
