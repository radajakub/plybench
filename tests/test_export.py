"""The website export translates a finished benchmark into the plybench-website ingestion layout:
a wrapped .tar.gz of experiment.json + results/ (flat LLM step fields) + analysis/ (flat metric keys,
per-round game stats). Mirrors what backend/.../ingestion.service.ts parses."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import tarfile
from pathlib import Path

from plybench.app import PlyBench
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.llm import LLMConfig
from plybench.trackers.game_tracker import GameStep, GameTracker

_spec = importlib.util.spec_from_file_location("_website_export", Path(__file__).parent.parent / "scripts" / "_website_export.py")
assert _spec is not None and _spec.loader is not None
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)

op = PlyBench(LLMConfig())
registry = op.registry

# the flat metric keys the website's matchupMetricsToDto requires (requireCi throws if any is missing)
_REQUIRED_METRIC_KEYS = {
    "n_games",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "fail_rate",
    "player_moves_per_game",
    "input_tokens_per_game",
    "output_tokens_per_game",
    "input_tokens_per_move",
    "output_tokens_per_move",
    "total_input_tokens",
    "total_output_tokens",
}


def test_step_translation_unfolds_data_to_flat_fields():
    i = registry.player_config("llm:actions:structured:openai:gpt-5.4:")
    o = registry.player_config("random:distribution=uniform")
    step = GameStep(
        1,
        i.to_string(),
        i.hash,
        "STATE",
        "OBS",
        "<C2R2>",
        input_tokens=10,
        output_tokens=20,
        reasoning_tokens=5,
        data={"full_output": "Action: <C2R2>", "reasoning_trace": "think", "system_message": "sys", "prompt_message": "prompt"},
    )
    tracker = GameTracker(1, i, o, {}, [step], None, 1, {})

    payload = export.game_tracker_json(tracker)
    out_step = payload["steps"][0]

    assert out_step["full_model_output"] == "Action: <C2R2>"
    assert out_step["reasoning_trace"] == "think"
    assert out_step["system_message"] == "sys" and out_step["prompt_message"] == "prompt"
    assert (out_step["input_tokens"], out_step["output_tokens"]) == (10, 20)
    assert "data" not in out_step and "full_output" not in out_step  # nested form is gone


def _tiny_benchmark(tmp_path, monkeypatch, players, opponents, num_games=2):
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], players, opponents, num_games)
    asyncio.run(benchmark.run(sync=True, concurrency=1))
    return benchmark


def test_analysis_metadata_has_required_flat_keys(tmp_path, monkeypatch):
    benchmark = _tiny_benchmark(tmp_path, monkeypatch, ["optimal:stochastic=True"], ["random:distribution=uniform"])
    tracker = benchmark.get_results().trackers[0]
    from plybench.analysis.stats.compute import compute_matchup_stats

    payload = export.analysis_metadata_json(compute_matchup_stats(tracker, registry))
    combined = payload["metrics"]["combined"]

    assert _REQUIRED_METRIC_KEYS <= set(combined.keys())
    # tic-tac-toe is solvable -> quality present; the optimal player is always optimal
    assert combined["optimality_rate"]["value"] == 1.0 and combined["regret"]["value"] == 0.0
    # tic-tac-toe is recognisable -> the website's optional recognition_rate is filled in
    assert "recognition_rate" in combined
    # renamed to the key the website reads; everything else is passed through under its own name
    assert "moves_per_game" not in combined and "player_moves_per_game" in combined
    assert "score" in combined  # unknown to the website, which ignores it
    assert isinstance(combined["total_input_tokens"], int)


def test_build_archive_matches_ingestion_layout(tmp_path, monkeypatch):
    benchmark = _tiny_benchmark(tmp_path, monkeypatch, ["random:distribution=uniform"], ["random:distribution=normal"], num_games=2)
    out = tmp_path / "exp.tar.gz"
    counts = export.build_archive(op, benchmark, str(out), "exp")
    assert counts == {"matchups": 1, "games": 2, "game_stats": 2}

    with tarfile.open(out) as tar:
        names = tar.getnames()
        payloads = {}
        for name in names:
            member = tar.extractfile(name)
            assert member is not None
            payloads[name] = json.loads(member.read())

    # every path is wrapped in a single top dir the website strips, then a 4-part matchup path
    assert "exp/experiment.json" in names
    assert any(n.startswith("exp/results/benchmarks/exp/tic_tac_toe_2/") and n.endswith("metadata.json") for n in names)
    assert any(n.startswith("exp/analysis/benchmarks/exp/tic_tac_toe_2/") and n.endswith("game_1.json") for n in names)
    for name in names:
        rel = name.split("/", 1)[1]  # strip wrapper, as the website does
        for prefix in ("results/benchmarks/", "analysis/benchmarks/"):
            if rel.startswith(prefix):
                # the website requires exactly <experiment>/<game>/<matchup>/<file> after the prefix
                assert len(rel[len(prefix) :].split("/")) == 4, f"bad matchup path: {rel}"

    # experiment.json carries the fields the website reads
    exp = payloads["exp/experiment.json"]
    assert exp["num_games"] == 2 and exp["baseline"] == "random:distribution=normal"
    assert exp["game_configs"] == [{"value": "tic_tac_toe:", "enabled": True}]

    # a per-round analysis game file carries per-step quality for the solvable game
    game_stats = next(payloads[n] for n in names if "analysis/benchmarks" in n and n.endswith("game_1.json"))
    assert "optimality_rate" in game_stats and game_stats["steps"]
    assert {"seq", "is_trivial", "is_optimal", "regret"} <= set(game_stats["steps"][0].keys())
