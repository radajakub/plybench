"""Phase 5: the harness (orchestration) layer. `run_matchup` runs/persists/resumes a matchup, the
Benchmark drives the full (player x opponent x game) matrix, both callback levels fire, and an
externally registered agent plays through the registry. Everything here is offline (bots, no providers)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from plybench.app import PlyBench
from plybench.callbacks.benchmark_callbacks import BenchmarkCallbacks
from plybench.callbacks.game_callbacks import GameCallbacks
from plybench.common.paths import BenchmarkPathBuilder
from plybench.configs.benchmark_config import BenchmarkConfig, ToggleItem
from plybench.configs.matchup import Matchup
from plybench.configs.player_params import PlayerParams
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.matchup import run_matchup_concurrent
from plybench.llm import LLMConfig
from plybench.player.player import Player, PlayerOutput
from plybench.player.spec import PlayerSpec

# LLMConfig() has no providers, so the whole run is offline (only bot players)
op = PlyBench(LLMConfig())
registry = op.registry


def test_run_matchup_persists_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # path builders write experiments/results/... under cwd
    game = registry.game_config("tic_tac_toe:")
    i = registry.player_config("random:distribution=uniform")
    o = registry.player_config("random:distribution=normal")

    starts: list[int] = []
    callbacks = BenchmarkCallbacks(round_start_callback=lambda gc, i, o, rnd: starts.append(rnd))

    tracker = asyncio.run(run_matchup_concurrent(op, Matchup(game, i, o, 2), matchup_callbacks=callbacks, experiment="exp", max_concurrent=1))

    assert tracker.is_complete()
    assert sorted(starts) == [1, 2]
    assert tracker.metadata_path.exists()
    assert (tracker.base_path / "game_1.json").exists()
    assert (tracker.base_path / "game_2.json").exists()

    # re-run the identical matchup: already complete, so no round is played again
    starts.clear()
    resumed = asyncio.run(run_matchup_concurrent(op, Matchup(game, i, o, 2), matchup_callbacks=callbacks, experiment="exp", max_concurrent=1))
    assert resumed.is_complete()
    assert starts == []


def test_both_callback_levels_fire(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    events: list[str] = []
    benchmark_callbacks = BenchmarkCallbacks(
        benchmark_start_callback=lambda g, p, b: events.append("benchmark_start"),
        benchmark_end_callback=lambda results: events.append("benchmark_end"),
        matchup_start_callback=lambda *a: events.append("matchup_start"),
        matchup_end_callback=lambda *a: events.append("matchup_end"),
        round_start_callback=lambda *a: events.append("round_start"),
        round_complete_callback=lambda *a: events.append("round_complete"),
        move_complete_callback=lambda *a: events.append("move_complete"),
    )
    moves: list[str] = []
    game_callbacks = GameCallbacks(
        game_start_callback=lambda tracker: moves.append("game_start"),
        before_move_callback=lambda *a: moves.append("before"),
        after_move_callback=lambda *a: moves.append("after"),
        game_end_callback=lambda *a: moves.append("game_end"),
    )

    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["random:distribution=uniform"], ["random:distribution=normal"], 2)
    asyncio.run(benchmark.run(sync=True, concurrency=1, game_callbacks=game_callbacks, benchmark_callbacks=benchmark_callbacks))

    assert events[0] == "benchmark_start" and events[-1] == "benchmark_end"
    assert "matchup_start" in events and "matchup_end" in events
    assert events.count("round_start") == 2 and events.count("round_complete") == 2

    assert moves.count("game_start") == 2 and moves.count("game_end") == 2
    assert moves.count("before") == moves.count("after") > 0
    # move events reach the benchmark level too, without displacing the caller's own game callbacks
    assert events.count("move_complete") == moves.count("after")


def test_load_experiment_and_results_matrix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BenchmarkConfig(
        game_configs=[ToggleItem("tic_tac_toe:", True), ToggleItem("nim:", False)],
        player_configs=[
            ToggleItem("random:distribution=uniform", True),
            ToggleItem("mcts:max_simulations=30,rollout_count=1,uct_c=2.0", True),
        ],
        opponent_configs=[
            ToggleItem("random:distribution=normal", True),
            ToggleItem("random:distribution=uniform", True),
        ],
        num_games=2,
    )
    experiment_path = BenchmarkPathBuilder().experiment_path("my_exp")
    with open(experiment_path, "w") as f:
        json.dump(config.to_dict(), f)

    benchmark = Benchmark.load_experiment(op, "my_exp")
    assert benchmark.game_configs == ["tic_tac_toe:"]  # nim was toggled off
    assert len(benchmark.player_configs) == 2 and len(benchmark.opponent_configs) == 2

    results = asyncio.run(benchmark.run(sync=True, concurrency=1))

    # full matrix: 1 game x 2 players x 2 opponents = 4 matchups
    assert len(results.trackers) == 4
    assert all(tracker.is_complete() for tracker in results.trackers)
    tracker = results.find(
        registry.game_config("tic_tac_toe:"),
        registry.player_config("random:distribution=uniform"),
        registry.player_config("random:distribution=normal"),
    )
    assert tracker.is_complete()
    assert len(results.for_player(registry.player_config("random:distribution=uniform"))) == 2


def test_config_backward_compatible_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # a legacy experiment file that predates the opponents list uses a single `baseline` string
    legacy = {
        "game_configs": [{"value": "tic_tac_toe:", "enabled": True}],
        "player_configs": [{"value": "random:distribution=uniform", "enabled": True}],
        "baseline": "random:distribution=normal",
        "num_games": 2,
    }
    config = BenchmarkConfig.from_dict(legacy)
    assert config.get_opponent_configs() == ["random:distribution=normal"]


def test_axis_overrides_restrict_matrix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BenchmarkConfig(
        game_configs=[ToggleItem("tic_tac_toe:", True)],
        player_configs=[ToggleItem("random:distribution=uniform", True), ToggleItem("random:distribution=normal", True)],
        opponent_configs=[ToggleItem("random:distribution=uniform", True), ToggleItem("random:distribution=normal", True)],
        num_games=2,
    )
    experiment_path = BenchmarkPathBuilder().experiment_path("override_exp")
    with open(experiment_path, "w") as f:
        json.dump(config.to_dict(), f)

    # overrides restrict players to one and opponents to one -> a single matchup
    benchmark = Benchmark.load_experiment(
        op,
        "override_exp",
        player_override=["random:distribution=uniform"],
        opponent_override=["random:distribution=normal"],
    )
    assert benchmark.player_configs == ["random:distribution=uniform"]
    assert benchmark.opponent_configs == ["random:distribution=normal"]

    results = asyncio.run(benchmark.run(sync=True, concurrency=1))
    assert len(results.trackers) == 1


def test_external_agent_plays_through_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @dataclass(frozen=True, eq=True)
    class FirstMoveParams(PlayerParams):
        @classmethod
        def from_string(cls, params_string: str) -> "FirstMoveParams":
            return cls()

        def to_string(self) -> str:
            return ""

        @property
        def path_suffix(self) -> str:
            return ""

    class FirstMovePlayer(Player):
        def initialize_policy(self, game, prompt_adapter_template) -> None:
            pass

        async def __call__(self, game, observation, legal_moves) -> PlayerOutput:
            return PlayerOutput(action=legal_moves[0] if legal_moves else None)

        def format_llm_output(self, player_output: PlayerOutput) -> str:
            return ""

    # a fresh op so this registration does not leak into the module-level registry
    agent_op = PlyBench(LLMConfig())
    agent_op.registry.register_player(PlayerSpec("firstmove", FirstMoveParams, lambda game, cfg, pid: FirstMovePlayer(cfg, pid)))
    assert "firstmove" in agent_op.registry.player_keys()

    game = agent_op.registry.game_config("tic_tac_toe:")
    agent = agent_op.registry.player_config("firstmove:")
    baseline = agent_op.registry.player_config("random:distribution=uniform")

    tracker = asyncio.run(run_matchup_concurrent(agent_op, Matchup(game, agent, baseline, 2), experiment="agent_exp", max_concurrent=1))
    assert tracker.is_complete()
    assert all(not step.move.startswith("FAIL") for game_tracker in tracker.games if game_tracker for step in game_tracker.steps)
