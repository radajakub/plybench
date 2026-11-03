"""Seeding: every game's seed is derived from the experiment's base seed and the game's record path, and fixes the
sampled instance and the draws of each player, the LLM's seeded calls included. Offline: bots and a stub LLM."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from plyllm import LLMConfig, LLMResponse, LLMTokens
from plyllm.response import OutputText

from plybench.app import PlyBench
from plybench.configs.benchmark_config import BenchmarkConfig
from plybench.configs.matchup import Matchup
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.matchup import run_matchup_concurrent
from plybench.player.llm_player import LLMPlayer
from plybench.player.output_strategies import StructuredOutputStrategy
from plybench.utils.seed import SEED_LIMIT, derive_seed

op = PlyBench(LLMConfig())
registry = op.registry

RANDOM = "random:distribution=uniform"
# every bot that draws: a uniform and a normal random player, the eps-optimal player and MCTS
OPPONENTS = ["random:distribution=normal", "optimal:stochastic=True,eps=0.3", "mcts:max_simulations=50,rollout_count=1,uct_c=2.0"]
GAMES = 4


def _games(root: Path, monkeypatch: pytest.MonkeyPatch, seed: int, game: str = "tic_tac_toe:") -> list[dict]:
    # one benchmark in its own directory; the records as written, in a fixed order
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(root)
    benchmark = Benchmark("seeded", op, [game], [RANDOM], OPPONENTS, GAMES, seed=seed)
    asyncio.run(benchmark.run(sync=True, concurrency=1))
    return [json.loads(path.read_text()) for path in sorted(Path("results").rglob("game_*.json"))]


def _moves(records: list[dict]) -> list[list[str]]:
    return [[step["move"] for step in record["steps"]] for record in records]


def test_derive_seed_is_stable_bounded_and_separates_its_parts():
    assert derive_seed(0, "a/game_1.json") == derive_seed(0, "a/game_1.json")
    assert 0 <= derive_seed(0, "a/game_1.json") < SEED_LIMIT
    seeds = {derive_seed(0, "a/game_1.json"), derive_seed(0, "a/game_2.json"), derive_seed(1, "a/game_1.json"), derive_seed(0, "a/game_1.json", "i")}
    assert len(seeds) == 4


def test_the_same_seed_plays_the_same_games(tmp_path, monkeypatch):
    first = _games(tmp_path / "first", monkeypatch, seed=5)
    second = _games(tmp_path / "second", monkeypatch, seed=5)

    assert len(first) == GAMES * len(OPPONENTS)
    assert _moves(first) == _moves(second)
    assert [record["seed"] for record in first] == [record["seed"] for record in second]


def test_every_game_gets_its_own_seed_and_another_base_seed_other_games(tmp_path, monkeypatch):
    five = _games(tmp_path / "five", monkeypatch, seed=5)
    six = _games(tmp_path / "six", monkeypatch, seed=6)

    # rounds and matchups differ in their record path, so in their seed
    assert len({record["seed"] for record in five}) == len(five)
    assert {record["seed"] for record in five}.isdisjoint(record["seed"] for record in six)
    assert _moves(five) != _moves(six)


def test_continuing_a_matchup_under_another_seed_is_refused(tmp_path, monkeypatch):
    _games(tmp_path, monkeypatch, seed=5)

    with pytest.raises(ValueError, match="recorded with seed 5, not 6"):
        asyncio.run(Benchmark("seeded", op, ["tic_tac_toe:"], [RANDOM], OPPONENTS, GAMES, seed=6).run(sync=True))
    # loading the results to analyse them needs no seed
    assert Benchmark("seeded", op, ["tic_tac_toe:"], [RANDOM], OPPONENTS, GAMES).get_results() is not None


def test_a_matchup_recorded_without_a_seed_stays_unseeded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    matchup = Matchup(registry.game_config("tic_tac_toe:"), registry.player_config(RANDOM), registry.player_config(OPPONENTS[0]), GAMES)
    tracker = asyncio.run(run_matchup_concurrent(op, matchup, experiment="legacy", max_concurrent=1, seed=None))
    # keep two rounds, as a run interrupted before seeding existed would have left the matchup
    for game_round in (3, 4):
        tracker.path_builder.game_file(tracker.base_path, game_round).unlink()
    metadata = json.loads(tracker.metadata_path.read_text())
    assert "seed" not in metadata
    tracker.metadata_path.write_text(json.dumps({**metadata, "completed": [1, 2]}))

    with pytest.warns(UserWarning, match="recorded without a seed"):
        resumed = asyncio.run(run_matchup_concurrent(op, matchup, experiment="legacy", max_concurrent=1, seed=5))

    assert resumed.is_complete()
    assert all(game is not None and game.seed is None for game in resumed.games)
    assert "seed" not in json.loads(resumed.metadata_path.read_text())


def test_only_a_set_config_seed_seeds_the_experiment():
    data = {"game_configs": [], "player_configs": [], "opponents": [], "num_games": 2}
    assert BenchmarkConfig.from_dict({**data, "seed": 7}).seed == 7
    assert BenchmarkConfig.from_dict(data).seed is None
    assert BenchmarkConfig.from_dict({**data, "seed": None}).seed is None


def test_a_sampled_instance_is_fixed_by_the_seed(tmp_path, monkeypatch):
    first = _games(tmp_path / "first", monkeypatch, seed=5, game="nim:sample=True")
    second = _games(tmp_path / "second", monkeypatch, seed=5, game="nim:sample=True")

    piles = [record["instance_params"]["pile_sizes"] for record in first]
    # the instance is sampled per game, so the rounds differ, and a rerun samples the same ones
    assert len(set(piles)) > 1
    assert piles == [record["instance_params"]["pile_sizes"] for record in second]
    assert _moves(first) == _moves(second)


class _SeedRecordingLLM:
    # answers a fixed legal opening and keeps the seed of each call; reports it applied, as a seeded provider does
    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    async def generate(self, model_config, system, messages, output_schema=None, seed=None):
        self.seeds.append(seed)
        text = '{"action": "C1R1"}'
        return LLMResponse(model_config.provider, model_config.model_name, LLMTokens(), [OutputText([text])], text, output_schema, seed=seed)


def _llm_game(seed: int | None) -> tuple[_SeedRecordingLLM, dict]:
    llm = _SeedRecordingLLM()
    engine = registry.build_engine(registry.game_config("tic_tac_toe:"))
    player = LLMPlayer(registry.player_config("llm:actions:structured:openai:gpt-5.4:"), StructuredOutputStrategy(), llm, "i")  # type: ignore[arg-type]
    opponent = registry.build_player(engine.game, registry.player_config(RANDOM), "o")
    tracker = asyncio.run(engine.play((player, opponent), seed=seed))
    return llm, tracker.to_dict()


def test_the_llm_player_sends_a_seed_per_move_and_records_the_applied_one():
    llm, record = _llm_game(11)
    again, replayed = _llm_game(11)
    other, _ = _llm_game(12)

    assert record["seed"] == 11
    assert llm.seeds and all(seed is not None for seed in llm.seeds)
    assert llm.seeds == again.seeds and llm.seeds != other.seeds
    assert [step.get("llm_seed") for step in record["steps"] if step["player_name"].startswith("llm")] == llm.seeds
    assert record == replayed


def test_an_unseeded_game_sends_the_llm_no_seed():
    llm, record = _llm_game(None)

    assert llm.seeds and all(seed is None for seed in llm.seeds)
    assert "seed" not in record and all("llm_seed" not in step for step in record["steps"])
