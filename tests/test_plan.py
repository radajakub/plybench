"""The plan of an experiment (`--check`): games planned and already saved per matchup or training run, the models its players call, and every problem
found, without playing a game or calling a model."""

from __future__ import annotations

import asyncio

from plyllm import LLMConfig, ModelConfig
from test_learnable import _Stub, _StubParams

from plybench.app import PlyBench
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.plan import benchmark_plan, training_plan
from plybench.harness.training.training import TrainingHarness
from plybench.player.spec import PlayerSpec

UNIFORM = "random:distribution=uniform"
LLM_PLAYER = "llm:actions:structured:openai:gpt-5.4:"


def _op() -> PlyBench:
    # no credentials: every LLM provider is unconfigured; `stub:` is the learnable player of test_learnable
    op = PlyBench(LLMConfig())
    op.registry.register_player(PlayerSpec("stub", _StubParams, lambda game, cfg, pid: _Stub(cfg, pid)))
    return op


def test_a_benchmark_plan_counts_planned_and_saved_games(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("plan", _op(), ["tic_tac_toe:", "nim:"], [UNIFORM], [UNIFORM], 4, seed=0)
    plan = benchmark_plan(benchmark)
    assert (len(plan.items), plan.planned_games, plan.remaining_games, plan.models, plan.problems) == (2, 8, 8, [], [])

    asyncio.run(benchmark.run(sync=True))
    plan = benchmark_plan(benchmark)
    assert [item.saved_games for item in plan.items] == [4, 4] and plan.remaining_games == 0


def test_a_training_plan_counts_every_phase(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    schedule = "num_epochs=2,num_training_games=3,num_test_games=2"
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    harness = TrainingHarness("plan", _op(), ["tic_tac_toe:"], ["stub:"], [UNIFORM], [UNIFORM, "optimal:stochastic=True"], [schedule], 1, seed=0)
    plan = training_plan(harness)
    # 2 epochs x 3 training games, and 3 evaluations (epoch 0 included) x 2 games x 2 testers
    assert plan.planned_games == 2 * 3 + 3 * 2 * 2 and plan.items[0].schedule == schedule and plan.problems == []

    asyncio.run(harness.run(sync=True))
    plan = training_plan(harness)
    assert plan.items[0].saved_games == plan.planned_games and plan.remaining_games == 0


def test_a_plan_gives_the_optimal_outcomes_of_solvable_single_player_games_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    games = ["monty_hall:", "monty_hall:prize=goat,rules=hidden", "tic_tac_toe:", "connect_four:"]
    plan = benchmark_plan(Benchmark("plan", _op(), games, [UNIFORM], [UNIFORM], 2, seed=0))
    # standard host: switching wins 2/3; goat prize: staying wins 2/3. tic_tac_toe has an opponent, connect_four is not solvable
    assert list(plan.optimal) == ["monty_hall:", "monty_hall:prize=goat,rules=hidden"] and plan.problems == []
    for outcomes in plan.optimal.values():
        assert outcomes.keys() == {"win", "draw", "loss"}
        assert abs(outcomes["win"] - 2 / 3) < 1e-12 and outcomes["draw"] == 0.0 and abs(outcomes["loss"] - 1 / 3) < 1e-12
    assert plan.to_dict()["optimal"] == plan.optimal

    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    harness = TrainingHarness("plan", _op(), ["monty_hall:"], ["stub:"], [UNIFORM], [UNIFORM], ["num_epochs=1,num_training_games=1,num_test_games=1"], 1, seed=0)
    assert training_plan(harness).optimal == {"monty_hall:": plan.optimal["monty_hall:"]}


def test_a_training_plan_gives_the_optimal_outcomes_of_the_game_a_switch_changes_into_from_its_epoch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _Stub.proposed, _Stub.fail_at, _Stub.nothing_at = [], None, set()
    game = "bandit:p=0;1,switch=4/5;1/5,switch_epoch=2"
    harness = TrainingHarness("plan", _op(), [game], ["stub:"], [UNIFORM], [UNIFORM], ["num_epochs=3,num_training_games=1,num_test_games=1"], 1, seed=0)
    plan = training_plan(harness)
    assert list(plan.optimal) == [game, f"{game} from epoch 2"] and plan.problems == []
    assert plan.optimal[game]["win"] == 1.0 and abs(plan.optimal[f"{game} from epoch 2"]["win"] - 0.8) < 1e-12
    assert f"optimal policy of {game} from epoch 2: win 0.800" in plan.summary()


def test_a_model_without_credentials_is_a_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = benchmark_plan(Benchmark("plan", _op(), ["tic_tac_toe:"], [LLM_PLAYER], [UNIFORM], 2, seed=0))
    assert plan.models == ["openai:gpt-5.4"]
    assert len(plan.problems) == 1 and "openai is not configured" in plan.problems[0]


def test_two_runs_in_one_folder_are_a_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    schedules = ["num_epochs=1,num_training_games=2,num_test_games=2", "num_epochs=2,num_training_games=2,num_test_games=2"]
    plan = training_plan(TrainingHarness("plan", _op(), ["tic_tac_toe:"], [UNIFORM], [UNIFORM], [UNIFORM], schedules, 1, seed=0))
    assert any("would share" in problem for problem in plan.problems)


def test_a_trainee_that_cannot_learn_is_a_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = training_plan(TrainingHarness("plan", _op(), ["tic_tac_toe:"], [UNIFORM], [UNIFORM], [UNIFORM], ["num_epochs=1"], 1, seed=0))
    assert plan.problems == [f"trainee {UNIFORM} is not a LearnablePlayer and cannot be trained"]


def test_an_llm_player_names_its_model():
    params = _op().registry.player_config(LLM_PLAYER).params
    assert params.models() == [ModelConfig.from_string("openai:gpt-5.4")]
    assert _op().registry.player_config(UNIFORM).params.models() == []
