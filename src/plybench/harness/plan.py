from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from plyllm import ModelConfig

from plybench.app import PlyBench
from plybench.configs.names import item_names
from plybench.configs.player_config import PlayerConfig
from plybench.core.minimax import optimal_outcomes
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LearnablePlayer

# What an experiment would do, checked without playing a game or calling a model: per matchup or training run the games planned and the games already
# saved, the models its players call, and every problem found (a player that cannot be built for a game, a provider without credentials, a model the
# provider does not know, two training runs in one folder); also the exact outcomes of the optimal policy of every solvable single-player game, for
# tools that show them as a reference. For `--check` of plybench-run and plybench-train, and for tools that start runs for us.


@dataclass(frozen=True)
class PlannedItem:
    # one matchup of a benchmark, or one training run; `opponent` is the opponent of a matchup or the trainer of a run
    game: str
    player: str
    opponent: str
    planned_games: int
    saved_games: int
    schedule: str | None = None
    replicate: int | None = None
    # the item's results folder relative to the experiment's (the key of its report item), its short name and its name on each axis (plybench.configs.names)
    key: str = ""
    name: str = ""
    names: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentPlan:
    experiment: str
    kind: str  # "benchmark" or "training"
    items: list[PlannedItem] = field(default_factory=list)
    models: list[str] = field(default_factory=list)  # "<provider>:<model>", sorted, without options
    problems: list[str] = field(default_factory=list)
    # per game config, the probabilities of a win, a draw and a loss under the optimal policy; solvable single-player games only (with an opponent
    # the outcome depends on the opponent's policy)
    optimal: dict[str, dict[str, float]] = field(default_factory=dict)
    # the experiment's labels by the player's config string as the items write it, for tools that name the players
    labels: dict[str, str] = field(default_factory=dict)
    # per axis, what every item shares and its names leave out
    shared: dict[str, list[str]] = field(default_factory=dict)

    @property
    def planned_games(self) -> int:
        return sum(item.planned_games for item in self.items)

    @property
    def remaining_games(self) -> int:
        return sum(max(item.planned_games - item.saved_games, 0) for item in self.items)

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment": self.experiment,
            "kind": self.kind,
            "planned_games": self.planned_games,
            "remaining_games": self.remaining_games,
            "items": [item.__dict__ for item in self.items],
            "models": self.models,
            "problems": self.problems,
            "optimal": self.optimal,
            "labels": self.labels,
            "shared": self.shared,
        }

    def summary(self) -> str:
        lines = [
            f"{self.kind} {self.experiment}: {len(self.items)} {'matchup(s)' if self.kind == 'benchmark' else 'run(s)'}, {self.planned_games} games planned, {self.remaining_games} to play"
        ]
        for item in self.items:
            schedule = f" [{item.schedule}, replicate {item.replicate}]" if item.schedule is not None else ""
            lines.append(f"  {item.game} | {item.player} vs {item.opponent}{schedule}: {item.saved_games}/{item.planned_games} saved")
        lines.append(f"models: {', '.join(self.models) if self.models else 'none'}")
        for game, outcomes in self.optimal.items():
            lines.append(f"optimal policy of {game}: {', '.join(f'{result} {probability:.3f}' for result, probability in outcomes.items())}")
        lines.extend(f"PROBLEM: {problem}" for problem in self.problems)
        if not self.problems:
            lines.append("no problems found")
        return "\n".join(lines)


def _named(plan: ExperimentPlan, configs: list[dict[str, str]], prefixes: dict[str, str]) -> None:
    # the items' names, as a report of the same items names them
    names, joined, plan.shared = item_names(configs, plan.labels, prefixes)
    plan.items = [replace(item, name=name, names=axes) for item, axes, name in zip(plan.items, names, joined, strict=True)]


def _saved(folder: Path) -> int:
    return sum(1 for _ in folder.rglob("game_*.json")) if folder.exists() else 0


def _check_players(op: PlyBench, games: list[str], players: list[str], problems: list[str], learners: frozenset[str] = frozenset()) -> list[ModelConfig]:
    # build every player for every game (a plugin validates its params there), and collect the models they call; `learners` must build a LearnablePlayer
    models: list[ModelConfig] = []
    for game_string in games:
        engine = op.registry.build_engine(op.registry.game_config(game_string))
        engine.reset()
        for player_string in players:
            try:
                config: PlayerConfig = op.registry.player_config(player_string)
                player = op.registry.build_player(engine.game, config, "i")
            except Exception as error:
                # any error: the check reports every problem at once, instead of the run failing on its first player
                problems.append(f"{player_string} cannot be built for {game_string}: {type(error).__name__}: {error}")
                continue
            if player_string in learners and not isinstance(player, LearnablePlayer):
                problems.append(f"trainee {player_string} is not a LearnablePlayer and cannot be trained")
            models.extend(config.params.models())
    return models


def _check_models(op: PlyBench, models: list[ModelConfig], problems: list[str]) -> list[str]:
    names = sorted({f"{model.provider.value}:{model.model_name}" for model in models})
    for model in {(model.provider, model.model_name) for model in models}:
        try:
            op.llm.resolve_model(*model)
        except ValueError as error:
            problems.append(str(error))
    return names


def optimal_outcomes_of(op: PlyBench, games: list[str], problems: list[str]) -> dict[str, dict[str, float]]:
    # the outcomes of the optimal policy of every solvable single-player game; the solution is read from the minimax cache, or solved and saved there
    # on first use, as a run with an optimal player or an oracle component would
    optimal: dict[str, dict[str, float]] = {}
    for game_string in games:
        config = op.registry.game_config(game_string)
        if not op.registry.solvable(config.key):
            continue
        engine = op.registry.build_engine(config)
        engine.reset()
        if not engine.game.single_player():
            continue
        try:
            outcomes = optimal_outcomes(engine.game)
        except Exception as error:
            # any error, as for the players: the check reports every problem at once
            problems.append(f"the optimal policy of {game_string} cannot be evaluated: {type(error).__name__}: {error}")
            continue
        optimal[game_string] = {result.name.lower(): probability for result, probability in outcomes.items()}
    return optimal


def benchmark_plan(benchmark: Benchmark) -> ExperimentPlan:
    op = benchmark.op
    plan = ExperimentPlan(benchmark.experiment, "benchmark", labels=dict(benchmark.labels))
    # in the order the benchmark plays them
    matrix = [(game, player, opponent) for player in benchmark.player_configs for opponent in benchmark.opponent_configs for game in benchmark.game_configs]
    for game, player, opponent in matrix:
        game_config, player_config, opponent_config = op.registry.game_config(game), op.registry.player_config(player), op.registry.player_config(opponent)
        folder = benchmark.path_builder.game_base(benchmark.experiment, game_config, player_config, opponent_config, benchmark.num_games)
        # the configs as the results write them (to_string), as a training run's, so a tool can match an item with its results
        key = folder.relative_to(benchmark.path_builder.results_dir / benchmark.experiment).as_posix()
        plan.items.append(PlannedItem(game_config.to_string(), player_config.to_string(), opponent_config.to_string(), benchmark.num_games, _saved(folder), key=key))
    _named(plan, [{"game": item.game, "player": item.player, "opponent": item.opponent} for item in plan.items], {"opponent": "vs "})
    models = _check_players(op, benchmark.game_configs, list(dict.fromkeys(benchmark.player_configs + benchmark.opponent_configs)), plan.problems)
    plan.models = _check_models(op, models, plan.problems)
    plan.optimal = optimal_outcomes_of(op, benchmark.game_configs, plan.problems)
    return plan


def training_plan(harness: TrainingHarness) -> ExperimentPlan:
    op = harness.op
    plan = ExperimentPlan(harness.experiment, "training", labels=dict(harness.labels))
    for run in harness.runs:
        schedule = run.training_config
        # epoch 0 is only evaluated; every epoch after it trains, then is evaluated against every tester
        planned = schedule.num_epochs * schedule.num_training_games + (schedule.num_epochs + 1) * schedule.num_test_games * len(harness.tester_configs)
        folder = harness.path_builder.run_dir(harness.experiment, run)
        key = folder.relative_to(harness.path_builder.results_dir / harness.experiment).as_posix()
        plan.items.append(
            PlannedItem(run.game.to_string(), run.trainee.to_string(), run.trainer.to_string(), planned, _saved(folder), schedule.to_string(), run.replicate, key=key)
        )
    configs = [{"game": i.game, "player": i.player, "opponent": i.opponent, "schedule": i.schedule or "", "replicate": str(i.replicate)} for i in plan.items]
    _named(plan, configs, {"opponent": "vs ", "replicate": "rep "})
    try:
        harness.check_runs()
    except ValueError as error:
        plan.problems.append(str(error))
    players = list(dict.fromkeys(harness.player_configs + harness.trainer_configs + harness.tester_configs))
    models = _check_players(op, harness.game_configs, players, plan.problems, frozenset(harness.player_configs))
    plan.models = _check_models(op, models, plan.problems)
    plan.optimal = optimal_outcomes_of(op, harness.game_configs, plan.problems)
    return plan
