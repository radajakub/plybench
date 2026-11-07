from __future__ import annotations

# shared by the command-line tools (plybench-run, plybench-train) and the repo-local scripts: one PlyBench object, the account's
# per-model limits, and a Benchmark or TrainingHarness from an experiment file or inline arguments. Everything reads and writes
# relative to the working directory (experiments/, results/), the package's path convention
import argparse
import json
import warnings
from pathlib import Path

import clankers

from plybench.app import PlyBench
from plybench.cli.limits import LIMITS_ENV, PROVIDER_WIDE, Limits, limits_path, load_limits
from plybench.common.paths import BenchmarkPathBuilder
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.harness.training.training import TrainingHarness


def warn_if_unconfigured() -> None:
    # clankers builds its backend lazily and only warns when a send fails, so a missing NTFY_URL/NTFY_TOPIC
    # would otherwise go unnoticed until the first notification is due
    try:
        _ = clankers.default().backend
    except ValueError as error:
        print(f"warning: --notify set but notifications are not configured ({error}); they will be skipped")


def add_limits_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limits", type=Path, help=f"per-model quotas of your account, as JSON (default: ${LIMITS_ENV}, then ./limits.json if present)")


def apply_model_limits(op: PlyBench, limits: Limits, scale: float = 1.0) -> None:
    # scale leaves headroom when several runs share the account, since each process paces only itself
    available = set(op.llm.available_providers)
    for provider, models in limits.items():
        if provider not in available:
            continue
        for model_name, model_limits in models.items():
            # the whole provider's concurrency is not scaled, as a model's concurrency is not (ModelLimits.scaled)
            if model_name == PROVIDER_WIDE:
                op.llm.set_concurrency(provider, model_limits.max_concurrent)
            else:
                op.llm.set_model_limits(provider, model_name, model_limits.scaled(scale) if scale != 1.0 else model_limits)


def _one_line_warning(message: Warning | str, category: type[Warning], filename: str, lineno: int, line: str | None = None) -> str:
    # the CLI's own notices are one "warning: ..." line; Python's default adds the file, line number, category and source line
    return f"warning: {message}\n"


def build_op(concurrency: int | None = None, limit_scale: float = 1.0, limits: Path | None = None) -> PlyBench:
    # every CLI and script starts here, so this is where it decides how warnings look (the libraries only raise them)
    warnings.formatwarning = _one_line_warning
    # PlyBench's env config self-disables providers whose keys are absent, so bot-only scripts work offline too
    # concurrency caps in-flight requests per provider -- the only limit that maps to an API rate quota
    op = PlyBench(concurrency=concurrency)
    # models are registered without limits; the account's quotas come from its own file (see cli/limits.py)
    path = limits_path(limits)
    if path is not None:
        print(f"per-model limits from {path}")
        apply_model_limits(op, load_limits(path), limit_scale)
    return op


def add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", help="experiment name under experiments/benchmarks/<name>.json")
    parser.add_argument("--name", help="experiment name for an inline benchmark (and export wrapper)")
    parser.add_argument("--games", nargs="+", metavar="CONFIG", help="game config strings, e.g. tic_tac_toe:")
    parser.add_argument("--players", nargs="+", metavar="CONFIG", help="player config strings")
    parser.add_argument("--opponents", nargs="+", metavar="CONFIG", help="opponent config strings")
    parser.add_argument("--num-games", type=int, default=2, help="rounds per matchup (inline only; default 2)")
    parser.add_argument("--seed", type=int, help="base seed of every game's seed (inline only; an experiment file sets its own; default: unseeded)")


def benchmark_from_args(op: PlyBench, args: argparse.Namespace) -> Benchmark:
    if args.experiment:
        # per-axis overrides restrict the experiment's enabled set at run time
        return Benchmark.load_experiment(op, args.experiment, game_override=args.games, player_override=args.players, opponent_override=args.opponents)
    if not (args.games and args.players and args.opponents):
        raise SystemExit("provide --experiment, or all of --games / --players / --opponents (+ optional --num-games)")
    return Benchmark(args.name or "benchmark", op, args.games, args.players, args.opponents, args.num_games, seed=args.seed)


def add_training_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", help="experiment name under experiments/training/<name>.json")
    parser.add_argument("--name", help="experiment name for an inline training run")
    parser.add_argument("--games", nargs="+", metavar="CONFIG", help="game config strings, e.g. tic_tac_toe:")
    parser.add_argument("--trainees", nargs="+", metavar="CONFIG", help="learnable player config strings")
    parser.add_argument("--trainers", nargs="+", metavar="CONFIG", help="opponent config strings for the training phases")
    parser.add_argument("--testers", nargs="+", metavar="CONFIG", help="opponent config strings for the evaluation phases")
    parser.add_argument("--schedules", nargs="+", metavar="CONFIG", help="schedules, e.g. num_epochs=10,num_training_games=50,num_test_games=50")
    parser.add_argument("--replicates", type=int, default=1, help="independent repeats of each run (inline only; default 1)")
    parser.add_argument("--seed", type=int, help="base seed of every game's seed (inline only; an experiment file sets its own; default: unseeded)")


def training_from_args(op: PlyBench, args: argparse.Namespace) -> TrainingHarness:
    if args.experiment:
        # per-axis overrides restrict the experiment's enabled set at run time; num_replicates is not
        # overridable, it belongs to the experiment the results are filed under
        return TrainingHarness.load_experiment(
            op,
            args.experiment,
            game_override=args.games,
            player_override=args.trainees,
            trainer_override=args.trainers,
            tester_override=args.testers,
            training_override=args.schedules,
        )
    if not (args.games and args.trainees and args.trainers and args.testers and args.schedules):
        raise SystemExit("provide --experiment, or all of --games / --trainees / --trainers / --testers / --schedules (+ optional --replicates)")
    return TrainingHarness(args.name or "training", op, args.games, args.trainees, args.trainers, args.testers, args.schedules, args.replicates, seed=args.seed)


def discover_matchups(op: PlyBench, experiment: str, game_str: str, paths: BenchmarkPathBuilder) -> tuple[set[str], set[str], int]:
    # every recorded matchup's metadata for one game gives its exact model/opponent config strings and the games-per-matchup
    # count, so nothing about the config space is hard-coded
    game = op.registry.game_config(game_str)
    models: set[str] = set()
    opponents: set[str] = set()
    num_games = 0
    for metadata in (paths.results_dir / experiment).glob(f"{game.path}_*/*/metadata.json"):
        data = json.loads(metadata.read_text())
        if data.get("game_config") != game.to_string():
            continue  # a different game whose path shares the prefix
        models.add(data["i_config"])
        opponents.add(data["o_config"])
        num_games = max(num_games, data.get("n_games", 0))
    return models, opponents, num_games


def load_paired_results(op: PlyBench, experiment: str, game_a: str, game_b: str, paths: BenchmarkPathBuilder) -> BenchmarkResults:
    # recorded results for two games over the models and opponents they have in common, which is what any A-vs-B comparison
    # needs (a model present in only one game cannot be compared)
    models_a, opponents_a, n_a = discover_matchups(op, experiment, game_a, paths)
    models_b, opponents_b, n_b = discover_matchups(op, experiment, game_b, paths)
    models, opponents = sorted(models_a & models_b), sorted(opponents_a & opponents_b)
    if not models or not opponents:
        raise SystemExit("no shared models/opponents found for the two games (check --experiment and config strings)")
    return Benchmark(experiment, op, [game_a, game_b], models, opponents, max(n_a, n_b), paths).get_results()
