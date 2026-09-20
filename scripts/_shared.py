"""Shared helpers for the repo-local scripts (not part of the installed package). Bootstraps a single
PlyBench object and resolves a Benchmark from either an experiment file or inline CLI arguments.

All scripts operate relative to the current working directory: benchmarks read/write
`experiments/benchmarks/` and `results/benchmarks/` under the cwd (the package's path convention)."""

from __future__ import annotations

import argparse
import json

from plybench.app import PlyBench
from plybench.common.paths import BenchmarkPathBuilder
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.harness.training.training import TrainingHarness
from plybench.llm import ModelLimits, Provider

# Per-model quotas for *this* account, keyed by provider and model name (read them off the provider's
# console -- for Mistral, Admin -> API -> Limits). They live here rather than in the package because
# they are account-specific: the package ships no limits, and another tier's quotas would differ.
# Providers and models absent from this map are governed only by the provider-wide semaphore.
LIMITS: dict[Provider, dict[str, ModelLimits]] = {
    Provider.MISTRAL: {
        "mistral-medium-3.5": ModelLimits(max_concurrent=16, rps=16.67, tpm=500_000),
        "mistral-small-4": ModelLimits(max_concurrent=8, rps=1.67, tpm=100_000),
    },
    # every provider is gated the same way, so any of them can be paced by adding a block here; each
    # field is optional, e.g. tokens only:
    # Provider.OPENAI: {
    #     "gpt-5.4": ModelLimits(tpm=2_000_000),
    # },
}


def apply_model_limits(op: PlyBench, scale: float = 1.0) -> None:
    """Install the account's per-model quotas. `scale` leaves headroom when several runs share the
    account, since each process paces only itself."""
    available = set(op.llm.available_providers)
    for provider, models in LIMITS.items():
        if provider not in available:
            continue
        for model_name, limits in models.items():
            op.llm.set_model_limits(provider, model_name, limits.scaled(scale) if scale != 1.0 else limits)


def build_op(concurrency: int | None = None, limit_scale: float = 1.0) -> PlyBench:
    # PlyBench's env config self-disables providers whose keys are absent, so bot-only scripts work offline too
    # concurrency caps in-flight requests per provider -- the only limit that maps to an API rate quota
    op = PlyBench(concurrency=concurrency)
    # models are registered without limits, so quotas are applied here rather than by the package
    apply_model_limits(op, limit_scale)
    return op


def add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", help="experiment name under experiments/benchmarks/<name>.json")
    parser.add_argument("--name", help="experiment name for an inline benchmark (and export wrapper)")
    parser.add_argument("--games", nargs="+", metavar="CONFIG", help="game config strings, e.g. tic_tac_toe:")
    parser.add_argument("--players", nargs="+", metavar="CONFIG", help="player config strings")
    parser.add_argument("--opponents", nargs="+", metavar="CONFIG", help="opponent config strings")
    parser.add_argument("--num-games", type=int, default=2, help="rounds per matchup (inline only; default 2)")


def benchmark_from_args(op: PlyBench, args: argparse.Namespace) -> Benchmark:
    if args.experiment:
        # per-axis overrides restrict the experiment's enabled set at run time
        return Benchmark.load_experiment(op, args.experiment, game_override=args.games, player_override=args.players, opponent_override=args.opponents)
    if not (args.games and args.players and args.opponents):
        raise SystemExit("provide --experiment, or all of --games / --players / --opponents (+ optional --num-games)")
    return Benchmark(args.name or "benchmark", op, args.games, args.players, args.opponents, args.num_games)


def add_training_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", help="experiment name under experiments/training/<name>.json")
    parser.add_argument("--name", help="experiment name for an inline training run")
    parser.add_argument("--games", nargs="+", metavar="CONFIG", help="game config strings, e.g. tic_tac_toe:")
    parser.add_argument("--trainees", nargs="+", metavar="CONFIG", help="learnable player config strings")
    parser.add_argument("--trainers", nargs="+", metavar="CONFIG", help="opponent config strings for the training phases")
    parser.add_argument("--testers", nargs="+", metavar="CONFIG", help="opponent config strings for the evaluation phases")
    parser.add_argument("--schedules", nargs="+", metavar="CONFIG", help="schedules, e.g. num_epochs=10,num_training_games=50,num_test_games=50")
    parser.add_argument("--replicates", type=int, default=1, help="independent repeats of each run (inline only; default 1)")


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
    return TrainingHarness(args.name or "training", op, args.games, args.trainees, args.trainers, args.testers, args.schedules, args.replicates)


def discover_matchups(op: PlyBench, experiment: str, game_str: str, paths: BenchmarkPathBuilder) -> tuple[set[str], set[str], int]:
    """Read every recorded matchup's metadata for one game to recover its exact model/opponent config
    strings and the games-per-matchup count, so nothing about the config space is hard-coded."""
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
    """Load recorded results for two games over the models and opponents they have in common, which is
    what any A-vs-B comparison needs (a model present in only one game cannot be compared)."""
    models_a, opponents_a, n_a = discover_matchups(op, experiment, game_a, paths)
    models_b, opponents_b, n_b = discover_matchups(op, experiment, game_b, paths)
    models, opponents = sorted(models_a & models_b), sorted(opponents_a & opponents_b)
    if not models or not opponents:
        raise SystemExit("no shared models/opponents found for the two games (check --experiment and config strings)")
    return Benchmark(experiment, op, [game_a, game_b], models, opponents, max(n_a, n_b), paths).get_results()
