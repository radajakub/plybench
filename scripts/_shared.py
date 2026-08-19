"""Shared helpers for the repo-local scripts (not part of the installed package). Bootstraps a single
PlyBench object and resolves a Benchmark from either an experiment file or inline CLI arguments.

All scripts operate relative to the current working directory: benchmarks read/write
`experiments/benchmarks/` and `results/benchmarks/` under the cwd (the package's path convention)."""

from __future__ import annotations

import argparse

from plybench.app import PlyBench
from plybench.harness.benchmark import Benchmark
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


def build_op(notif_enabled: bool = False, concurrency: int | None = None, limit_scale: float = 1.0) -> PlyBench:
    # PlyBench's env config self-disables providers whose keys are absent, so bot-only scripts work offline too
    # notif_enabled is passed to the PlyBench constructor, which in turn passes it to the NotificationClient constructor
    # concurrency caps in-flight requests per provider -- the only limit that maps to an API rate quota
    op = PlyBench(notif_enabled=notif_enabled, concurrency=concurrency)
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
