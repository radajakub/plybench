"""Token accounting over recorded results: how much an experiment actually spent, per matchup, and
what that costs in USD.

This is a sum, not a statistic -- the metric suite already reports per-game/per-move token means with
confidence intervals; here every recorded call is added up. Two known gaps, both understatements of
the true bill in one direction each: the recorded steps carry no cache split (`cached_input_tokens`
stays 0, so a cost derived from them charges all input at the uncached rate -> upper bound), and only
the response that produced a move is recorded (retried or unparsable calls consumed tokens that were
never written down -> lower bound on the count itself).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeVar

from plybench.common.progress import track
from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.llm.model_config import ModelConfig
from plybench.llm.router import LLM
from plybench.llm.tokens import LLMTokens
from plybench.player.llm_player import LLMParams, options_to_string
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.result_tracker import ResultTracker

K = TypeVar("K")

# a step that recorded any token count recorded all of them, so one field decides whether it was a call
CALL_ATTR = "input_tokens"


@dataclass(frozen=True)
class TokenUsage:
    tokens: LLMTokens = field(default_factory=LLMTokens)
    calls: int = 0

    @property
    def total(self) -> int:
        # reasoning_tokens is the reasoning subset of output_tokens, so only the two billed sides add up
        return self.tokens.input_tokens + self.tokens.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(self.tokens + other.tokens, self.calls + other.calls)


@dataclass(frozen=True)
class MatchupUsage:
    experiment: str
    game: GameConfig
    player: PlayerConfig
    opponent: PlayerConfig
    model: ModelConfig | None
    usage: TokenUsage


def player_model(config: PlayerConfig) -> ModelConfig | None:
    params = config.params
    return params.model if isinstance(params, LLMParams) else None


def model_label(model: ModelConfig | None) -> str:
    if model is None:
        return "unpriced (not a model)"
    options = options_to_string(model.options)
    return f"{model.provider.value}/{model.model_name}" + (f" ({options})" if options else "")


def game_usage(game: GameTracker, player: PlayerConfig) -> TokenUsage:
    steps = game.steps_of(player, CALL_ATTR)
    tokens = sum(
        (LLMTokens(input_tokens=step.input_tokens or 0, output_tokens=step.output_tokens or 0, reasoning_tokens=step.reasoning_tokens or 0) for step in steps),
        LLMTokens(),
    )
    return TokenUsage(tokens, len(steps))


def matchup_usage(tracker: ResultTracker) -> list[MatchupUsage]:
    games = [game for game in tracker.games if game is not None]
    sides = [(tracker.i, tracker.o)] if tracker.i.hash == tracker.o.hash else [(tracker.i, tracker.o), (tracker.o, tracker.i)]

    entries = []
    for player, opponent in sides:
        usage = sum((game_usage(game, player) for game in games), TokenUsage())
        if usage.calls:
            entries.append(MatchupUsage(tracker.experiment, tracker.game, player, opponent, player_model(player), usage))
    return entries


def benchmark_usage(results: BenchmarkResults, progress: bool | None = None) -> list[MatchupUsage]:
    trackers = track(results.trackers, "Summing tokens", len(results.trackers), progress)
    return [usage for tracker in trackers for usage in matchup_usage(tracker)]


def group_by(usages: Iterable[MatchupUsage], key: Callable[[MatchupUsage], K]) -> dict[K, list[MatchupUsage]]:
    grouped: dict[K, list[MatchupUsage]] = {}
    for usage in usages:
        grouped.setdefault(key(usage), []).append(usage)
    return grouped


def total_usage(usages: Iterable[MatchupUsage]) -> TokenUsage:
    return sum((usage.usage for usage in usages), TokenUsage())


def entry_cost(llm: LLM, usage: MatchupUsage) -> float | None:
    if usage.model is None or usage.model.provider not in llm.available_providers:
        return None
    try:
        return llm.calculate_cost(usage.model, usage.usage.tokens)
    except ValueError:
        return None


def total_cost(llm: LLM, usages: Iterable[MatchupUsage]) -> tuple[float, int]:
    costs = [entry_cost(llm, usage) for usage in usages]
    return sum(cost for cost in costs if cost is not None), sum(cost is None for cost in costs)
