"""The dimensions a cell can be split or pooled along.

A cell is one (game config, player config) pair, but that is rarely the question anyone asks. "How does
gpt-5-nano behave" cuts across every game it played; "what does obfuscation do" cuts across every model
that played the obfuscated variant. Each facet below names one such cut, and pooling is grouping cells by
any combination of them -- so the axes are data rather than a hardcoded list of three."""

from __future__ import annotations

from collections.abc import Callable

from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.recognition import original_game_name, recognizable
from plybench.configs.player_config import PlayerConfig
from plybench.llm.providers.providers import Provider
from plybench.player.llm_player import LLMParams

UNKNOWN = "-"


def _model(player: PlayerConfig) -> str:
    # read off the typed config rather than the serialized string: a bot carries no model at all, and the
    # llm format (observation:strategy:provider:model:options) is not something to re-parse by hand
    params = player.params
    return f"{params.model.provider.value}:{params.model.model_name}" if isinstance(params, LLMParams) else player.key


def _effort(player: PlayerConfig) -> str:
    params = player.params
    if not isinstance(params, LLMParams):
        return UNKNOWN
    return params.model.options.reasoning_effort or UNKNOWN


# Providers we host ourselves, which return the chain of thought. Everyone else returns a summary their
# own summariser wrote -- OpenAI under reasoning={"summary": "detailed"}, Gemini under
# include_thoughts=True -- and both do so by design, never the reasoning itself.
SELF_HOSTED: frozenset[Provider] = frozenset({Provider.METACENTRUM, Provider.HUGGINGFACE})

RAW_REASONING = "raw_reasoning"
PROVIDER_SUMMARY = "provider_summary"


def _trace_kind(player: PlayerConfig) -> str:
    """What the stored trace actually is, which decides what an error rate over it means.

    A rate measured on a summary is a rate over what the summariser kept, and is not comparable with a
    rate over a raw chain of thought: a lower figure may mean cleaner reasoning or a tidier summary, and
    nothing in this design separates the two. The stored-trace over billed-output ratios measured on this
    corpus are 0.56-0.78 for the self-hosted models and 0.10-0.45 for the commercial ones, with one
    overlap, so this is a two-group design rather than a threshold. Any number pooling the two groups is
    uninterpretable, which is why the split is a facet and not a footnote."""
    params = player.params
    if not isinstance(params, LLMParams):
        return UNKNOWN
    return RAW_REASONING if params.model.provider in SELF_HOSTED else PROVIDER_SUMMARY


def _family(funnel: FunnelResult) -> str:
    """The real-world game underneath, so every obfuscation of it groups together. This is what makes
    `presentation` mean something: the two facets are only informative held against each other."""
    key = funnel.game.key
    return original_game_name(key) if recognizable(key) else key


FACETS: dict[str, Callable[[FunnelResult], str]] = {
    "experiment": lambda funnel: funnel.experiment,
    "family": _family,  # tic tac toe, nim -- the game underneath any obfuscation
    "presentation": lambda funnel: funnel.game.key,  # magic_square, story_magic_square -- the obfuscation itself
    "game": lambda funnel: funnel.game.to_string(),  # the full config, parameters included
    "model": lambda funnel: _model(funnel.player),
    # open-weight vs commercial, named after what actually differs: the trace itself
    "trace_kind": lambda funnel: _trace_kind(funnel.player),
    "effort": lambda funnel: _effort(funnel.player),
    "player": lambda funnel: funnel.player.to_string(),  # the full config: model, effort and everything else
}

# a cell is the finest grain the funnel produces, so this is what `Scope.of` records for one
CELL_FACETS: tuple[str, ...] = ("experiment", "family", "presentation", "game", "trace_kind", "model", "effort", "player")


def facet_values(funnel: FunnelResult, names: tuple[str, ...] = CELL_FACETS) -> tuple[tuple[str, str], ...]:
    return tuple((name, FACETS[name](funnel)) for name in names)


def parse_axis(axis: str) -> tuple[str, ...]:
    """A pooling axis written as "model" or "model,presentation". Unknown names fail here rather than
    silently pooling over everything, which would look like a result instead of a typo."""
    names = tuple(name.strip() for name in axis.split(",") if name.strip())
    unknown = [name for name in names if name not in FACETS]
    if unknown or not names:
        raise ValueError(f"Unknown facet(s) {unknown or ['(empty)']} -- available: {', '.join(FACETS)}")
    return names
