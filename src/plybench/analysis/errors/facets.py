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
    "effort": lambda funnel: _effort(funnel.player),
    "player": lambda funnel: funnel.player.to_string(),  # the full config: model, effort and everything else
}

# a cell is the finest grain the funnel produces, so this is what `Scope.of` records for one
CELL_FACETS: tuple[str, ...] = ("experiment", "family", "presentation", "game", "model", "effort", "player")


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
