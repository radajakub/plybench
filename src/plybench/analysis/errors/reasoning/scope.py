from __future__ import annotations

from functools import lru_cache

from plybench.analysis.errors.moves import TracedMove
from plybench.analysis.errors.reasoning.codebook import SCOPE_UNIVERSAL, Code
from plybench.analysis.recognition import original_game_name, recognizable


def move_game_key(move: TracedMove) -> str:
    return move.matchup.game.split(":", 1)[0]


@lru_cache(maxsize=None)
def scope_applies(scope: str, game_key: str) -> bool:
    """Whether a code declared at `scope` may be used on a move from `game_key`.

    Cached because it is asked once per (code, move) pair in the prevalence tables -- tens of millions of
    times over a full census -- while the answer depends on nothing but these two strings. Uncached, the
    family branch resolves the obfuscated game name on every call."""
    if scope == SCOPE_UNIVERSAL:
        return True
    if scope.startswith("game:"):
        return scope == f"game:{game_key}"
    if scope.startswith("family:"):
        family = original_game_name(game_key) if recognizable(game_key) else game_key
        return scope == f"family:{family}"
    return False


def code_applies(code: Code, move: TracedMove) -> bool:
    return scope_applies(code.scope, move_game_key(move))
