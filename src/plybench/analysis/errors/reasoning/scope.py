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


def move_family(move: TracedMove) -> str:
    """The game underneath the obfuscation -- tic_tac_toe for magic_square and story_magic_square alike."""
    game_key = move_game_key(move)
    return original_game_name(game_key) if recognizable(game_key) else game_key


def code_applies(code: Code, move: TracedMove) -> bool:
    """Whether the move is inside the code's declared level.

    This is a *reporting* predicate. It is deliberately not consulted when deciding which codes an
    annotator may use: a code offered only where it is expected to occur can only ever be observed there,
    and reporting that as a finding about levels would be assuming the conclusion."""
    return scope_applies(code.scope, move_game_key(move))
