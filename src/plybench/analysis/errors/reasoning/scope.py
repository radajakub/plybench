from __future__ import annotations

from plybench.analysis.errors.moves import TracedMove
from plybench.analysis.errors.reasoning.codebook import SCOPE_UNIVERSAL, Code
from plybench.analysis.recognition import original_game_name, recognizable


def move_game_key(move: TracedMove) -> str:
    return move.matchup.game.split(":", 1)[0]


def code_applies(code: Code, move: TracedMove) -> bool:
    if code.scope == SCOPE_UNIVERSAL:
        return True
    game_key = move_game_key(move)
    if code.scope.startswith("game:"):
        return code.scope == f"game:{game_key}"
    if code.scope.startswith("family:"):
        family = original_game_name(game_key) if recognizable(game_key) else game_key
        return code.scope == f"family:{family}"
    return False
