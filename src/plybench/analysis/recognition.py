from __future__ import annotations

import re
from functools import lru_cache

from plybench.trackers.game_tracker import GameStep

# The real-world game each (possibly obfuscated) variant is built on. A reasoning trace "recognizes"
# the underlying game when it names this game (or one of its aliases) despite the obfuscated framing.
_ORIGINAL_GAME_NAMES: dict[str, str] = {
    "tic_tac_toe": "tic tac toe",
    "modified_tic_tac_toe": "tic tac toe",
    "magic_square": "tic tac toe",
    "story_magic_square": "tic tac toe",
    "nim": "nim",
    "modified_nim": "nim",
    "inverse_nim": "nim",
    "story_nim": "nim",
    "connect_four": "connect four",
    "breakthrough": "breakthrough",
}

_EXTRA_ALIASES: dict[str, set[str]] = {
    "tic tac toe": {
        "tic tac toe",
        "tic-tac-toe",
        "tic_tac_toe",
        "tictactoe",
        "ttt",
        "noughts and crosses",
        "naughts and crosses",
    },
    "nim": {
        "nim",
        "game of nim",
    },
    "connect four": {
        "connect four",
        "connect-four",
        "connect 4",
        "connect-4",
    },
    "breakthrough": {
        "breakthrough",
        "break through",
    },
}


def recognizable(game_key: str) -> bool:
    return game_key in _ORIGINAL_GAME_NAMES


def original_game_name(game_key: str) -> str:
    return _ORIGINAL_GAME_NAMES[game_key]


def original_game_aliases(game_key: str) -> set[str]:
    name = original_game_name(game_key)
    aliases = set(_EXTRA_ALIASES.get(name, {name}))

    words = name.split()
    if len(words) > 1:
        for sep in (" ", "-", "_", ""):
            aliases.add(sep.join(words))
        aliases.add("".join(word[0] for word in words))
    return aliases


def _alias_pattern(alias: str) -> re.Pattern[str]:
    # Split on separators first, then rejoin with a flexible separator class so "connect four",
    # "connect-four" and "connect_four" all match. Escaping then substituting would corrupt the class.
    parts = [re.escape(part) for part in re.split(r"[\s_-]+", alias) if part]
    body = r"[\s_-]+".join(parts) if parts else re.escape(alias)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


@lru_cache(maxsize=None)
def _compiled_patterns(game_key: str) -> tuple[re.Pattern[str], ...]:
    return tuple(_alias_pattern(alias) for alias in sorted(original_game_aliases(game_key)))


def trace_mentions_original_game(trace: str | None, game_key: str) -> bool:
    if not trace or not recognizable(game_key):
        return False
    return any(pattern.search(trace) for pattern in _compiled_patterns(game_key))


def step_reasoning_trace(step: GameStep) -> str | None:
    if not step.data:
        return None
    trace = step.data.get("reasoning_trace")
    return trace if isinstance(trace, str) and trace else None
