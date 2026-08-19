from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from plybench.analysis.errors.moves import TracedMove

# Section headers are fixed strings so every pass shows a move the same way: a prompt-revision bump then
# means the instructions changed, not the rendering.
POSITION_HEADER = "### Position the player was shown"
LEGAL_HEADER = "### Legal moves (exact strings)"
TRACE_HEADER = "### The player's reasoning trace"
CHOICE_HEADER = "### The move the player actually played"
OPTIMAL_HEADER = "### Optimal moves according to the solver"


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


def _section(header: str, body: str) -> str:
    return f"{header}\n{body}"


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_move(move: TracedMove, reveal_choice: bool = False, reveal_optimal: bool = False) -> str:
    """The move as a prompt block. `reveal_choice` and `reveal_optimal` are off by default because a judge
    told what was played, or what the solver preferred, tends to reason backwards from it — induction
    turns them on deliberately (discovery needs to see what went wrong), measurement does not."""
    sections = [
        _section(POSITION_HEADER, move.observation),
        _section(LEGAL_HEADER, _bullets(move.legal_moves)),
        _section(TRACE_HEADER, move.trace or "(no trace recorded)"),
    ]
    if reveal_choice:
        sections.append(_section(CHOICE_HEADER, move.move))
    if reveal_optimal:
        sections.append(_section(OPTIMAL_HEADER, _bullets(move.optimal_moves)))
    return "\n\n".join(sections)


def render_batch(moves: Sequence[TracedMove], reveal_choice: bool = False, reveal_optimal: bool = False) -> str:
    """Several moves in one prompt, numbered from 1. The number is how a judge refers back to a move, so
    it is the only handle it gets — uids are never shown, since a judge has no use for them."""
    blocks = [f"## Move {index}\n\n{render_move(move, reveal_choice, reveal_optimal)}" for index, move in enumerate(moves, start=1)]
    return "\n\n".join(blocks)
