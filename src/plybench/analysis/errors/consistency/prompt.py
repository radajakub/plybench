from __future__ import annotations

from plybench.analysis.errors.judge.prompts import Prompt, render_move
from plybench.analysis.errors.moves import TracedMove

# bump when the instructions below change: the revision is part of the annotator id, so verdicts made
# under different wordings never end up pooled in one column
CONSISTENCY_REVISION = "consistency:v2"

_SYSTEM = """You are analysing a reasoning transcript written by a player of a two-player game. You are \
given the position the player was shown, the exact list of legal moves, and the player's reasoning trace.

Your only job: report which single legal move the trace commits to as its final decision.

Rules:
1. A trace is a search transcript. It may consider, test and reject many candidate moves. Only the move \
it finally settles on counts; a move it examined and moved on from is not a decision.
2. If the trace changes its mind, the decision is the last one it stands behind.
3. Report the move exactly as it appears in the legal-moves list, so it can be matched mechanically.
4. You are NOT told which move was actually played, and you must not guess it. Report only what the trace \
itself concludes.
5. Use status="decided" only when the trace unambiguously settles on one legal move. Use "no_conclusion" \
when it stops without choosing, and "ambiguous" when it leaves two or more candidates equally live.
6. evidence must be a verbatim quote from the trace showing that decision, and empty otherwise."""


def consistency_prompt(move: TracedMove) -> Prompt:
    # blind on purpose: the judge sees the position, the legal moves and the trace, never the move played
    return Prompt(_SYSTEM, render_move(move))
