"""Walking one cell's moves through the detectors. Kept apart from `detection` so the rules stay a pure
function of a position and the statistics never have to import the engine to read a label."""

from __future__ import annotations

from plybench.analysis.errors.funnel.result import FunnelResult
from plybench.analysis.errors.procedural.detection import MoveDiagnosis, detect
from plybench.analysis.errors.procedural.position import MovePosition
from plybench.analysis.errors.procedural.refutation import RefutationSolver, solver_for
from plybench.analysis.replay import ReplayerCache
from plybench.common.progress import track


def _diagnose(position: MovePosition, game_key: str, refute: RefutationSolver) -> MoveDiagnosis:
    return MoveDiagnosis(detect(position, game_key), refute(position))


def label_moves(funnel: FunnelResult, replayers: ReplayerCache, progress: bool | None = None) -> dict[str, MoveDiagnosis]:
    moves = [move for move in funnel.moves if not move.failed]
    if not moves:
        return {}

    # the replayer is shared per game and so is the refutation memo, for the same reason: both are built
    # from the whole solved tree, and every cell of a game asks about the same positions
    replayer, refute = replayers(funnel.game), solver_for(funnel.game.to_string())
    tracked = track(moves, f"Labelling {funnel.game.key}", len(moves), progress)
    return {move.uid: _diagnose(MovePosition.from_probe(replayer, move), funnel.game.key, refute) for move in tracked}
