from plybench.analysis.errors.moves import MatchupId, OutputFailure, TracedMove
from plybench.analysis.stats.moves import MoveRecord
from plybench.common.enums import StateClass


def _move(round_: int, seq: int = 1, move: str = "<A1>", opponent: str = "random:") -> TracedMove:
    matchup = MatchupId("exp", "tic_tac_toe:", "llm:player", opponent)
    record = MoveRecord(StateClass.DECISION, True, 0.0, None, 3, None, 2, 1)
    return TracedMove(matchup, round_, seq, record, "reasoning", "board", move, ("<A1>", "<A2>"), ("<A1>",))


def test_output_failures_are_classified_separately_from_strategy():
    assert _move(1, move="FAIL: Wrong action format (hello)").output_failure == OutputFailure.MALFORMED
    assert _move(1, move="FAIL: Illegal action selected (<Z9>)").output_failure == OutputFailure.ILLEGAL
    assert _move(1).output_failure is None


def test_every_move_of_one_recorded_game_shares_a_game_uid():
    # the hold-out is game-level because these two positions are one move apart and nearly identical
    first, later = _move(7, seq=1), _move(7, seq=9)
    assert first.game_uid == later.game_uid
    assert first.uid != later.uid


def test_a_game_uid_separates_rounds_matchups_and_opponents():
    assert _move(7).game_uid != _move(8).game_uid
    assert _move(7).game_uid != _move(7, opponent="optimal:").game_uid


def test_the_move_uid_is_not_derived_from_the_game_uid():
    # every stored annotation is keyed by move uid, so deriving one from the other would make a future
    # change to game identity silently orphan the annotations already collected
    move = _move(7, seq=3)
    assert move.game_uid not in move.uid
