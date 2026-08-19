from plybench.analysis.errors.moves import MatchupId, OutputFailure, TracedMove
from plybench.analysis.errors.split import AnalysisSplit, split_for
from plybench.analysis.stats.moves import MoveRecord
from plybench.common.enums import StateClass


def _move(round_: int, seq: int = 1, move: str = "<A1>") -> TracedMove:
    matchup = MatchupId("exp", "tic_tac_toe:", "llm:player", "random:")
    record = MoveRecord(StateClass.DECISION, True, 0.0, None, 3, None, 2, 1)
    return TracedMove(matchup, round_, seq, record, "reasoning", "board", move, ("<A1>", "<A2>"), ("<A1>",))


def test_complete_games_stay_in_one_stable_split():
    first, later = _move(7, 1), _move(7, 9)
    assert split_for(first) == split_for(later)
    assert split_for(first) == split_for(first)
    assert split_for(first) in (AnalysisSplit.DISCOVERY, AnalysisSplit.EVALUATION)


def test_evaluation_is_the_large_partition():
    splits = [split_for(_move(round_)) for round_ in range(1000)]
    share = splits.count(AnalysisSplit.EVALUATION) / len(splits)
    assert 0.75 < share < 0.85


def test_output_failures_are_classified_separately_from_strategy():
    assert _move(1, move="FAIL: Wrong action format (hello)").output_failure == OutputFailure.MALFORMED
    assert _move(1, move="FAIL: Illegal action selected (<Z9>)").output_failure == OutputFailure.ILLEGAL
    assert _move(1).output_failure is None
