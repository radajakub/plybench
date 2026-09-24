"""The procedural move labels: what the solved game tree can say about a move without asking a model.
Positions are built by playing real moves into a real engine and the verdict is stated by hand, so each
test pins one detector against a position whose right answer is clear by inspection rather than against
minimax -- otherwise the detectors would only be tested against the solver they are meant to interpret.

Stating the verdict is also what lets a position be declared already lost, which is where these labels do
their least obvious work: the tree rates every losing move equal, so the counterfactual each detector
searches for is the only thing left that can separate them."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pyspiel as sp
import pytest

from plybench.analysis.errors.funnel.run import build_funnel
from plybench.analysis.errors.procedural.detection import SEVERITY_LABELS, MoveLabel, detect, diagnostics_for, nim_sum, piles
from plybench.analysis.errors.procedural.position import MovePosition
from plybench.analysis.errors.procedural.refutation import Refutation, RefutationSolver
from plybench.analysis.errors.procedural.run import label_moves
from plybench.analysis.replay import ReplayerCache, build_replayer
from plybench.app import PlyBench
from plybench.common.enums import GameResults, StateClass
from plybench.core.minimax import AVQ, depth_limited_value
from plybench.harness.benchmark.benchmark import Benchmark
from plybench.llm import LLMConfig

op = PlyBench(LLMConfig())

WIN, DRAW, LOSS = 1.0, 0.0, -1.0


def _state(game_key: str, actions: Sequence[int] = ()) -> sp.State:
    engine = op.registry.build_engine(op.registry.game_config(game_key))
    engine.reset()
    state = engine.game.state
    for action in actions:
        state.apply_action(action)
    return state


def _position(state: sp.State, chosen: int, optimal: list[int], value: float = WIN) -> MovePosition:
    """A stated verdict: `optimal` keeps `value`, everything else loses. Enough for every detector, since
    none of them reads a Q-value it was not given."""
    values = {action: (value if action in optimal else LOSS) for action in state.legal_actions()}
    return MovePosition(state, chosen, AVQ(optimal, value, values), WIN, LOSS)


# tic-tac-toe actions are row-major cells 0..8, x moves first
def _ttt(moves: Sequence[int]) -> sp.State:
    return _state("tic_tac_toe:", moves)


def _nim_down_to(last_pile_take: int) -> sp.State:
    """1 3 5 7 emptied down to whatever is left of the last pile."""
    state = _state("nim:")
    for pile, count in ((0, 1), (1, 3), (2, 5), (3, last_pile_take)):
        state.apply_action(_take(state, pile, count))
    return state


def _take(state: sp.State, pile: int, count: int) -> int:
    wanted = f"pile:{pile + 1}, take:{count};"
    action = next((a for a in state.legal_actions() if state.action_to_string(state.current_player(), a) == wanted), None)
    assert action is not None, f"no such nim action: {wanted}"
    return action


# --- reading the tree ----------------------------------------------------------------------------
def test_the_analysed_player_is_whoever_was_on_turn():
    # naming the other player would apply this player's action to them and read every outcome off the
    # wrong side of the board, so the position derives it instead of being told
    state = _ttt([0, 3, 1, 4])  # x holds 0,1 and can finish at 2; o holds 3,4 and threatens 5
    assert _position(state, 2, [2]).player == state.current_player() == 0


def test_outcomes_are_always_read_from_the_analysed_players_side():
    state = _ttt([0, 3, 1, 4])
    winning = _position(state, 2, [2])
    assert winning.outcome_before is GameResults.WIN
    assert winning.terminal_outcome(winning.after) is GameResults.WIN
    assert winning.wins_now(state, winning.player)

    # the same reading, from the other side: after x plays 6, o completes 3-4-5 -- an opponent win, which
    # is this player's loss, and the detector has to see it as such
    ignored = _position(state, 6, [2])
    assert ignored.wins_now(ignored.after, ignored.opponent)


def test_a_lost_position_is_still_graded_because_the_tree_gave_up_and_the_board_did_not():
    state = _ttt([4, 0, 8, 1])  # o holds 0,1 and threatens 2; x holds 4,8 and can block there
    # stated as already lost: every legal move keeps the value, which is what a decided position looks like
    # to the solver -- it rates blocking and not blocking alike, and the board does not
    lost = MovePosition(state, 6, AVQ(list(state.legal_actions()), LOSS, dict.fromkeys(state.legal_actions(), LOSS)), WIN, LOSS)
    labels = detect(lost, "tic_tac_toe")
    assert MoveLabel.MISSED_BLOCK in labels
    assert not labels & SEVERITY_LABELS  # nothing was thrown: the value was already gone
    assert MoveLabel.DEEP_ERROR not in labels  # and the solver agreed with the move, so no judge is owed one


# --- universal diagnostics -----------------------------------------------------------------------
def test_missed_forced_win_is_the_win_that_was_left_on_the_board():
    state = _ttt([0, 3, 1, 4])  # x can finish at 2
    assert MoveLabel.MISSED_FORCED_WIN in detect(_position(state, 6, [2]), "tic_tac_toe")
    assert MoveLabel.MISSED_FORCED_WIN not in detect(_position(state, 2, [2]), "tic_tac_toe")


def test_missed_block_is_the_move_that_left_the_opponent_a_win_next_turn():
    state = _ttt([4, 0, 8, 1])  # o holds 0,1 and threatens 2; x holds 4,8
    assert MoveLabel.MISSED_BLOCK not in detect(_position(state, 2, [2]), "tic_tac_toe")
    assert MoveLabel.MISSED_BLOCK in detect(_position(state, 6, [2]), "tic_tac_toe")


def test_self_destruct_is_the_move_that_ended_the_game_where_playing_on_was_possible():
    state = _nim_down_to(5)  # 0 0 0 2: take both and the game is over, take one and the opponent must move
    both, one = _take(state, 3, 2), _take(state, 3, 1)
    labels = detect(_position(state, both, [one]), "nim")
    assert MoveLabel.SELF_DESTRUCT in labels and MoveLabel.THREW_WIN in labels
    assert MoveLabel.SELF_DESTRUCT not in detect(_position(state, one, [one]), "nim")


def test_a_forced_win_counts_when_the_opponent_is_the_one_who_has_to_end_it():
    # misere: whoever takes the last match loses, so no move of ours ever ends the game as a win and a
    # class that only looks at our own terminal moves is blind for the whole family. Leaving the last
    # match is the win here, and not leaving it is the blunder
    state = _nim_down_to(5)  # 0 0 0 2
    both, one = _take(state, 3, 2), _take(state, 3, 1)
    assert MoveLabel.MISSED_FORCED_WIN in detect(_position(state, both, [one]), "nim")
    assert MoveLabel.MISSED_FORCED_WIN not in detect(_position(state, one, [one]), "nim")


def test_a_move_with_no_alternative_is_never_blamed_for_where_it_led():
    state = _nim_down_to(6)  # 0 0 0 1: the last match, and it must be taken
    forced = _take(state, 3, 1)
    # the counterfactual is searched rather than assumed, so a position offering nothing else to play
    # cannot produce a class that says something else should have been played
    assert detect(_position(state, forced, [forced], value=LOSS), "nim") == frozenset({MoveLabel.CLEAN})


# --- severity, and the residual it must not hide --------------------------------------------------
def test_severity_labels_grade_the_drop_and_never_stand_in_for_a_diagnosis():
    state = _ttt([0, 4, 1, 8])  # x holds 0,1 and wins at 2; o holds 4,8
    assert SEVERITY_LABELS == {MoveLabel.THREW_WIN, MoveLabel.THREW_DRAW}
    assert MoveLabel.THREW_WIN in detect(_position(state, 6, [2], value=WIN), "tic_tac_toe")

    drawn = detect(_position(state, 6, [2], value=DRAW), "tic_tac_toe")
    assert MoveLabel.THREW_DRAW in drawn and MoveLabel.THREW_WIN not in drawn


def test_every_graded_move_carries_a_label_and_the_residual_is_told_from_the_clean():
    state = _ttt([4])  # o to move on an open board: nothing tactical is decided this early
    deep = detect(_position(state, 1, [0], value=DRAW), "tic_tac_toe")
    # suboptimal with no shallow class firing -> the residual, which is exactly what a judge is for
    assert deep - SEVERITY_LABELS == {MoveLabel.DEEP_ERROR}

    # the same silence on a move the solver agreed with is a result, not a gap: labelling it deep_error
    # would hand the judge every quiet move in the experiment
    assert detect(_position(state, 0, [0], value=DRAW), "tic_tac_toe") == frozenset({MoveLabel.CLEAN})


# --- per-game detectors --------------------------------------------------------------------------
def test_each_family_gets_its_own_detectors_on_top_of_the_universal_ones():
    line = {code for code, _ in diagnostics_for("story_magic_square")}
    nim = {code for code, _ in diagnostics_for("nim")}
    universal = {code for code, _ in diagnostics_for("connect_four")}  # no family registered
    assert line - universal == {MoveLabel.MISSED_FORK, MoveLabel.ALLOWED_FORK}
    assert nim - universal == {MoveLabel.NIM_SUM_IGNORED, MoveLabel.MISERE_RULE_IGNORED}
    assert universal == {MoveLabel.MISSED_FORCED_WIN, MoveLabel.MISSED_BLOCK, MoveLabel.SELF_DESTRUCT}


def test_missed_fork_is_a_forced_win_in_three_that_was_available():
    # x holds 0 and 4, o holds 1 and 8: playing 6 leaves two winning lines and o can block only one
    state = _ttt([0, 1, 4, 8])
    forking, other = 6, 2
    assert MoveLabel.MISSED_FORK in detect(_position(state, other, [forking]), "tic_tac_toe")
    assert MoveLabel.MISSED_FORK not in detect(_position(state, forking, [forking]), "tic_tac_toe")


def test_allowed_fork_is_the_same_threat_handed_to_the_opponent():
    # o to move against x on opposite corners: an edge holds, a third corner leaves x two lines to finish
    state = _ttt([0, 4, 8])
    assert MoveLabel.ALLOWED_FORK in detect(_position(state, 2, [1]), "tic_tac_toe")
    assert MoveLabel.ALLOWED_FORK not in detect(_position(state, 1, [1]), "tic_tac_toe")


def test_a_threat_no_move_could_have_denied_is_not_blamed_on_the_move_played():
    # x holds a corner and the centre, o holds one edge: every o move leaves x a fork somewhere, so the
    # fork is the shape of the position rather than the consequence of this move
    state = _ttt([0, 1, 4])
    assert all(MoveLabel.ALLOWED_FORK not in detect(_position(state, action, [8]), "tic_tac_toe") for action in state.legal_actions())


def test_nim_sum_ignored_fires_where_misere_and_normal_play_agree():
    state = _state("nim:")
    state.apply_action(_take(state, 3, 1))  # 1 3 5 6, nim-sum 1: a zeroing move exists
    assert piles(state) == [1, 3, 5, 6] and nim_sum(piles(state)) == 1

    zeroing = _take(state, 0, 1)  # -> 0 3 5 6, nim-sum 0
    leaves_it = _take(state, 3, 1)  # -> 1 3 5 5, nim-sum 2
    assert MoveLabel.NIM_SUM_IGNORED not in detect(_position(state, zeroing, [zeroing]), "nim")
    assert MoveLabel.NIM_SUM_IGNORED in detect(_position(state, leaves_it, [zeroing]), "nim")


def test_leaving_the_nim_sum_at_zero_inside_the_endgame_is_the_misere_confusion():
    state = _state("nim:")
    for pile, count in ((1, 1), (2, 5), (3, 7)):  # 1 3 5 7 -> 1 2 0 0
        state.apply_action(_take(state, pile, count))
    sizes = piles(state)
    assert sizes == [1, 2, 0, 0] and sum(size >= 2 for size in sizes) == 1  # the endgame, where the rule flips

    # misere wants an odd number of single piles left, so emptying the big pile wins; the nim-sum rule says
    # leave 1 1, which is precisely the move that hands the last match back
    winning, normal_play = _take(state, 1, 2), _take(state, 1, 1)
    assert MoveLabel.MISERE_RULE_IGNORED not in detect(_position(state, winning, [winning]), "nim")
    confused = detect(_position(state, normal_play, [winning]), "nim")
    assert MoveLabel.MISERE_RULE_IGNORED in confused and MoveLabel.NIM_SUM_IGNORED not in confused

    # outside the endgame the two rules agree, so the same shape is not this confusion
    early = _state("nim:")
    opening = _take(early, 0, 1)
    assert MoveLabel.MISERE_RULE_IGNORED not in detect(_position(early, opening, [opening]), "nim")


# --- how hard the blunder was to see ----------------------------------------------------------------
def _refute(position: MovePosition):
    return RefutationSolver()(position)


def _refuted(position: MovePosition) -> Refutation:
    """The refutation, for the cases that require one. A move the solver agreed with has nothing to
    refute, so the solver may return None; asserting it here keeps that distinction in the test that is
    about it rather than repeating a narrowing check in every other test."""
    refutation = _refute(position)
    assert refutation is not None
    return refutation


def test_a_move_the_solver_agreed_with_has_nothing_to_refute():
    state = _ttt([0, 3, 1, 4])  # x holds 0,1 and finishes at 2
    assert _refute(_position(state, 2, [2])) is None


def test_a_blunder_the_move_itself_settles_is_refuted_at_one_ply():
    # x can finish at 2 and plays elsewhere: one ply of lookahead already separates the two, because the
    # alternative ends the game and nothing else within that horizon does
    state = _ttt([0, 3, 1, 4])
    assert _refuted(_position(state, 6, [2])).depth == 1

    # and the same from the other direction: the move played is the one that ends the game, as a loss
    nim = _nim_down_to(5)  # 0 0 0 2 left, and taking both hands over the last match
    both, one = _take(nim, 3, 2), _take(nim, 3, 1)
    assert _refuted(_position(nim, both, [one])).depth == 1


def test_a_block_that_was_not_played_needs_the_opponents_reply_to_show_it():
    # o holds 0,1 and finishes at 2; x holds 4,8 and can only block. Playing anything else looks no worse
    # than blocking until the opponent's answer is on the board, which is the second ply
    state = _ttt([4, 0, 8, 1])
    refutation = _refuted(_position(state, 6, [2], value=DRAW))
    assert refutation.depth == 2


def test_width_is_the_share_of_replies_that_punish():
    # after x abandons the block at 2, o has four replies and exactly one of them wins; the other three
    # hand x the fork it just built on 2-4-6 and 6-7-8
    state = _ttt([4, 0, 8, 1])
    refutation = _refuted(_position(state, 6, [2], value=DRAW))
    assert (refutation.n_refuting, refutation.n_replies) == (1, 4)
    assert refutation.width == 0.25
    assert refutation.n_better == 1  # the block, and nothing else, was worth more

    # a move that ends the game leaves no reply to punish it, so there is no share to take
    nim = _nim_down_to(5)
    both, one = _take(nim, 3, 2), _take(nim, 3, 1)
    ended = _refuted(_position(nim, both, [one]))
    assert (ended.n_replies, ended.width) == (0, None)


# --- against the real solver ----------------------------------------------------------------------
def test_the_invariants_hold_against_real_minimax_verdicts(tmp_path, monkeypatch):
    """The labels are argued for on paper, and a stated verdict can be stated incoherently, so the
    arguments are checked here against the real solver on every position a real game reached."""
    monkeypatch.chdir(tmp_path)
    benchmark = Benchmark("exp", op, ["tic_tac_toe:"], ["random:distribution=uniform"], ["optimal:stochastic=True"], 6)
    results = asyncio.run(benchmark.run(sync=True, concurrency=1))

    game, player = results.game_configs[0], results.player_configs[0]
    funnel = build_funnel(results, game, player, op.registry, progress=False)
    graded = [move for move in funnel.moves if not move.failed]
    assert any(move.record.state_class == StateClass.DECISION for move in graded), "a random player must reach some real decisions"

    replayer, refute = build_replayer(op.registry, game), RefutationSolver()
    memo, movers = {}, set()
    for move in graded:
        position = MovePosition.from_probe(replayer, move)
        labels, refutation = detect(position, game.key), refute(position)
        assert labels, "every graded move carries at least one label"

        # the search reports player-0 units while the AVQ stores the mover's, so converting between them is
        # the one place a sign error could hide. At the full horizon nothing is cut off, so the two have to
        # agree exactly -- and agree for whoever moves second, which is what the mover set below checks
        horizon = position.state.get_game().max_game_length()
        movers.add(position.player)
        as_mover = 1 if position.player == 0 else -1
        assert as_mover * depth_limited_value(position.state, horizon, memo) == position.verdict.V()
        for action in position.state.legal_actions():
            exact = depth_limited_value(position.child(action), horizon - 1, memo)
            assert as_mover * exact == position.verdict.Q(action)
        assert (labels == {MoveLabel.CLEAN}) == (labels == labels & {MoveLabel.CLEAN}), "clean is never carried alongside anything"
        assert bool(labels & SEVERITY_LABELS) is not move.record.is_optimal, "a drop in value is exactly what the solver disagreed with"
        assert MoveLabel.DEEP_ERROR not in labels or not move.record.is_optimal, "the residual is a blunder nothing explained"
        if move.record.is_optimal and move.record.state_class == StateClass.DECISION:
            # in a decision the position value is above a loss, so an optimal move cannot leave the
            # opponent a win -- the counterfactual the prevention classes search for does not exist
            assert not labels & {MoveLabel.MISSED_BLOCK, MoveLabel.ALLOWED_FORK, MoveLabel.NIM_SUM_IGNORED, MoveLabel.MISERE_RULE_IGNORED}

        assert (refutation is None) is move.record.is_optimal, "a refutation is owed for exactly the blunders"
        if refutation is None:
            continue
        assert refutation.depth >= 1 and refutation.n_better >= 1
        assert refutation.width is None or 0.0 < refutation.width <= 1.0
        # the depth is what says whether a detector really caught the shallow pattern it names: a move
        # that ends the game is settled in one ply, and a missed block in two
        if MoveLabel.SELF_DESTRUCT in labels:
            assert refutation.depth == 1
        if MoveLabel.MISSED_BLOCK in labels:
            assert refutation.depth <= 2
        # and the residual is the other side of the same statement: a one-ply separation is a terminal one,
        # which self_destruct and missed_forced_win already share out between them. Two is not excluded,
        # because in a misere game the win arrives on the opponent's forced move and no universal class
        # names that shape yet
        if MoveLabel.DEEP_ERROR in labels:
            assert refutation.depth >= 2

    assert movers == {0, 1}, "the sign only matters for the second player, so both have to appear"

    # the bots leave no trace, and the labels do not care: what a move is graded on is the board it was
    # played on, which is what makes a bot cell the baseline an LLM cell is read against
    assert set(label_moves(funnel, ReplayerCache(op.registry), progress=False)) == {move.uid for move in graded}


if __name__ == "__main__":
    pytest.main([__file__])
