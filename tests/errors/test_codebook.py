"""The codebook and the annotation store: the artefacts LLM annotation writes into. Both are built so a
label made months ago still resolves after the taxonomy has been renamed, merged and re-versioned."""

from __future__ import annotations

import pytest

from plybench.analysis.errors.judge.sampling import by_matchup_outcome, discovery_plan, interleave, sample, strata
from plybench.analysis.errors.moves import MatchupId, TracedMove
from plybench.analysis.errors.reasoning.annotations import Annotation, AnnotationStore, MistakeLabel
from plybench.analysis.errors.reasoning.codebook import SCOPE_UNIVERSAL, Code, Codebook, Etalon, game_scope
from plybench.analysis.errors.reasoning.protocol import CODING_RULES, rules_block
from plybench.analysis.errors.stores import AnalysisStores
from plybench.analysis.stats.moves import MoveRecord
from plybench.common.enums import StateClass

MATCHUP = MatchupId("exp", "tic_tac_toe:", "llm:player", "random:")


def _traced(seq, is_optimal=True, state_class=StateClass.DECISION, game=None, player=None, opponent=None):
    record = MoveRecord(state_class, is_optimal, 0.0, None, 100, None, 3, 1)
    matchup = MatchupId(
        MATCHUP.experiment,
        game or MATCHUP.game,
        player or MATCHUP.player,
        opponent or MATCHUP.opponent,
    )
    return TracedMove(matchup, 1, seq, record, "reasoning...", "board", "<A1>", ("<A1>", "<A2>"), ("<A1>",))


def _book(tmp_path):
    book = Codebook("exp")
    book.add(Code("threat_blindness", "Missed an immediate threat", "Did not check what the opponent threatens next"))
    book.add(Code("nim_sum", "Nim-sum miscomputed", "Arithmetic on the xor of pile sizes is wrong", game_scope("nim"), "threat_blindness"))
    return book


# --- the codebook ------------------------------------------------------------------------------
def test_codes_carry_a_tier_and_a_scope(tmp_path):
    book = _book(tmp_path)
    assert [code.id for code in book.spine()] == ["threat_blindness"]  # the universal tier
    assert [code.id for code in book.children("threat_blindness")] == ["nim_sum"]
    assert book.codes["threat_blindness"].scope == SCOPE_UNIVERSAL and book.codes["nim_sum"].scope == "game:nim"
    with pytest.raises(ValueError):
        book.add(Code("orphan", "x", "y", SCOPE_UNIVERSAL, "does_not_exist"))
    with pytest.raises(ValueError):
        book.add(Code("nim_sum", "duplicate", "y"))  # ids are assigned once and never reused


def test_only_codes_applicable_to_the_move_are_exposed():
    book = _book(None)
    nim = _traced(1, game="nim:")
    ttt = _traced(2, game="tic_tac_toe:")
    assert {code.id for code in book.applicable(nim)} == {"threat_blindness", "nim_sum"}
    assert {code.id for code in book.applicable(ttt)} == {"threat_blindness"}


def test_merging_retires_a_code_without_orphaning_its_labels(tmp_path):
    book = _book(tmp_path)
    book.add(Code("xor_error", "XOR slip", "Same failure, different wording", game_scope("nim"), "threat_blindness", ("uid-1",)))
    book.merge("xor_error", "nim_sum")

    # a label written against the retired id still counts, under the code it was merged into
    assert book.resolve("xor_error").id == "nim_sum"
    assert "uid-1" in book.resolve("xor_error").examples  # provenance follows the merge
    assert [code.id for code in book.active()] == ["threat_blindness", "nim_sum"]
    with pytest.raises(ValueError):
        book.merge("nim_sum", "xor_error")  # would close a cycle


def test_codebook_round_trips_and_versions_only_its_definitions(tmp_path):
    book = _book(tmp_path)
    book.discovery_design = {"strata": "game,player,opponent,outcome", "suboptimal_cap": 32}
    path = book.save(tmp_path / "codebook.json")
    reloaded = Codebook.load("exp", path)
    assert reloaded.version == book.version and set(reloaded.codes) == set(book.codes)
    assert reloaded.discovery_design == book.discovery_design

    # the version tracks what a label means, so wording changes bump it and provenance does not
    before = reloaded.version
    reloaded.codes["nim_sum"] = Code("nim_sum", "Nim-sum miscomputed", "sharpened definition", game_scope("nim"), "threat_blindness")
    assert reloaded.version != before

    # a missing file is an empty codebook, not an error: nothing has been induced yet
    assert Codebook.load("exp", tmp_path / "absent.json").codes == {}


# --- the annotation store ----------------------------------------------------------------------
def test_store_appends_and_keeps_both_annotators_of_a_double_coded_move(tmp_path):
    store = AnnotationStore("exp", tmp_path / "annotations.jsonl")
    label = MistakeLabel("threat_blindness", "I can just take the last pile", False)
    store.add(Annotation("uid-1", "opus:v1", "abc123", (label,)))
    store.add(Annotation("uid-1", "sonnet:v1", "abc123", ()))
    store.add(Annotation("uid-2", "opus:v1", "abc123", (label,)))

    assert len(store) == 3 and store.double_annotated() == ["uid-1"]  # the reliability subsample
    assert store.annotated_by("opus:v1") == {"uid-1", "uid-2"}  # what a resumed run skips
    assert ("uid-1", "opus:v1") in store and ("uid-2", "sonnet:v1") not in store

    # re-reading the file reproduces the store, so a crashed run keeps everything already paid for
    assert {a.key: a for a in AnnotationStore("exp", store.path)}.keys() == {a.key: a for a in store}.keys()


def test_self_corrected_labels_are_kept_but_excluded_from_uncorrected_mistakes():
    caught = MistakeLabel("nim_sum", "wait, that xor is wrong -- it is 3, not 1", self_corrected=True)
    survived = MistakeLabel("threat_blindness", "they cannot do anything about it", self_corrected=False)
    annotation = Annotation("uid-1", "opus:v1", "abc123", (caught, survived))

    # the protocol keeps the repaired error on the record so recovery can be measured, but the headline
    # mistake rate counts only what survived into the decision
    assert len(annotation.labels) == 2
    assert [label.code_id for label in annotation.uncorrected] == ["threat_blindness"]


def test_coding_rules_are_shared_verbatim_with_the_prompts():
    block = rules_block()
    assert all(rule in block for rule in CODING_RULES)
    assert block.startswith("1. ") and "self_corrected=true" in block  # the self-correction rule survives


# --- sampling ----------------------------------------------------------------------------------
def test_sampling_is_deterministic_and_keeps_the_rare_strata_whole():
    optimal = [_traced(i, is_optimal=True) for i in range(50)]
    suboptimal = [_traced(100 + i, is_optimal=False) for i in range(3)]
    forced = [_traced(200 + i, state_class=StateClass.LOST) for i in range(10)]
    moves = optimal + suboptimal + forced

    assert strata(moves) == {
        ("tic_tac_toe:", "llm:player", "optimal"): 50,
        ("tic_tac_toe:", "llm:player", "suboptimal"): 3,
        ("tic_tac_toe:", "llm:player", "non_decision"): 10,
    }

    picked = sample(moves, per_stratum=5, seed="s1")
    counts = strata(picked)
    assert counts[("tic_tac_toe:", "llm:player", "optimal")] == 5
    assert counts[("tic_tac_toe:", "llm:player", "suboptimal")] == 3  # smaller than the cap -> taken whole

    # same seed -> same moves, on any machine and in any input order; a different seed -> a different draw
    assert [move.uid for move in sample(moves, 5, "s1")] == [move.uid for move in picked]
    assert [move.uid for move in sample(list(reversed(moves)), 5, "s1")] == [move.uid for move in picked]
    assert [move.uid for move in sample(moves, 5, "s2")] != [move.uid for move in picked]


def test_interleaving_spreads_the_rare_strata_over_the_whole_sequence():
    """Induction batches are contiguous slices of this list, and only a slice holding a suboptimal move
    can end the loop. Sampling emits one stratum after another, so without this the first several batches
    are optimal moves and induction saturates having seen no failure at all."""
    moves = [_traced(i, is_optimal=True) for i in range(40)] + [_traced(100 + i, is_optimal=False) for i in range(8)]
    ordered = interleave(moves)

    assert len(ordered) == len(moves) and {move.uid for move in ordered} == {move.uid for move in moves}
    # every batch of 8 holds at least one suboptimal move, where the unordered list puts them all last
    batches = [ordered[start : start + 8] for start in range(0, len(ordered), 8)]
    assert all(any(not move.record.is_optimal for move in batch) for batch in batches)
    assert not any(not move.record.is_optimal for move in moves[:8])  # what it looked like before

    assert [move.uid for move in interleave(moves)] == [move.uid for move in ordered]  # order-stable


def test_discovery_enriches_errors_and_covers_every_matchup_stratum_first():
    weak_errors = [_traced(i, is_optimal=False, player="weak") for i in range(40)]
    strong_errors = [_traced(100 + i, is_optimal=False, player="strong") for i in range(2)]
    controls = [_traced(200 + i, player="weak") for i in range(20)]
    other_opponent = [_traced(300 + i, is_optimal=False, player="weak", opponent="perfect") for i in range(3)]
    moves = weak_errors + strong_errors + controls + other_opponent

    plan = discovery_plan(
        moves,
        suboptimal_cap=5,
        optimal_cap=2,
        non_decision_cap=1,
        coverage_per_stratum=2,
        seed="study",
    )
    counts = strata(plan.moves, key=by_matchup_outcome)

    assert sorted(counts.values()) == [2, 2, 3, 5]  # strong-model errors survive; only the large buckets shrink
    assert plan.candidate_strata == 4 and plan.coverage_moves == 8
    prefix_counts = strata(plan.moves[: plan.coverage_moves], key=by_matchup_outcome)
    assert set(prefix_counts) == set(counts) and set(prefix_counts.values()) == {2}
    assert [move.uid for move in discovery_plan(list(reversed(moves)), suboptimal_cap=5, optimal_cap=2, seed="study").moves] == [
        move.uid for move in discovery_plan(moves, suboptimal_cap=5, optimal_cap=2, seed="study").moves
    ]


# --- etalons -------------------------------------------------------------------------------------
def test_an_etalon_is_provenance_and_never_moves_the_version(tmp_path):
    """Annotations record the codebook version they were made under, so anything that moves it
    invalidates them. An etalon changes nothing about what a code means -- like `examples`, it is a
    pointer at the data -- so electing one must not force a re-annotation of the whole corpus."""
    book = Codebook("exp")
    book.add(Code("threat_blindness", "Missed a threat", "Did not check what the opponent threatens"))
    before = book.version

    book.set_etalon("threat_blindness", Etalon("uid-1", "Therefore the best move is A1.", "clearest instance"))

    assert book.version == before
    assert book.codes["threat_blindness"].etalon is not None
    reloaded = Codebook.load("exp", book.save(tmp_path / "codebook.json"))
    assert reloaded.codes["threat_blindness"].etalon == book.codes["threat_blindness"].etalon


def test_an_etalon_follows_a_merge_to_the_surviving_code():
    book = Codebook("exp")
    book.add(Code("narrow", "Narrow name", "definition"))
    book.add(Code("broad", "Broad name", "definition"))
    book.merge("narrow", "broad")

    book.set_etalon("narrow", Etalon("uid-1", "quote", ""))
    assert book.codes["broad"].etalon is not None, "resolved through the merge, not filed under a retired id"
    assert book.codes["narrow"].etalon is None


# --- provenance and forking ----------------------------------------------------------------------
def test_the_proposing_judge_is_recorded_but_does_not_change_what_a_code_means(tmp_path):
    """Whether a taxonomy is an artifact of whichever model induced it is the one question kappa cannot
    answer -- it measures how consistently judges *apply* a codebook, not whether they would have built
    the same one. That is only answerable if each code remembers who proposed it."""
    book = Codebook("exp")
    before = book.version
    book.add(Code("threat_blindness", "Missed a threat", "definition", inducer="metacentrum:gemma-4|induction:v1"))

    assert book.version != before, "a new code does change the taxonomy"
    versioned = book.version
    book.add(Code("other_code", "Other", "definition", inducer="openai:gpt-5.4|induction:v1"))
    assert {code.inducer for code in book.active()} == {"metacentrum:gemma-4|induction:v1", "openai:gpt-5.4|induction:v1"}
    assert book.version != versioned

    reloaded = Codebook.load("exp", book.save(tmp_path / "codebook.json"))
    assert reloaded.codes["threat_blindness"].inducer == "metacentrum:gemma-4|induction:v1"


def test_a_merge_keeps_both_inducers_because_the_survivor_now_covers_both_proposals():
    book = Codebook("exp")
    book.add(Code("narrow", "Narrow", "definition", inducer="judge-a"))
    book.add(Code("broad", "Broad", "definition", inducer="judge-b"))
    book.merge("narrow", "broad")
    assert book.codes["broad"].inducer == "judge-b, judge-a"


def test_a_labelled_codebook_is_a_separate_file_and_saves_back_to_itself(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main, forked = Codebook.load("exp"), Codebook.load("exp", label="gemma-trial")
    main.add(Code("from_main", "Main", "definition"))
    forked.add(Code("from_fork", "Fork", "definition"))
    main.save()
    forked.save()

    assert Codebook.load("exp").codes.keys() == {"from_main"}, "the fork must not reach the experiment's own book"
    reloaded = Codebook.load("exp", label="gemma-trial")
    assert reloaded.codes.keys() == {"from_fork"} and reloaded.label == "gemma-trial"
    assert (tmp_path / "analysis/exp/reasoning/codebook.gemma-trial.json").exists()


def test_the_stores_hand_out_the_labelled_codebook(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Codebook.load("exp", label="trial").save()
    assert AnalysisStores("trial").codebook("exp").label == "trial"
    assert AnalysisStores().codebook("exp").label == ""


def test_interleave_does_not_order_batches_by_how_the_games_are_named():
    """A live run induced its whole codebook off magic_square: every stratum contributes about one move
    per rank, so the tiebreak decides what a batch contains, and ordering by the stratum sorted it by the
    game config string. magic_square sorts first, so the first twelve batches held nothing else and the
    loop saturated before tic_tac_toe was ever shown."""
    # named so that alphabetical order and the order we want are different
    games = ["aaa_first:", "bbb_second:", "ccc_third:", "zzz_last:"]
    moves = [_traced(seq, game=game) for game in games for seq in range(12)]

    batches = [interleave(moves, seed="s")[start : start + 4] for start in range(0, len(moves), 4)]
    first_batch = {move.matchup.game for move in batches[0]}
    assert len(first_batch) > 1, "a batch drawn from four games must not come from only one of them"

    seen_by = {game: next(i for i, batch in enumerate(batches) if any(m.matchup.game == game for m in batch)) for game in games}
    assert max(seen_by.values()) <= 1, f"every game should appear almost immediately, got {seen_by}"


def test_interleave_is_deterministic_given_the_seed():
    # induction is resumable and its batches must be reproducible, so the shuffle is a hash and not an RNG
    moves = [_traced(seq, game=game) for game in ("a:", "b:") for seq in range(6)]
    assert [m.uid for m in interleave(moves, seed="x")] == [m.uid for m in interleave(moves, seed="x")]
    assert [m.uid for m in interleave(moves, seed="x")] != [m.uid for m in interleave(moves, seed="y")]
