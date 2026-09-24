"""The unseen-species estimators. They answer "how much is missing" from the shape of the counts alone,
which is the question the saturation curve cannot answer: a run stops finding codes both when the taxonomy
is complete and when the batches went quiet."""

from plybench.analysis.errors.reasoning.coverage import Coverage


def test_nothing_coded_estimates_nothing_rather_than_zero():
    empty = Coverage.of({})
    assert empty.unseen_mass is None and empty.chao1 is None  # "no evidence" is not "no gap"
    assert "nothing can be estimated" in empty.summary()


def test_unseen_mass_is_the_share_of_instances_sitting_in_codes_seen_once():
    # 3 singletons out of 30 instances: Good-Turing says a tenth of the error mass belongs to codes the
    # sample never produced at all
    spectrum = {"a": 20, "b": 7, "c": 1, "d": 1, "e": 1}
    assert Coverage.of(spectrum).unseen_mass == 3 / 30


def test_a_taxonomy_whose_codes_all_recur_is_the_one_that_looks_complete():
    recurring = Coverage.of({"a": 20, "b": 15, "c": 12})
    ragged = Coverage.of({"a": 20, "b": 15, "c": 1, "d": 1, "e": 1, "f": 1})

    assert recurring.unseen_mass == 0.0 and recurring.missing_codes == 0.0
    assert ragged.unseen_mass is not None and ragged.missing_codes is not None
    assert ragged.unseen_mass > 0.05 and ragged.missing_codes > 0  # singletons are what say to keep looking


def test_chao1_lower_bounds_the_number_of_types_including_the_unseen_ones():
    # f1=4, f2=2 -> 5 + 16/4 = 9, so the spectrum says about four more types exist than were observed
    assert Coverage.of({"a": 9, "b": 2, "c": 1, "d": 1, "e": 1, "f": 1, "g": 2}).n_codes == 7
    coverage = Coverage.of({"a": 9, "b": 2, "c": 1, "d": 1, "e": 1, "f": 1})
    assert coverage.f1 == 4 and coverage.f2 == 1
    assert coverage.chao1 == 6 + 16 / 2


def test_chao1_falls_back_when_no_code_was_seen_exactly_twice():
    # the classic estimator divides by f2; the bias-corrected form is used instead of crashing or lying
    coverage = Coverage.of({"a": 9, "b": 1, "c": 1, "d": 1})
    assert coverage.f2 == 0 and coverage.chao1 == 4 + 3 * 2 / 2
