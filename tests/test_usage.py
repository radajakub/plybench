"""Token accounting over recorded results: summing every recorded call of an experiment, grouping the
sums, and pricing them. The point of these tests is what gets counted -- both sides of a matchup, no
double count in self-play, and no bot step folded in as a zero."""

from __future__ import annotations

from plybench.analysis.usage import benchmark_usage, entry_cost, game_usage, group_by, matchup_usage, model_label, player_model, total_cost, total_usage
from plybench.app import PlyBench
from plybench.common.paths import BenchmarkPathBuilder
from plybench.harness.benchmark.results import BenchmarkResults
from plybench.llm import LLMConfig, LLMTokens
from plybench.llm.llm_config import OpenAIProviderConfig
from plybench.trackers.game_tracker import GameStep, GameTracker
from plybench.trackers.result_tracker import ResultTracker

# a dummy key is enough: the catalogue (and with it the price list) is built without any network call
op = PlyBench(LLMConfig(openai=OpenAIProviderConfig(api_key="test-key")))
registry = op.registry
PATHS = BenchmarkPathBuilder()

MODEL = "llm:actions:text:openai:gpt-5-nano:thinking_enabled=True,reasoning_effort=high"
OTHER_MODEL = "llm:actions:text:openai:gpt-5.4:thinking_enabled=True,reasoning_effort=high"
BOT = "random:distribution=uniform"


def _step(seq, player, tokens=None):
    if tokens is None:  # a bot step: no token fields at all
        return GameStep(seq, player.to_string(), player.hash, "", "obs", "a1")
    input_tokens, output_tokens, reasoning_tokens = tokens
    return GameStep(seq, player.to_string(), player.hash, "", "obs", "a1", input_tokens=input_tokens, output_tokens=output_tokens, reasoning_tokens=reasoning_tokens)


def _game(i, o, i_tokens, o_tokens=None):
    steps = [_step(seq, i, tokens) for seq, tokens in enumerate(i_tokens)]
    steps += [_step(len(steps) + seq, o, tokens) for seq, tokens in enumerate(o_tokens or [None] * len(i_tokens))]
    return GameTracker(1, i, o, {}, steps, None, len(steps), {})


def _result_tracker(i_string, o_string, games):
    i, o = registry.player_config(i_string), registry.player_config(o_string)
    game = registry.game_config("tic_tac_toe:")
    trackers = [_game(i, o, *game_tokens) for game_tokens in games]
    return ResultTracker("exp", i, o, game, len(trackers), set(), registry, PATHS, games=list(trackers), save_on_record=False)


# --- what a single side sums to --------------------------------------------------------------
def test_game_usage_sums_the_players_calls_and_ignores_the_opponents():
    i, o = registry.player_config(MODEL), registry.player_config(BOT)
    usage = game_usage(_game(i, o, [(100, 20, 10), (150, 30, 15)]), i)

    assert usage.calls == 2
    assert usage.tokens == LLMTokens(input_tokens=250, output_tokens=50, reasoning_tokens=25)
    # reasoning is a subset of output, so the billed total adds input and output only
    assert usage.total == 300


def test_a_bots_moves_are_not_counted_as_zero_token_calls():
    i, o = registry.player_config(MODEL), registry.player_config(BOT)
    assert game_usage(_game(i, o, [(100, 20, 10)]), o).calls == 0


# --- what a matchup sums to ------------------------------------------------------------------
def test_matchup_usage_counts_both_sides_when_both_are_models():
    tracker = _result_tracker(MODEL, OTHER_MODEL, [([(100, 20, 10)], [(200, 40, 20)])])
    entries = matchup_usage(tracker)

    assert len(entries) == 2
    by_model = {}
    for entry in entries:
        assert entry.model is not None
        by_model[entry.model.model_name] = entry.usage
    assert by_model["gpt-5-nano"].tokens.input_tokens == 100
    assert by_model["gpt-5.4"].tokens.input_tokens == 200
    # each side is recorded against the player it faced
    assert {model.model_name for entry in entries if (model := player_model(entry.opponent)) is not None} == {"gpt-5-nano", "gpt-5.4"}


def test_a_side_without_calls_is_dropped_rather_than_reported_empty():
    entries = matchup_usage(_result_tracker(MODEL, BOT, [([(100, 20, 10)],)]))
    assert len(entries) == 1
    model = player_model(entries[0].player)
    assert model is not None and model.model_name == "gpt-5-nano"


def test_self_play_is_counted_once_not_once_per_side():
    # both sides share a config (and therefore a hash), so one pass over the steps already has both
    tracker = _result_tracker(MODEL, MODEL, [([(100, 20, 10)], [(200, 40, 20)])])
    (entry,) = matchup_usage(tracker)
    assert entry.usage.calls == 2 and entry.usage.tokens.input_tokens == 300


def test_unplayed_games_contribute_nothing():
    tracker = _result_tracker(MODEL, BOT, [([(100, 20, 10)],)])
    tracker.games.append(None)  # a matchup that was never finished carries empty slots
    (entry,) = matchup_usage(tracker)
    assert entry.usage.calls == 1


# --- aggregation and pricing -----------------------------------------------------------------
def _results(*trackers):
    return BenchmarkResults([registry.game_config("tic_tac_toe:")], [], [], list(trackers))


def test_grouping_folds_the_same_model_across_matchups_into_one_row():
    usages = benchmark_usage(
        _results(
            _result_tracker(MODEL, BOT, [([(100, 20, 10)],)]),
            _result_tracker(MODEL, OTHER_MODEL, [([(50, 10, 5)], [(300, 60, 30)])]),
        ),
        progress=False,
    )
    grouped = {label: total_usage(entries) for label, entries in group_by(usages, lambda usage: model_label(usage.model)).items()}

    nano = grouped[model_label(player_model(registry.player_config(MODEL)))]
    assert nano.calls == 2 and nano.tokens.input_tokens == 150
    assert total_usage(usages).total == sum(usage.total for usage in grouped.values())


def test_cost_uses_the_model_catalogue_and_is_undefined_for_non_models():
    (entry,) = matchup_usage(_result_tracker(MODEL, BOT, [([(1_000_000, 1_000_000, 0)],)]))
    assert entry.model is not None
    model = op.llm.resolve_model(entry.model.provider, entry.model.model_name)

    assert entry_cost(op.llm, entry) == model.input_cost + model.output_cost
    # a matchup side that is not a model has no price, and the total says how many such sides there were
    cost, unpriced = total_cost(op.llm, [entry, entry.__class__(entry.experiment, entry.game, entry.player, entry.opponent, None, entry.usage)])
    assert cost == entry_cost(op.llm, entry) and unpriced == 1
