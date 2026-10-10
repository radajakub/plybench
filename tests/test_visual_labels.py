"""Display names and the colour/style encoding for plotted series. Nothing here draws anything --
it is the layer that decides what a tick, a legend entry and a line look like."""

from __future__ import annotations

import pytest
from plyllm import LLMConfig

from plybench.analysis.visual import (
    CATEGORICAL,
    LINESTYLES,
    ColorBy,
    Series,
    StyleEncoder,
    StyleKey,
    encoder_keys,
    game_labels,
    indistinguishable,
    marker_for,
    model_name,
    model_strength,
    player_label,
    player_labels,
    provider_key,
    style_key,
    tier_rank,
)
from plybench.app import PlyBench

op = PlyBench(LLMConfig())

NANO_HIGH = "llm:actions:text:openai:gpt-5-nano:thinking_enabled=True,reasoning_effort=high"
NANO_LOW = "llm:actions:text:openai:gpt-5-nano:thinking_enabled=True,reasoning_effort=low"
OPUS = "llm:actions:text:claude:claude-opus-5:thinking_enabled=True,reasoning_effort=high"


def _player(config_string):
    return op.registry.player_config(config_string)


def _game(config_string):
    return op.registry.game_config(config_string)


# --- labels ------------------------------------------------------------------------------------
def test_same_model_at_different_reasoning_efforts_gets_distinct_labels():
    # an experiment routinely enables one model at three efforts; collapsing them to the bare model
    # name would silently merge three lines in the legend
    assert player_label(_player(NANO_HIGH)) == "gpt-5-nano (high)"
    assert player_label(_player(NANO_LOW)) == "gpt-5-nano (low)"


def test_player_label_can_be_overridden_by_config_string():
    assert player_label(_player(OPUS), {OPUS: "Opus 5"}) == "Opus 5"


def test_non_llm_players_fall_back_to_their_key_and_params():
    assert player_label(_player("random:distribution=uniform")) == "random (distribution=uniform)"
    assert player_label(_player("optimal:stochastic=True")) == "optimal (stochastic=True)"


def test_game_labels_spell_out_only_the_parameters_that_disambiguate():
    labels = game_labels([_game("tic_tac_toe:"), _game("magic_square:sample=False,magic_constant_add=0"), _game("magic_square:sample=False,magic_constant_add=5")])
    # a key used once needs no parameters at all, and the two magic squares differ only in the
    # constant -- sample=False is shared, so printing it would just crowd the axis
    assert labels[0] == "Tic Tac Toe"
    assert labels[1] == "Magic Square\nadd=0"
    assert labels[2] == "Magic Square\nadd=5"


def test_several_differing_parameters_each_get_their_own_shortened_line():
    labels = game_labels([_game("nim:sample=True,num_piles=4,pile_sum=16"), _game("nim:sample=True,num_piles=6,pile_sum=20")])
    # num_piles and pile_sum both vary and shorten to distinct names, so both are shown
    assert labels[0] == "Nim\npiles=4\nsum=16"
    assert labels[1] == "Nim\npiles=6\nsum=20"


# --- tiers -------------------------------------------------------------------------------------
def test_tier_rank_orders_each_providers_ladder_from_small_to_flagship():
    assert tier_rank("gpt-5-nano") < tier_rank("gpt-5-mini") < tier_rank("gpt-5.4")
    assert tier_rank("gemini-3.1-flash-lite") < tier_rank("gemini-3-flash") < tier_rank("gemini-3.1-pro")
    assert tier_rank("claude-haiku-4.5") < tier_rank("claude-sonnet-4.6") < tier_rank("claude-opus-5")
    assert tier_rank("mistral-small-4") < tier_rank("mistral-medium-3.5")


def test_a_model_with_no_size_suffix_is_treated_as_the_full_size_flagship():
    # "gpt-5.4" is the undifferentiated full model, so it must outrank both of its suffixed siblings
    assert tier_rank("gpt-5.4") > tier_rank("gpt-5-mini")
    assert tier_rank("glm-5.2") > tier_rank("gemini-3-flash")


def test_flash_lite_is_ranked_below_flash_even_though_the_name_contains_flash():
    assert tier_rank("gemini-2.5-flash-lite") < tier_rank("gemini-2.5-flash")


# --- encoding ----------------------------------------------------------------------------------
def test_colour_is_keyed_to_the_provider_so_one_hue_covers_a_whole_family():
    assert provider_key(_player(NANO_HIGH)) == "openai"
    assert provider_key(_player(OPUS)) == "claude"


def test_the_stronger_model_of_a_provider_gets_the_more_solid_line():
    players = [
        _player(NANO_HIGH),
        _player("llm:actions:text:openai:gpt-5-mini:thinking_enabled=True,reasoning_effort=high"),
        _player("llm:actions:text:openai:gpt-5.4:thinking_enabled=True"),
    ]
    encoder = StyleEncoder(encoder_keys(players, ColorBy.PROVIDER))
    styles = [encoder.style(style_key(player, ColorBy.PROVIDER)).linestyle for player in players]
    # LINESTYLES runs solid -> sparse, and the flagship takes the first slot
    assert styles == [LINESTYLES[2], LINESTYLES[1], LINESTYLES[0]]


def test_the_line_ladder_compacts_to_the_models_actually_drawn():
    # only the two smallest openai models are plotted, so they must read as solid/dashed rather than
    # keeping the sparse slots they would occupy if gpt-5.4 were present
    drawn = [_player(NANO_HIGH), _player("llm:actions:text:openai:gpt-5-mini:thinking_enabled=True,reasoning_effort=high")]
    roster = drawn + [_player("llm:actions:text:openai:gpt-5.4:thinking_enabled=True")]
    encoder = StyleEncoder(encoder_keys(drawn, ColorBy.PROVIDER), encoder_keys(roster, ColorBy.PROVIDER))
    styles = [encoder.style(style_key(player, ColorBy.PROVIDER)).linestyle for player in drawn]
    assert styles == [LINESTYLES[1], LINESTYLES[0]]


def test_a_narrowed_selection_keeps_its_colour_even_as_the_line_ladder_compacts():
    drawn = [_player(OPUS)]
    roster = [_player(NANO_HIGH), _player(OPUS)]
    encoder = StyleEncoder(encoder_keys(drawn, ColorBy.PROVIDER), encoder_keys(roster, ColorBy.PROVIDER))
    style = encoder.style(style_key(_player(OPUS), ColorBy.PROVIDER))
    # claude is second in the roster, so it keeps the second hue even though openai is not drawn
    assert style.color == CATEGORICAL[1]
    assert style.linestyle == LINESTYLES[0]


def test_the_efforts_of_one_model_share_a_linestyle_and_differ_only_by_marker():
    players = [_player(NANO_HIGH), _player(NANO_LOW)]
    encoder = StyleEncoder(encoder_keys(players, ColorBy.PROVIDER))
    high, low = (encoder.style(style_key(player, ColorBy.PROVIDER)) for player in players)
    # same tier, so the tier channel must not separate them; the marker does
    assert high.linestyle == low.linestyle
    assert high.color == low.color
    assert high.marker != low.marker


def test_efforts_of_one_model_are_reported_as_indistinguishable_once_markers_are_dropped():
    players = [_player(NANO_HIGH), _player(NANO_LOW), _player(OPUS)]
    encoder = StyleEncoder(encoder_keys(players, ColorBy.PROVIDER))
    series = [Series(player, player_label(player), style_key(player, ColorBy.PROVIDER), []) for player in players]
    # the two nano efforts share a colour and a linestyle, so only the marker separates them
    assert indistinguishable(series, encoder, ignore_markers=True) == [["gpt-5-nano (high)", "gpt-5-nano (low)"]]
    # with the markers kept, nothing collides
    assert indistinguishable(series, encoder) == []


def test_two_configs_of_one_model_are_named_and_drawn_apart():
    # same model, same effort, different temperature: the label mentions neither by default, and the
    # legend keys on the label -- so both series would be drawn and only one of them named
    hot = _player(OPUS + ",temperature=1.0")
    cold = _player(OPUS + ",temperature=0.2")
    assert player_labels([hot, cold]) == ["claude-opus-5 (high, temperature=1.0)", "claude-opus-5 (high, temperature=0.2)"]

    encoder = StyleEncoder(encoder_keys([hot, cold], ColorBy.PROVIDER))
    styles = [encoder.style(style_key(player, ColorBy.PROVIDER)) for player in (hot, cold)]
    # no channel is left to separate them but the linestyle: the marker already means effort
    assert styles[0].linestyle != styles[1].linestyle
    assert styles[0].color == styles[1].color


def test_a_parameter_shared_by_every_player_stays_out_of_the_labels():
    # thinking_enabled and the observation/output plumbing are identical across a normal experiment,
    # so spelling them out would crowd every legend entry to no purpose
    assert player_labels([_player(OPUS), _player(NANO_HIGH)]) == ["claude-opus-5 (high)", "gpt-5-nano (high)"]


def test_a_parameter_that_is_not_a_version_never_reaches_the_tier_ladder():
    # "temperature=0.2" parses as a number; ranked as a version it would outrank nothing sanely
    assert tier_rank(model_name(_player(OPUS + ",temperature=0.2"))) == tier_rank("claude-opus-5")
    assert model_strength(_player(OPUS + ",temperature=0.2"))[:2] == model_strength(_player(OPUS))[:2]


def test_a_drawn_colour_group_missing_from_the_roster_is_a_clear_error():
    drawn = encoder_keys([_player(OPUS)], ColorBy.PROVIDER)
    roster = encoder_keys([_player(NANO_HIGH)], ColorBy.PROVIDER)
    with pytest.raises(ValueError, match="missing from the colour roster"):
        StyleEncoder(drawn, roster)


def test_models_of_different_tiers_stay_distinguishable_without_markers():
    players = [_player(NANO_HIGH), _player("llm:actions:text:openai:gpt-5.4:thinking_enabled=True")]
    encoder = StyleEncoder(encoder_keys(players, ColorBy.PROVIDER))
    series = [Series(player, player_label(player), style_key(player, ColorBy.PROVIDER), []) for player in players]
    assert indistinguishable(series, encoder) == []


def test_reasoning_effort_maps_to_a_fixed_marker_so_it_reads_the_same_in_every_figure():
    assert marker_for("low") != marker_for("high") != marker_for(None)
    with pytest.raises(ValueError, match="no marker defined"):
        marker_for("enormous")


def test_filtering_the_selection_does_not_repaint_the_models_that_remain():
    full = [_player(NANO_HIGH), _player(OPUS)]
    encoder = StyleEncoder(encoder_keys(full, ColorBy.PROVIDER))
    # rebuilt from the same full config, as the script does, a dropped openai model must not
    # promote claude into the first slot
    refiltered = StyleEncoder(encoder_keys(full, ColorBy.PROVIDER))
    opus = style_key(_player(OPUS), ColorBy.PROVIDER)
    assert encoder.style(opus).color == refiltered.style(opus).color == CATEGORICAL[1]


def _key(color, line, strength):
    return StyleKey(color, line, None, (strength, (), line))


def test_more_colour_groups_than_palette_slots_is_an_error_rather_than_a_recycled_hue():
    keys = [_key(f"provider-{index}", "model", 0) for index in range(len(CATEGORICAL) + 1)]
    with pytest.raises(ValueError, match="categorical palette"):
        StyleEncoder(keys)


def test_too_many_models_inside_one_colour_group_is_an_error():
    keys = [_key("openai", f"model-{index}", index) for index in range(len(LINESTYLES) + 1)]
    with pytest.raises(ValueError, match="line styles"):
        StyleEncoder(keys)


def test_line_styles_are_ranked_within_a_colour_group_not_across_the_whole_palette():
    # two providers with two models each must not exhaust four line styles between them
    keys = [_key("openai", "a", 1), _key("openai", "b", 0), _key("claude", "c", 1), _key("claude", "d", 0)]
    encoder = StyleEncoder(keys)
    assert encoder.style(keys[0]).linestyle == encoder.style(keys[2]).linestyle == LINESTYLES[0]
    assert encoder.style(keys[0]).color != encoder.style(keys[2]).color
