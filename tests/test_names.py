"""Short names of config strings (plybench.configs.names): what every config of an axis shares is left out, models are named by their name alone, an
experiment's label wins, and two configs never get one name."""

from __future__ import annotations

from plybench.configs.names import item_names, short_names, split_top

MODEL = "gemini:gemini-3.5-flash-lite[thinking_enabled=True]"


def test_split_top_splits_only_outside_brackets():
    assert split_top(f"actor=llm[model={MODEL},budget=600],critic=oracle") == [f"actor=llm[model={MODEL},budget=600]", "critic=oracle"]
    assert split_top("") == []


def test_a_shared_model_and_shared_params_are_left_out():
    players = [f"tac:actor=llm[model={MODEL},memory=False],debug=true", f"tac:actor=llm[model={MODEL},budget=600,apply=compress[model={MODEL}]],critic=oracle,debug=true"]
    names, shared = short_names(players)
    assert names == {players[0]: "actor=llm[memory=False]", players[1]: "actor=llm[budget=600,apply=compress],critic=oracle"}
    assert shared == [f"model={MODEL}", "debug=true"]


def test_different_models_keep_their_names_and_a_label_wins():
    players = ["tac:actor=llm[model=openai:gpt-5-nano]", "tac:actor=llm[model=gemini:gemini-3.5-flash-lite]", "tac:actor=tabular"]
    names, _ = short_names(players, {players[2]: "tabular baseline"})
    assert names == {players[0]: "actor=llm[model=gpt-5-nano]", players[1]: "actor=llm[model=gemini-3.5-flash-lite]", players[2]: "tabular baseline"}


def test_names_stay_unique_when_models_differ_only_in_provider_or_options():
    players = ["tac:actor=llm[model=gemini:flash[thinking_enabled=True]]", "tac:actor=llm[model=gemini:flash[thinking_enabled=False]]"]
    names, _ = short_names(players)
    assert len(set(names.values())) == 2


def test_a_single_config_keeps_its_params_and_configs_of_different_keys_keep_their_keys():
    assert short_names(["monty_hall:num_doors=3,rules=hidden"]) == ({"monty_hall:num_doors=3,rules=hidden": "monty_hall:num_doors=3,rules=hidden"}, [])
    names, _ = short_names(["tic_tac_toe:", "magic_square:sample=False"])
    assert names == {"tic_tac_toe:": "tic_tac_toe", "magic_square:sample=False": "magic_square:sample=False"}


def test_an_item_is_named_by_the_axes_that_differ():
    items = [{"game": "nim:", "player": "random:distribution=uniform", "opponent": o} for o in ("optimal:stochastic=True", "optimal:stochastic=False")]
    names, joined, shared = item_names(items, prefixes={"opponent": "vs "})
    assert joined == ["vs stochastic=True", "vs stochastic=False"]
    assert names[0] == {"game": "nim", "player": "random:distribution=uniform", "opponent": "stochastic=True"} and shared == {}
    # a single item: its player's name
    assert item_names(items[:1])[1] == ["random:distribution=uniform"]
