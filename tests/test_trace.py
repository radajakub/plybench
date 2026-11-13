"""Reading a saved game back as a trace: both players with their description from the session that played the round, one event per move
(a bot's: the observation in, the move out; an LLM move's: the prompts in, the answer, what was read from it and the move out, with its call),
and for a training game the learner folder it was learned into."""

from __future__ import annotations

import asyncio
import json

from plyllm import LLMConfig

from plybench.app import PlyBench
from plybench.cli import trace as trace_cli
from plybench.configs.matchup import Matchup
from plybench.harness.matchup import run_matchup_concurrent
from plybench.trace.load import load_game, move_event
from plybench.trace.model import Event, GameTrace, MemoryEntry, MemoryView, PlayerTrace, Source, Value
from plybench.trace.text import render
from plybench.trackers.game_tracker import GameStep

op = PlyBench(LLMConfig())
registry = op.registry
MATCHUP = Matchup(registry.game_config("tic_tac_toe:"), registry.player_config("random:distribution=uniform"), registry.player_config("random:distribution=normal"), 2)


def test_a_benchmark_game_reads_back_with_both_players_described(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = asyncio.run(run_matchup_concurrent(op, MATCHUP, experiment="exp", max_concurrent=1, seed=3))
    game_file = tracker.path_builder.game_file(tracker.base_path, 1)

    trace = load_game(game_file, registry)

    assert trace.round == 1 and trace.seed is not None and trace.game == "tic_tac_toe:" and trace.place is None and trace.result is not None
    i, o = trace.players["i"], trace.players["o"]
    assert i.description == MATCHUP.i.describe() and o.config == "random:distribution=normal"
    [component] = i.components
    assert (component.id, component.type, component.params) == ("random", "random", MATCHUP.i.describe()["params"])
    # one event per move of the player, in order, each from the game to the move
    steps = json.loads(game_file.read_text())["steps"]
    assert len(i.events) + len(o.events) == len(steps)
    first = i.events[0]
    assert (first.phase, first.step, first.call) == ("move", 1, None)
    assert first.inputs[0].name == "observation" and first.inputs[0].source == Source("game") and first.outputs[0].name == "move"
    assert json.loads(json.dumps(trace.to_dict()))["players"]["i"]["events"][0]["inputs"][0]["source"]["kind"] == "game"
    # the game as played, for a view of its states: both players' moves in order with their seats, the observation it ended in, its params
    assert [(step.seat, step.observation, step.move) for step in trace.steps] == [
        ("i" if step["player_hash"] == MATCHUP.i.hash else "o", step["observation"], step["move"]) for step in steps
    ]
    assert {step.seat for step in trace.steps} == {"i", "o"} and trace.final is not None and trace.final.startswith("State:")
    assert trace.instance_params is not None and trace.other_params is not None


def test_an_llm_move_reads_as_its_prompts_answer_and_call():
    step = GameStep(
        1,
        "llm:x",
        "h",
        "state",
        "obs",
        "<C1R1>",
        input_tokens=10,
        output_tokens=3,
        data={"system_message": "rules", "prompt_message": "your move", "reasoning_trace": "think", "full_output": "Action: <C1R1>"},
        llm_seed=7,
        model_string="m-1",
        model_options={"max_tokens": None},
        finish_reason="stop",
        truncated=False,
        retries=0,
        extracted="<C1R1>",
    )

    event = move_event(step, 1, "llm")

    assert [value.name for value in event.inputs] == ["system", "prompt"] and event.inputs[1].value == "your move"
    assert {value.name: value.value for value in event.outputs} == {"reasoning": "think", "output": "Action: <C1R1>", "extracted": "<C1R1>", "move": "<C1R1>"}
    assert event.call is not None and (event.call["model_string"], event.call["finish_reason"], event.call["llm_seed"]) == ("m-1", "stop", 7)
    # a bot's step: the observation in, the move out, no call
    bot = move_event(GameStep(2, "random:", "h", "state", "obs", "<C2R2>"), 2, "random")
    assert [value.name for value in bot.inputs] == ["observation"] and bot.call is None


def test_the_text_and_the_json_of_a_game(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    tracker = asyncio.run(run_matchup_concurrent(op, MATCHUP, experiment="exp", max_concurrent=1, seed=3))
    game_file = tracker.path_builder.game_file(tracker.base_path, 1)
    trace = load_game(game_file, registry)

    text = render(trace)
    assert text.startswith("Game tic_tac_toe: round 1") and "== Player i: random:distribution=uniform" in text and " Step 1" in text
    assert "    in  observation ← game: " in text and "    out move: <" in text
    # only the chosen player and sections; a limit keeps every text on one line
    short = render(trace, ["o"], ("moves",), 40)
    assert "== Player i" not in short and "Components" not in short and all(len(line) < 120 for line in short.splitlines()[2:])

    monkeypatch.setattr("sys.argv", ["plybench-trace", str(game_file), "--json"])
    trace_cli.main()
    assert json.loads(capsys.readouterr().out) == trace.to_dict()


def test_a_step_of_another_game_of_the_batch_and_the_samples_of_an_entry_are_shown():
    write = Event("learn", "actor.apply", (Value("part", "a lesson", Source("batch", index=2, round=1)),), (Value("new", "notes"),))
    memory = MemoryView("actor", "state", (MemoryEntry("k", "notes", 2, {}, 1, (1,), (1,), 3), MemoryEntry("other", "", 0, {}, 0, (2,))))
    trace = GameTrace("game_2.json", "tic_tac_toe:", 2, 0, None, "win", {"i": PlayerTrace("textrl", "textrl:", None, (), (write,), (memory,))})

    text = render(trace)

    assert "    in  part ← game 1, step 2: a lesson" in text
    # samples where the player counts them, nothing where it does not
    assert "visits 2, writes 1, samples 3, actions" in text and "visits 0, writes 0, actions" in text


def test_a_folder_of_games_reads_as_played_without_the_players(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    tracker = asyncio.run(run_matchup_concurrent(op, MATCHUP, experiment="exp", max_concurrent=1, seed=3))
    folder = tracker.path_builder.game_file(tracker.base_path, 1).parent

    monkeypatch.setattr("sys.argv", ["plybench-trace", str(folder), "--steps"])
    trace_cli.main()
    games = json.loads(capsys.readouterr().out)["games"]

    # every game of the folder by round, each as played: its moves with their seats, its end and params, and no players
    assert [game["round"] for game in games] == list(range(1, MATCHUP.num_games + 1))
    played = load_game(tracker.path_builder.game_file(tracker.base_path, 1), registry)
    assert games[0]["players"] == {} and games[0]["steps"] == json.loads(json.dumps(played.to_dict()))["steps"] and games[0]["final"] == played.final
