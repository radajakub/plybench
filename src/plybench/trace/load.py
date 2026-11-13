from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plybench.registry import Registry
from plybench.trace.model import Component, Event, GameTrace, PlayerTrace, Source, TraceContext, TraceStep, Value
from plybench.trackers.game_tracker import GameStep, GameTracker
from plybench.trackers.step_data import StepData

GAME = Source("game")
# the fields of a move's one LLM call kept on its GameStep, as an event's call shows them
_MOVE_CALL = ("model_string", "model_options", "finish_reason", "truncated", "retries", "blocks", "llm_seed", "input_tokens", "output_tokens", "reasoning_tokens")


def load_game(game_file: Path, registry: Registry, players: bool = True) -> GameTrace:
    # one saved game with both players read: each by its plugin's reader (PlayerSpec.trace), or by read_player; players=False: the game as played
    # only (its moves, its end, its params), without reading the players or their learning
    game = GameTracker.from_dict(json.loads(game_file.read_text()), registry)
    metadata_file = game_file.parent / "metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    place = metadata.get("place")
    # the players as the session that played this round described them, matched by config: a game's seats need not be the matchup's (a training
    # phase alternates who starts)
    session = next((session for session in metadata.get("sessions", []) if game.game_round in session.get("rounds", [])), {})
    described = {description["config"]: description for description in session.get("players", {}).values()}
    read: dict[str, PlayerTrace] = {}
    for identifier, config in (("i", game.i_player), ("o", game.o_player)) if players else ():
        ending = game.ending.players.get(config.hash) if game.ending is not None else None
        context = TraceContext(
            game,
            game_file,
            config,
            identifier,
            game.steps_of(config),
            described.get(config.to_string()),
            ending,
            _learned(game_file, game.game_round, config.to_string() == metadata.get("i_config"), ending, place),
            place,
            lambda recorded: _resolve(recorded, game_file),
        )
        reader = registry.resolve_player(config.key).trace or read_player
        read[identifier] = reader(context)
    result = game.ending.result.name.lower() if game.ending is not None else None
    # the seat of each move by its player; none when both seats are one player (self-play), as the steps do not tell them apart
    seats = {game.i_player.hash: "i", game.o_player.hash: "o"} if game.i_player.hash != game.o_player.hash else {}
    steps = tuple(TraceStep(seats.get(step.player_hash), step.observation, step.move) for step in game.steps)
    final = game.ending.observation if game.ending is not None else None
    return GameTrace(game_file.as_posix(), metadata.get("game_config"), game.game_round, game.seed, place, result, read, steps, final, game.instance_params, game.other_params)


def game_files(folder: Path) -> list[Path]:
    # the saved games directly in a folder (a matchup's, or a training phase's), by round
    return sorted(folder.glob("game_*.json"), key=lambda path: int(path.stem.removeprefix("game_")) if path.stem.removeprefix("game_").isdigit() else 0)


def _resolve(recorded: str, game_file: Path) -> Path:
    # a path the run recorded relative to its working directory ("results/..."), under the folder that holds the results being read
    path = Path(recorded)
    if path.is_absolute() or not path.parts:
        return path
    parts = game_file.resolve().parts
    root = path.parts[0]
    return Path(*parts[: parts.index(root)]) / path if root in parts else path


def _learned(game_file: Path, game_round: int, trainee: bool, ending: dict[str, Any] | None, place: dict[str, Any] | None) -> Path | None:
    # the learner folder of a training game: as the learner recorded it (LearnablePlayer.finish), or, in records from before, where plybench's
    # layout puts it (epoch_<e>/learner/game_<N> beside epoch_<e>/train) for the trainee, the training matchup's player i
    if ending is not None and ending.get("learned"):
        return _resolve(ending["learned"], game_file)
    # a matchup saved before places were kept is a training phase by its folder's name
    phase = place.get("phase") if place is not None else game_file.parent.name
    if phase != "train" or not trainee:
        return None
    folder = game_file.parent.parent / "learner" / f"game_{game_round}"
    return folder if folder.exists() else None


def move_call(step: GameStep) -> dict[str, Any] | None:
    # the fields of a move's LLM call as its step keeps them; None for a move without a model (a bot)
    call = {name: getattr(step, name) for name in _MOVE_CALL if getattr(step, name) not in (None, [])}
    return call if call.get("model_string") is not None or call.get("input_tokens") is not None else None


def move_event(step: GameStep, number: int, component: str, inputs: tuple[Value, ...] = ()) -> Event:
    # a move as the step records it: an LLM move (the prompts in, the reasoning, the answer, what was read from it and the move out, its call),
    # or a bot's (the observation in, the move out); `inputs` are added before the prompts (e.g. a context a component built)
    data = StepData.from_dict(step.data)
    if data.prompt_message or data.full_output:
        prompts = (Value("system", data.system_message, GAME), Value("prompt", data.prompt_message, GAME))
        outputs = (Value("reasoning", data.reasoning_trace), Value("output", data.full_output), Value("extracted", step.extracted), Value("move", step.move))
        return Event("move", component, (*inputs, *prompts), outputs, number, move_call(step))
    return Event("move", component, (*inputs, Value("observation", step.observation, GAME)), (Value("move", step.move),), number)


def read_player(context: TraceContext) -> PlayerTrace:
    # a player without a reader of its own: one component, the player as described, and one event per move
    described = context.description or {}
    params = described.get("params")
    model = params.get("model") if isinstance(params, dict) else None
    model_name = f"{model['provider']}:{model['model_name']}" if isinstance(model, dict) and "provider" in model else None
    component = Component(context.player.key, "player", context.player.key, params, model_name)
    events = tuple(move_event(step, number, context.player.key) for number, step in enumerate(context.steps, start=1))
    return PlayerTrace(context.player.key, context.player.to_string(), context.description, (component,), events, (), context.ending)
