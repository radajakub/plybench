from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plybench.common.serializable import record_of
from plybench.configs.player_config import PlayerConfig
from plybench.trackers.game_tracker import GameStep, GameTracker

# One game as its records tell it, per player: the components with their full config, and the events in order (the moves, then the learning),
# each with what went in, where it came from, and what came out. Built by joining the recorded files only; no component code is run.


@dataclass(frozen=True)
class Source:
    # where a value came from: the game ("game"), an entry of a memory ("memory": memory and key), the reference counts at a key ("reference": key),
    # another event of the same player ("event": its index in events), an LLM call ("call": its index in the learner's llm_calls.json)
    kind: str
    memory: str | None = None
    key: str | None = None
    index: int | None = None


@dataclass(frozen=True)
class Value:
    name: str
    value: Any
    source: Source | None = None


@dataclass(frozen=True)
class Event:
    # one component at one step: "move" while the game is played, "learn" when the game is learned from; step is the player's step (1-based), None
    # for an event about the whole game; call is the LLM call it made (an LLMCall as recorded, or a move's call fields), with its index when it has one
    phase: str
    component: str
    inputs: tuple[Value, ...] = ()
    outputs: tuple[Value, ...] = ()
    step: int | None = None
    call: dict[str, Any] | None = None


@dataclass(frozen=True)
class Component:
    # one part of the player as its description gives it: id (e.g. "actor", "actor.apply"), role, type, every param, and the model it calls
    id: str
    role: str
    type: str
    params: Any
    model: str | None = None


@dataclass(frozen=True)
class MemoryEntry:
    # one entry of a memory as the game started from it, with the steps that read it and the steps whose lessons were written into it
    key: str
    text: str
    visits: int
    actions: dict[str, int]
    writes: int
    read_at: tuple[int, ...] = ()
    written_at: tuple[int, ...] = ()


@dataclass(frozen=True)
class MemoryView:
    # a memory of the player at the start of the game (from the state file `before`), every entry with what the game did to it; entries never read
    # are those not retrieved
    memory: str
    before: str | None
    entries: tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class PlayerTrace:
    key: str
    config: str
    description: dict[str, Any] | None
    components: tuple[Component, ...]
    events: tuple[Event, ...]
    memories: tuple[MemoryView, ...] = ()
    # what the learner kept at the end of the game (GameEnding.players), e.g. its state links and shaped outcome
    ending: dict[str, Any] | None = None


@dataclass(frozen=True)
class GameTrace:
    path: str
    # the game config from the matchup's metadata.json; None without it
    game: str | None
    round: int
    seed: int | None
    # where a training game belongs (metadata.json place); None for a benchmark's
    place: dict[str, Any] | None
    # the result for seat i ("win", "loss", "draw", "my_fail", "opponent_fail"); None for a game without an ending
    result: str | None
    players: dict[str, PlayerTrace] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return record_of(self)


@dataclass(frozen=True)
class TraceContext:
    # what a reader of one player's part of a game is given: the game, its file, the player and its seat, its steps in order, its description from
    # the session that played the round, what it kept at the end, and its learner folder for this game (update.json, llm_calls.json, state/; None
    # outside training). `resolve` turns a recorded path (relative to the working directory of the run) into one under the results being read
    game: GameTracker
    game_file: Path
    player: PlayerConfig
    identifier: str
    steps: list[GameStep]
    description: dict[str, Any] | None
    ending: dict[str, Any] | None
    learned: Path | None
    place: dict[str, Any] | None
    resolve: Callable[[str], Path]


# a player plugin's reader (PlayerSpec.trace); a player without one is read by plybench.trace.load.read_player
type TraceReader = Callable[[TraceContext], PlayerTrace]
