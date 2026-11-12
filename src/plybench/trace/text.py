from __future__ import annotations

import json
from typing import Any

from plybench.trace.model import Event, GameTrace, MemoryView, PlayerTrace, Source

# A trace as text to read: the game, then per player its components, its moves step by step and its learning (each event with what went in, where
# it came from, and what came out), and its memories as the game started. Texts are printed whole unless a limit is given.

SECTIONS = ("components", "moves", "learning", "memory")
_INDENT = "      "


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"


def _value(value: Any, limit: int | None) -> str:
    # a string as it is (cut to one line with a limit), "(empty)" for an empty one, a read of a memory entry by its state and text, anything else as
    # compact JSON; a multi-line text continues on indented lines
    if isinstance(value, dict) and {"memory", "key", "text"} <= set(value):
        state = f"{'found' if value.get('found', True) else 'no entry'}, visits {value.get('visits')}, writes {value.get('writes')}"
        return f"{state}: {_value(value['text'], limit)}"
    if value == "":
        return "(empty)"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if limit is not None:
        return _one_line(text, limit)
    return text.replace("\n", f"\n{_INDENT}  ")


def _source(source: Source | None, events: tuple[Event, ...]) -> str:
    if source is None:
        return ""
    if source.kind == "memory":
        return f" ← memory {source.memory}: {_one_line(source.key or '', 60)}"
    if source.kind == "reference":
        return " ← reference counts"
    if source.kind == "event" and source.index is not None and 0 <= source.index < len(events):
        event = events[source.index]
        return f" ← [{source.index}] {event.component}" + (f" (step {event.step})" if event.step is not None else "")
    if source.kind == "call":
        return f" ← call #{source.index}"
    return f" ← {source.kind}"


def _call(call: dict[str, Any]) -> str:
    # the call in one line: which one, the model that answered, why it stopped, the retries and the tokens
    parts = [f"call #{call['index']} {call.get('role', '')}".strip() if "index" in call else "call"]
    model = call.get("model_string")
    parts.append(f"model {model}" if model else "")
    if call.get("finish_reason") is not None:
        parts.append(f"finish {call['finish_reason']}" + (" (cut at the output limit)" if call.get("truncated") else ""))
    if call.get("retries"):
        parts.append(f"{call['retries']} retries")
    if call.get("blocks"):
        parts.append(f"{len(call['blocks'])} blocked attempts {call['blocks']}")
    if call.get("failure"):
        parts.append(f"failure {call['failure']}")
    tokens = (call.get("input_tokens"), call.get("output_tokens"), call.get("reasoning_tokens"))
    if tokens[0] is not None:
        parts.append(f"tokens in {tokens[0]}, out {tokens[1]}" + (f" (reasoning {tokens[2]})" if tokens[2] else ""))
    # the options set for the call; those left to the provider (None) or off are not listed
    options = {name: value for name, value in (call.get("options") or call.get("model_options") or {}).items() if value not in (None, False)}
    if options:
        parts.append(f"options {json.dumps(options, ensure_ascii=False)}")
    return ", ".join(part for part in parts if part)


def _event(index: int, event: Event, events: tuple[Event, ...], limit: int | None) -> list[str]:
    head = f"  [{index}] {event.component}" + (f" (step {event.step})" if event.step is not None and event.phase == "learn" else "")
    lines = [head]
    # a value a record does not hold (an older record) is left out; an input known only by its source shows where it came from
    for value in event.inputs:
        if value.value is not None:
            lines.append(f"    in  {value.name}{_source(value.source, events)}: {_value(value.value, limit)}")
        elif value.source is not None:
            lines.append(f"    in  {value.name}{_source(value.source, events)}")
    lines += [f"    out {value.name}{_source(value.source, events)}: {_value(value.value, limit)}" for value in event.outputs if value.value not in (None, "", [], {})]
    if event.call is not None:
        lines.append(f"    {_call(event.call)}")
    return lines


def _memory(view: MemoryView, limit: int | None, all_entries: bool) -> list[str]:
    lines = [f"  Memory {view.memory}, as the game started (from {view.before}): {len(view.entries)} entries"]
    touched = [entry for entry in view.entries if entry.read_at or entry.written_at]
    untouched = [entry for entry in view.entries if not (entry.read_at or entry.written_at)]
    for entry in touched:
        marks = ", ".join(
            part for part in (f"read at steps {list(entry.read_at)}" if entry.read_at else "", f"written from steps {list(entry.written_at)}" if entry.written_at else "") if part
        )
        lines.append(f"    {marks}; visits {entry.visits}, writes {entry.writes}, actions {json.dumps(entry.actions, ensure_ascii=False)}")
        lines.append(f"      key: {_value(entry.key, limit)}")
        lines.append(f"      text: {_value(entry.text, limit) or '(empty)'}")
    lines.append(f"    not read or written in this game: {len(untouched)} entries")
    for entry in untouched:
        lines.append(f"      - {_one_line(entry.key, limit or 120)} (visits {entry.visits}, writes {entry.writes})")
        if all_entries:
            lines.append(f"        text: {_value(entry.text, limit) or '(empty)'}")
    return lines


def _player(seat: str, player: PlayerTrace, sections: tuple[str, ...], limit: int | None, all_entries: bool) -> list[str]:
    lines = [f"== Player {seat}: {player.config}"]
    for name, kept in (player.ending or {}).items():
        lines.append(f"  {name}: {_value(kept, None)}")
    if "components" in sections:
        lines.append("  Components" + ("" if player.description is not None else " (no description recorded for this round)"))
        for component in player.components:
            model = f" {component.model}" if component.model else ""
            lines.append(f"    {component.id}: {component.type}{model} {_value(component.params, limit)}")
    events = player.events
    if "moves" in sections:
        step = None
        for index, event in enumerate(events):
            if event.phase != "move":
                continue
            if event.step != step:
                step = event.step
                lines.append(f" Step {step}")
            lines += _event(index, event, events, limit)
    if "learning" in sections and any(event.phase == "learn" for event in events):
        lines.append(" Learning")
        lines += [line for index, event in enumerate(events) if event.phase == "learn" for line in _event(index, event, events, limit)]
    if "memory" in sections:
        lines += [line for view in player.memories for line in _memory(view, limit, all_entries)]
    return lines


def render(trace: GameTrace, players: list[str] | None = None, sections: tuple[str, ...] = SECTIONS, limit: int | None = None, all_entries: bool = False) -> str:
    # players: seats ("i", "o") or player keys to show (default both); limit: cut every text to one line of this length
    place = trace.place or {}
    where = f", {place.get('experiment')}, epoch {place.get('epoch')}/{place.get('epochs')} {place.get('phase')}" if place else ""
    lines = [f"Game {trace.game} round {trace.round}, seed {trace.seed}, result for i: {trace.result}{where}", f"  {trace.path}"]
    for seat, player in trace.players.items():
        if players is None or seat in players or player.key in players:
            lines += _player(seat, player, sections, limit, all_entries)
    return "\n".join(lines)
