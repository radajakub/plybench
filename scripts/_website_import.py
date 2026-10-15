"""Reverse of `_website_export.py`: unpack an exported .tar.gz back into the repo's on-disk layout.

The archive keeps everything under a single wrapper dir (the experiment name) and flattens the LLM
extras of every step into the website's schema; both are undone here, so the restored files are
byte-equivalent to what `ResultTracker`/`GameTracker` write:
    <name>/results/benchmarks/...   ->  results/benchmarks/...   (steps re-nested under `data`)
    <name>/analysis/benchmarks/...  ->  analysis/benchmarks/...  (verbatim, opt-in: derived data)
    <name>/experiment.json          ->  experiments/benchmarks/<name>.json (opt-in: lossy, `baseline`
                                        is split back into the `opponents` list)

Only the paths present in the archive are touched -- other experiments, matchups and games stay as
they are. A matchup that already exists locally keeps its own completed rounds: `completed` is merged
rather than replaced, since the archive may hold a subset of the games sitting on disk.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_STEP_KEYS = ("seq", "player_name", "player_hash", "serialized_state", "observation", "move")
_TOKEN_KEYS = ("input_tokens", "output_tokens", "reasoning_tokens")
# website flat key -> key inside the step's `data` extras (mirror of the export's unfolding);
# ordered as LLMPlayerTracker.record writes them, so the restored files are byte-identical
_DATA_KEYS = {"reasoning_trace": "reasoning_trace", "full_model_output": "full_output", "system_message": "system_message", "prompt_message": "prompt_message"}


@dataclass
class ImportCounts:
    matchups: int = 0
    games: int = 0
    game_stats: int = 0
    overwritten: int = 0
    experiments: set[str] = field(default_factory=set)


def step_json(step: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {key: step[key] for key in _STEP_KEYS}
    for key in _TOKEN_KEYS:
        if step.get(key) is not None:
            out[key] = step[key]
    data = {data_key: step[flat_key] for flat_key, data_key in _DATA_KEYS.items() if step.get(flat_key)}
    if data:
        out["data"] = data
    return out


def game_tracker_json(payload: dict[str, Any]) -> dict[str, Any]:
    payload["steps"] = [step_json(step) for step in payload["steps"]]
    return payload


def experiment_json(payload: dict[str, Any]) -> dict[str, Any]:
    # the export joins the opponents into a single informational `baseline` string; no opponent config
    # serialization contains ", ", so the split restores the original list
    baseline = payload.get("baseline") or ""
    return {
        "game_configs": payload["game_configs"],
        "opponents": [{"value": value, "enabled": True} for value in baseline.split(", ") if value],
        "player_configs": payload["player_configs"],
        "num_games": payload["num_games"],
    }


def metadata_json(payload: dict[str, Any], existing: Path) -> dict[str, Any]:
    if not existing.exists():
        return payload
    with open(existing, "r") as f:
        local = json.load(f)
    # keep the rounds already on disk: the archive may carry only part of the matchup
    payload["completed"] = sorted(set(payload["completed"]) | {int(round_) for round_ in local["completed"]})
    return payload


def _wrapper_prefix(names: list[str]) -> str:
    prefixes = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(prefixes) != 1:
        raise SystemExit(f"expected a single wrapper directory in the archive, found: {sorted(prefixes) or ['none']}")
    return prefixes.pop()


def _safe_target(dest: Path, relative: str) -> Path:
    # checked lexically: results/ may be a symlink to a data directory outside the repository
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"archive member escapes the destination directory: {relative}")
    return dest / path


def _write(target: Path, payload: dict[str, Any], counts: ImportCounts, dry_run: bool) -> None:
    if target.exists():
        counts.overwritten += 1
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(payload, f, indent=None)


def restore_archive(archive: str, dest: Path, with_analysis: bool = False, with_experiment: bool = False, dry_run: bool = False) -> ImportCounts:
    counts = ImportCounts()

    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        name = _wrapper_prefix([member.name for member in members])

        for member in members:
            relative = member.name[len(name) + 1 :]
            payload = json.load(tar.extractfile(member))  # type: ignore[arg-type]
            parts = Path(relative).parts

            if relative == "experiment.json":
                if with_experiment:
                    _write(_safe_target(dest, f"experiments/benchmarks/{name}.json"), experiment_json(payload), counts, dry_run)
                continue
            if parts[0] == "analysis":
                if not with_analysis:
                    continue
                _write(_safe_target(dest, relative), payload, counts, dry_run)
                counts.game_stats += parts[-1] != "metadata.json"
                continue
            if parts[0] != "results":
                continue

            counts.experiments.add(parts[2])
            target = _safe_target(dest, relative)
            if parts[-1] == "metadata.json":
                counts.matchups += 1
                _write(target, metadata_json(payload, target), counts, dry_run)
            else:
                counts.games += 1
                _write(target, game_tracker_json(payload), counts, dry_run)

    return counts
