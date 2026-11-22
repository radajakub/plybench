from __future__ import annotations

import re
from types import EllipsisType
from typing import Any

from plybench.common.store import store
from plybench.configs.benchmark_config import BenchmarkConfig
from plybench.configs.training_harness_config import TrainingHarnessConfig
from plybench.harness.matchup import COLOUR_RULES

# Experiments are records of the store, by key experiments/<kind>/<name>.json (files under experiments/ by default, as before): the harnesses read
# them, plybench-experiment writes them. A store that keeps history (plystore's Postgres) keeps every earlier version. An item of an experiment that
# has results is never removed, only disabled (enabled: false), so a result always has its item in the experiment.

KINDS: dict[str, type[BenchmarkConfig] | type[TrainingHarnessConfig]] = {"benchmarks": BenchmarkConfig, "training": TrainingHarnessConfig}
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    return kind


def experiment_key(kind: str, name: str) -> str:
    _kind(kind)
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid experiment name {name!r}: letters, digits, '_', '.' and '-', not starting with '.' or '-'")
    return f"experiments/{kind}/{name}.json"


def load_experiment(kind: str, name: str) -> dict[str, Any]:
    key = experiment_key(kind, name)
    data = store().get(key)
    if data is None:
        raise FileNotFoundError(f"no experiment {key}")
    return data


def list_experiments(kind: str) -> dict[str, dict[str, Any]]:
    # every experiment of the kind, by name
    prefix = f"experiments/{_kind(kind)}"
    keys = [key for key in store().keys(prefix) if re.fullmatch(rf"{prefix}/[^/]+\.json", key)]
    return {key.removeprefix(f"{prefix}/").removesuffix(".json"): value for key, value in store().get_many(keys).items()}


def _items(data: dict[str, Any]) -> dict[str, set[str]]:
    # the values of every list of toggle items ({value, enabled}), by the list's field
    return {
        field: {item["value"] for item in value} for field, value in data.items() if isinstance(value, list) and all(isinstance(item, dict) and "value" in item for item in value)
    }


def check_experiment(kind: str, data: Any) -> None:
    # what the harness would refuse when it loads the experiment, refused before it is saved
    try:
        config = KINDS[kind].from_dict(data)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"not a {kind} experiment: {type(error).__name__}: {error}") from error
    if isinstance(config, BenchmarkConfig) and config.colour_rule not in COLOUR_RULES:
        raise ValueError(f"unknown colour_rule {config.colour_rule!r}; expected one of {sorted(COLOUR_RULES)}")


def save_experiment(kind: str, name: str, data: dict[str, Any], expected: dict[str, Any] | None | EllipsisType = ...) -> None:
    # a new experiment, or a new version of one; refused when it removes an item of an experiment that has results. `expected`: the version it must
    # replace (None: it must be new), refused when the store holds another, so a version decided on elsewhere never overwrites a newer one
    key = experiment_key(kind, name)
    check_experiment(kind, data)
    old = store().get(key)
    if expected is not ... and old != expected:
        raise ValueError(f"{key} is not the version expected: {'it does not exist' if old is None else 'it exists' if expected is None else 'it changed'}")
    if old is not None:
        new = _items(data)
        removed = sorted(f"{field}: {value}" for field, values in _items(old).items() for value in values - new.get(field, set()))
        if removed and store().keys(f"results/{kind}/{name}"):
            raise ValueError(f"{key} has results; disable items (enabled: false) instead of removing them. Removed: {'; '.join(removed)}")
    store().put(key, data, replace=True)
