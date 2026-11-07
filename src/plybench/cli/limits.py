from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from plyllm import ModelLimits, Provider

# Per-model quotas are account-specific, so none ship with the package: each checkout keeps its own in a gitignored JSON file,
# {"<provider>": {"<model>": {"max_concurrent": 16, "rps": 16.67, "tpm": 500000}}}, every field optional (see limits.example.json).
# The file is read from, in order: --limits, $PLYBENCH_LIMITS, ./limits.json; with none of them, only the provider-wide
# concurrency applies. The model `*` is the whole provider, {"metacentrum": {"*": {"max_concurrent": 4}}}: every call to it, over all models, and it
# replaces --concurrency for that provider; max_concurrent only, as plyllm paces rps and tpm per model
LIMITS_ENV = "PLYBENCH_LIMITS"
PROVIDER_WIDE = "*"
DEFAULT_LIMITS = Path("limits.json")

type Limits = dict[Provider, dict[str, ModelLimits]]


def limits_path(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit
    from_env = os.environ.get(LIMITS_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_LIMITS if DEFAULT_LIMITS.exists() else None


def load_limits(path: Path) -> Limits:
    data: dict[str, dict[str, dict[str, Any]]] = json.loads(path.read_text())
    limits: Limits = {}
    for provider_name, models in data.items():
        provider = Provider.from_value(provider_name)
        if provider is None:
            raise ValueError(f"{path}: unknown provider {provider_name!r}; known: {[provider.value for provider in Provider]}")
        try:
            limits[provider] = {model: ModelLimits(**fields) for model, fields in models.items()}
        except TypeError as error:
            raise ValueError(f"{path}: invalid limits for {provider_name}: {error}; fields are max_concurrent, rps, tpm") from error
        whole = limits[provider].get(PROVIDER_WIDE)
        if whole is not None and (whole.rps is not None or whole.tpm is not None or whole.max_concurrent is None):
            raise ValueError(f"{path}: the limit of the whole provider {provider_name} ({PROVIDER_WIDE!r}) takes max_concurrent only")
    return limits
