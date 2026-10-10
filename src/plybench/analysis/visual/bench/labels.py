from __future__ import annotations

from dataclasses import replace

from plyllm import ReasoningEffort

from plybench.configs.game_config import GameConfig
from plybench.configs.player_config import PlayerConfig
from plybench.player.llm_player import LLMParams

Overrides = dict[str, str]


def _variant_params(config: PlayerConfig) -> dict[str, str]:
    params = config.params
    if not isinstance(params, LLMParams):
        return {}
    options = params.model.options
    values = {
        "obs": params.observation_type.value,
        "out": params.output_strategy.value,
        "thinking": str(options.thinking_enabled),
        "temperature": None if options.temperature is None else str(options.temperature),
        "max_tokens": None if options.max_tokens is None else str(options.max_tokens),
    }
    return {key: value for key, value in values.items() if value is not None}


def player_label(config: PlayerConfig, overrides: Overrides | None = None, show: list[str] | None = None) -> str:
    if overrides and (override := overrides.get(config.to_string())):
        return override
    params = config.params
    if not isinstance(params, LLMParams):
        return _simple_player_label(config)
    variants = _variant_params(config)
    effort = params.model.options.reasoning_effort
    parts = [effort] if effort else []
    parts += [f"{key}={variants[key]}" for key in show or [] if key in variants]
    return f"{params.model.model_name} ({', '.join(parts)})" if parts else params.model.model_name


def player_labels(configs: list[PlayerConfig], overrides: Overrides | None = None) -> list[str]:
    labels: list[str] = []
    for config in configs:
        params = _variant_params(config)
        siblings = [other for other in configs if other is not config and player_label(other, overrides) == player_label(config, overrides)]
        show = [key for key in params if any(_variant_params(other).get(key) != params[key] for other in siblings)]
        labels.append(player_label(config, overrides, show))
    return labels


def _simple_player_label(config: PlayerConfig) -> str:
    suffix = config.params.to_string()
    return f"{config.key} ({suffix})" if suffix else config.key


def provider_key(config: PlayerConfig) -> str:
    params = config.params
    return params.model.provider.value if isinstance(params, LLMParams) else config.key


def model_name(config: PlayerConfig) -> str:
    params = config.params
    return params.model.model_name if isinstance(params, LLMParams) else config.params.to_string()


def model_key(config: PlayerConfig) -> str:
    params = config.params
    if not isinstance(params, LLMParams):
        return params.to_string()
    return replace(params, model=replace(params.model, options=replace(params.model.options, reasoning_effort=None))).to_string()


def effort_key(config: PlayerConfig) -> ReasoningEffort | None:
    params = config.params
    return params.model.options.reasoning_effort if isinstance(params, LLMParams) else None


# Size tiers, weakest first. Names are matched as whole hyphen-delimited words against the model name,
# longest first, so "flash-lite" wins over "flash". A model carrying none of these is the provider's
# undifferentiated flagship (gpt-5.4, grok-4.5, glm-5.2) and outranks every suffixed sibling.
TIER_ORDER: tuple[tuple[str, ...], ...] = (
    ("nano", "tiny"),
    ("lite", "flash-lite", "small", "mini-lite"),
    ("mini", "haiku", "flash-8b"),
    ("flash", "medium", "sonnet"),
    ("large", "pro", "opus", "ultra", "max"),
)
FLAGSHIP_TIER = len(TIER_ORDER)


def tier_rank(model_name: str) -> int:
    words = model_name.lower().replace("_", "-").split("-")
    joined = "-".join(words)

    # compound tiers are matched first and longest-first, so "flash-lite" wins over the "flash" it
    # contains -- otherwise a lite model would be ranked as a full one
    compound = [(rank, token) for rank, tokens in enumerate(TIER_ORDER) for token in tokens if "-" in token and token in joined]
    if compound:
        return min(compound, key=lambda item: (-len(item[1]), item[0]))[0]

    # among bare words the weakest match wins: a qualifier like "lite" demotes whatever it modifies
    matched = [rank for rank, tokens in enumerate(TIER_ORDER) for token in tokens if token in words]
    return min(matched) if matched else FLAGSHIP_TIER


def _version(model_name: str) -> tuple[float, ...]:
    numbers: list[float] = []
    for word in model_name.lower().replace("_", "-").split("-"):
        try:
            numbers.append(float(word))
        except ValueError:
            continue
    return tuple(numbers)


def model_strength(config: PlayerConfig) -> tuple[int, tuple[float, ...], str]:
    name = model_name(config)
    return (tier_rank(name), _version(name), model_key(config))


def game_params(config: GameConfig) -> dict[str, str]:
    pairs = [part.split("=", 1) for part in config.params.to_string().split(",") if "=" in part]
    return {key: value for key, value in pairs}


def _short_keys(keys: list[str]) -> dict[str, str]:
    short = {key: key.rsplit("_", 1)[-1] for key in keys}
    return short if len(set(short.values())) == len(keys) else {key: key for key in keys}


def game_label(config: GameConfig, overrides: Overrides | None = None, show: list[str] | None = None) -> str:
    if overrides and (override := overrides.get(config.to_string())):
        return override
    name = config.key.replace("_", " ").title()
    params = game_params(config)
    keys = [key for key in show or [] if key in params]
    short = _short_keys(keys)
    return "\n".join([name, *(f"{short[key]}={params[key]}" for key in keys)])


def game_labels(configs: list[GameConfig], overrides: Overrides | None = None) -> list[str]:
    labels: list[str] = []
    for config in configs:
        siblings = [other for other in configs if other.key == config.key]
        params = game_params(config)
        show = [key for key in params if any(game_params(other).get(key) != params[key] for other in siblings)]
        labels.append(game_label(config, overrides, show))
    return labels


def metric_label(metric: str) -> str:
    return metric.replace("_", " ").title()
