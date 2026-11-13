from __future__ import annotations

from collections.abc import Sequence

# Short readable names of config strings, for tools that show many of them side by side (the report, `--check`, plyweb). It works on the config
# grammar only, never on what a param means, so it names any player a plugin adds (a TextRL player as well as an `llm` player): a config is `<key>:<params>`,
# its params split at the commas outside brackets, and a param may nest `type[k=v,...]`. A name leaves out what every config of an axis shares (its
# params in common, the one model they all call) and keeps a model by its name alone (no provider, no options); an experiment file's label wins.

MODEL = "model="


def split_top(params: str) -> list[str]:
    # the params of a config at the outer level: split at the commas outside brackets
    parts: list[str] = []
    depth, start = 0, 0
    for index, char in enumerate(params):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(params[start:index])
            start = index + 1
    parts.append(params[start:])
    return [part for part in parts if part]


def _model_params(config: str) -> list[tuple[int, int, str]]:
    # each `model=<spec>` param (start, end, spec); a spec is <provider>:<name>[<options>] and ends at the first `,` or `]` outside its own brackets
    found: list[tuple[int, int, str]] = []
    start = config.find(MODEL)
    while start != -1:
        if start == 0 or config[start - 1] in "[,":
            end, depth = start + len(MODEL), 0
            while end < len(config):
                char = config[end]
                if char == "[":
                    depth += 1
                elif char == "]":
                    if depth == 0:
                        break
                    depth -= 1
                elif char == "," and depth == 0:
                    break
                end += 1
            found.append((start, end, config[start + len(MODEL) : end]))
            start = config.find(MODEL, end)
        else:
            start = config.find(MODEL, start + 1)
    return found


def models(config: str) -> list[str]:
    return [spec for _, _, spec in _model_params(config)]


def model_name(spec: str) -> str:
    # gemini:gemini-3.5-flash-lite[thinking_enabled=True] -> gemini-3.5-flash-lite
    name = spec.split("[", 1)[0]
    return name.split(":", 1)[1] if ":" in name else name


def _shorten_models(config: str, shared: str | None) -> str:
    # without the model params when every config calls the one model `shared`, else each model by its name alone; a param list emptied so loses its
    # brackets (compress[model=...] is compress)
    out = config
    for start, end, spec in reversed(_model_params(config)):
        if shared is None:
            out = f"{out[:start]}{MODEL}{model_name(spec)}{out[end:]}"
        elif end < len(out) and out[end] == ",":
            out = out[:start] + out[end + 1 :]
        elif start > 0 and out[start - 1] == ",":
            out = out[: start - 1] + out[end:]
        elif start > 0 and out[start - 1] == "[":
            out = out[: start - 1] + out[end + 1 :]
        else:
            out = out[:start] + out[end:]
    return out


def _parts(config: str) -> tuple[str, list[str]]:
    key, _, params = config.partition(":")
    return key, split_top(params)


def _short(values: Sequence[str], shared_model: str | None) -> tuple[dict[str, str], list[str]]:
    shortened = {value: _shorten_models(value, shared_model) for value in values}
    parsed = {value: _parts(text) for value, text in shortened.items()}
    keys = {key for key, _ in parsed.values()}
    common = set.intersection(*(set(params) for _, params in parsed.values())) if len(values) > 1 else set()
    names: dict[str, str] = {}
    for value, (key, params) in parsed.items():
        distinct = [param for param in params if param not in common]
        if len(values) == 1:
            names[value] = shortened[value].rstrip(":")
        elif not distinct:
            names[value] = key
        else:
            names[value] = (f"{key}:" if len(keys) > 1 else "") + ",".join(distinct)
    shared = [param for param in parsed[values[0]][1] if param in common]
    return names, shared


def short_names(values: Sequence[str], labels: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    # the name of every config of one axis, and what they share (left out of the names): a label where the experiment gives one, else the config without
    # the params every config has and with models by name. When two names would be the same (two models of one name, other options), the models keep
    # provider and options; when still the same, the whole config
    distinct = list(dict.fromkeys(values))
    if not distinct:
        return {}, []
    specs = {spec for value in distinct for spec in models(value)}
    shared_model = next(iter(specs)) if len(specs) == 1 and all(models(value) for value in distinct) else None
    names, shared = _short(distinct, shared_model)
    if len(set(names.values())) < len(names):
        names, shared = _short(distinct, None)
        if len(set(names.values())) < len(names):
            names = {}
            for value in distinct:
                key, params = _parts(value)
                names[value] = f"{key}:{','.join(params)}" if params else key
            shared = []
    if len(set(names.values())) < len(names):
        names, shared = {value: value for value in distinct}, []
    if shared_model is not None and len(distinct) > 1:
        shared = [f"{MODEL}{shared_model}", *shared]
    labelled = {value: (labels or {}).get(value, name) for value, name in names.items()}
    return labelled, shared


def item_names(
    items: Sequence[dict[str, str]], labels: dict[str, str] | None = None, prefixes: dict[str, str] | None = None
) -> tuple[list[dict[str, str]], list[str], dict[str, list[str]]]:
    # names of items described by their configs per axis (e.g. game, player, opponent): per item the name on each axis, and one name of the item made
    # of the axes that differ between the items (all of them for a single item: its player's name), each with its prefix (e.g. "vs " for an opponent);
    # with what each axis shares
    if not items:
        return [], [], {}
    axes = list(items[0])
    per_axis: dict[str, dict[str, str]] = {}
    shared: dict[str, list[str]] = {}
    for axis in axes:
        per_axis[axis], shared[axis] = short_names([item[axis] for item in items], labels)
    differing = [axis for axis in axes if len({item[axis] for item in items}) > 1] or axes[1:2] or axes[:1]
    names = [{axis: per_axis[axis][item[axis]] for axis in axes} for item in items]
    joined = [" · ".join(f"{(prefixes or {}).get(axis, '')}{name[axis]}" for axis in differing) for name in names]
    return names, joined, {axis: parts for axis, parts in shared.items() if parts}
