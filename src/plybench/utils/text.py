from collections.abc import Iterable


def to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("true", "True", "1", "yes", "Yes", "YES", "y", "Y"):
        return True
    if value in ("false", "False", "0", "no", "No", "NO", "n", "N"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def extract_params(param_string: str) -> dict[str, str]:
    # name=value pairs separated by commas; a value may itself contain "=" (a path, a model option), so only the first one splits
    params: dict[str, str] = {}
    for param in param_string.split(","):
        if not param:
            continue
        name, separator, value = param.partition("=")
        if not separator:
            raise ValueError(f"Invalid parameter {param!r} in {param_string!r}; expected name=value")
        params[name] = value
    return params


def inline_multiline_string(prompt: str) -> str:
    return " ".join(line.strip() for line in prompt.split("\n") if line.strip())


def _num_to_char(number: int, start: str) -> str:
    return chr(ord(start) + number)


def number_to_char_upper(number: int) -> str:
    return _num_to_char(number, "A")


def number_to_char_lower(number: int) -> str:
    return _num_to_char(number, "a")


def _char_to_num(char: str, start: str) -> int:
    return ord(char) - ord(start)


def char_to_number_upper(char: str) -> int:
    return _char_to_num(char, "A")


def char_to_number_lower(char: str) -> int:
    return _char_to_num(char, "a")


def compress_ranges(numbers: Iterable[int]) -> str:
    # "1,2,3,5,9,10" -> "1-3,5,9-10", so a long list of round numbers stays a short token in a log line
    ordered = sorted(set(numbers))
    groups: list[list[int]] = []
    for number in ordered:
        if groups and number == groups[-1][1] + 1:
            groups[-1][1] = number
        else:
            groups.append([number, number])
    return ",".join(str(start) if start == end else f"{start}-{end}" for start, end in groups)


def order_suffix(order: int) -> str:
    last_char = str(order)[-1]
    return {"1": "st", "2": "nd", "3": "rd"}.get(last_char, "th")
