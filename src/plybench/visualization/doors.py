from itertools import zip_longest


class DoorPrinter:
    # a row of doors numbered from 1: what each shows ("?" while it is closed, with a knob) and its marks under it, one per line (e.g. "picked")
    def __call__(self, contents: list[str], marks: list[list[str]]) -> str:
        if not contents or len(contents) != len(marks):
            return ""

        width = max(7, *(len(text) for text in contents), *(len(mark) for door in marks for mark in door))
        closed = [content == "?" for content in contents]

        def row(cells: list[str]) -> str:
            return " ".join(f"|{cell}|" for cell in cells)

        border = " ".join("+" + "-" * width + "+" for _ in contents)
        lines = [
            border,
            row([" " * width for _ in contents]),
            row([content.center(width) for content in contents]),
            row([("o" if is_closed else "").rjust(width - 1).ljust(width) for is_closed in closed]),
            row([" " * width for _ in contents]),
            border,
            " ".join(f"door {door}".center(width + 2) for door in range(1, len(contents) + 1)),
        ]
        lines += [" ".join(mark.center(width + 2) for mark in line) for line in zip_longest(*marks, fillvalue="")]
        return "\n".join(line.rstrip() for line in lines)
