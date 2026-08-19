from __future__ import annotations

# The rules an annotating model must follow, shared verbatim by codebook induction and by annotation so
# the codes are induced under exactly the conditions they are later applied under. Rule 1 is the one that
# does the work: a trace is a search transcript, and most of what looks like an error in it is the model
# examining a line and discarding it. Only the reasoning the final move actually rests on is a mistake.
CODING_RULES: tuple[str, ...] = (
    "Code only load-bearing reasoning: claims the final chosen move actually rests on. Reasoning that "
    "does not support the final choice is not a mistake, however wrong it looks in isolation.",
    "Exploring and rejecting a line is normal search, not an error. 'If I take 2 here, then they take 3 "
    "and I lose — so not that' is the model working correctly. Never code a rejected hypothetical.",
    "If the trace makes an error and then catches and repairs it itself, and the final move follows the "
    "repair, record the code with self_corrected=true. It is not an uncorrected mistake and must not be "
    "reported as one — but it is recorded, because how often a model catches itself is a finding.",
    "A repair that is itself wrong, or that the final move contradicts, is not a correction. Code the error that survived into the decision, with self_corrected=false.",
    "Every code needs verbatim evidence quoted from the trace. No quote, no code.",
    "Name the reasoning step that failed, not the game topic. 'Missed the opponent's immediate winning threat' travels across variants; 'forgot the center square' does not.",
    "A move can be correct and the reasoning still wrong, and a move can be wrong with no identifiable "
    "reasoning error. If no load-bearing error is present, return no codes rather than inventing one.",
)


def rules_block() -> str:
    return "\n".join(f"{index}. {rule}" for index, rule in enumerate(CODING_RULES, start=1))
