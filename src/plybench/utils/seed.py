from __future__ import annotations

import hashlib

# seeds stay below 2**31, which every seeded provider and numpy accept
SEED_LIMIT = 2**31


def derive_seed(*parts: object) -> int:
    # a stable seed from its parts (unlike hash(), the same in every process): the game's from the base seed and the
    # record's path, a player's or the instance generator's from the game's and its role
    digest = hashlib.sha256("/".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % SEED_LIMIT
