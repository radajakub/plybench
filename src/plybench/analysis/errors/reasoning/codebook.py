from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from plybench.common.paths import ReasoningPathBuilder

if TYPE_CHECKING:
    from plybench.analysis.errors.moves import TracedMove

SCOPE_UNIVERSAL = "universal"

# The three levels the taxonomy is cut at, and the whole point of the hierarchy: a code shared across
# every obfuscation of one game says something about the game, while one confined to a single rendering
# says the formulation introduced it. A level is a *claim* recorded on the code -- it is measured against
# where the code actually occurs, never used to decide which moves the code may be applied to.
LEVEL_UNIVERSAL = "universal"  # a reasoning step that would fail the same way in any game
LEVEL_FAMILY = "family"  # tied to the rules underneath, shared by every obfuscation of them
LEVEL_PRESENTATION = "presentation"  # only possible given this particular formulation
LEVELS: tuple[str, ...] = (LEVEL_UNIVERSAL, LEVEL_FAMILY, LEVEL_PRESENTATION)


def game_scope(game_key: str) -> str:
    return f"game:{game_key}"


def family_scope(family: str) -> str:
    return f"family:{family}"


def level_of(scope: str) -> str:
    if scope.startswith("game:"):
        return LEVEL_PRESENTATION
    return LEVEL_FAMILY if scope.startswith("family:") else LEVEL_UNIVERSAL


@dataclass(frozen=True, slots=True)
class Etalon:
    """The reference instance of a code: one real move and the words in its trace that show the mistake.

    A definition says what a code means; an etalon shows it. It is chosen from the code's own examples
    rather than written, so what appears in a table is text a model actually produced."""

    move_uid: str
    evidence: str  # verbatim from that move's trace
    reason: str = ""  # why this instance over the others

    def to_dict(self) -> dict[str, Any]:
        return {"move_uid": self.move_uid, "evidence": self.evidence, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Etalon:
        return cls(move_uid=data["move_uid"], evidence=data["evidence"], reason=data.get("reason", ""))


@dataclass(frozen=True, slots=True)
class Code:
    """One mistake type. `id` is assigned once and never reused or renumbered — annotations reference it,
    so a code can be renamed, its definition sharpened, or the whole code retired without orphaning the
    labels already collected. Retiring is `merged_into`, never deletion."""

    id: str
    name: str
    definition: str  # mechanistic: which reasoning step failed, phrased to travel across game variants
    scope: str = SCOPE_UNIVERSAL  # SCOPE_UNIVERSAL, or game:<key> for a variant-specific leaf
    parent_id: str | None = None  # a leaf hangs off the universal code it specialises
    examples: tuple[str, ...] = ()  # move uids the code was induced from
    inducer: str = ""  # the judge that proposed it: whether a taxonomy is judge-dependent is answerable only if this is recorded
    # the clearest instance: a verbatim trace quote and the move it came from, picked from the examples
    # once the taxonomy has settled. Provenance like `examples`, so it stays out of `version`
    etalon: Etalon | None = None
    merged_into: str | None = None  # set when consolidated into another code

    @property
    def active(self) -> bool:
        return self.merged_into is None

    @property
    def level(self) -> str:
        """Which of the three tiers this code claims to sit at, read off its scope."""
        return level_of(self.scope)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "definition": self.definition,
            "scope": self.scope,
            "parent_id": self.parent_id,
            "examples": list(self.examples),
            "inducer": self.inducer,
            "etalon": self.etalon.to_dict() if self.etalon is not None else None,
            "merged_into": self.merged_into,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Code:
        return cls(
            id=data["id"],
            name=data["name"],
            definition=data["definition"],
            scope=data.get("scope", SCOPE_UNIVERSAL),
            parent_id=data.get("parent_id"),
            examples=tuple(data.get("examples", ())),
            inducer=data.get("inducer", ""),
            etalon=Etalon.from_dict(data["etalon"]) if data.get("etalon") else None,
            merged_into=data.get("merged_into"),
        )


@dataclass
class Codebook:
    """The mistake taxonomy for one experiment: code definitions and their provenance, and nothing else.
    Prevalence is never stored here — it is computed by joining annotations against this at read time, so
    the file stays valid against any annotation set, including a partial re-annotation."""

    experiment: str
    codes: dict[str, Code] = field(default_factory=dict)
    label: str = ""  # empty for the experiment's codebook; set to fork an independent one beside it
    # Every move uid induction has been shown, whether or not it yielded a code. Reports use this
    # provenance to compare uncovered rates on induction-naive and already-seen traces.
    induced: set[str] = field(default_factory=set)
    # The games those moves came from. Hold-out is game-level because adjacent positions in one game are
    # near-identical; a codebook validated on a different move of a game it was induced on is validated on
    # itself. Empty on a codebook written before this was recorded, which `induced_games_known` reports.
    induced_games: set[str] = field(default_factory=set)
    discovery_design: dict[str, int | str] = field(default_factory=dict)

    @property
    def induced_games_known(self) -> bool:
        """Whether the game-level hold-out can be applied. False for a codebook induced before games were
        recorded: the uids are hashes, so the games cannot be recovered, and the completeness figure falls
        back to the weaker move-level exclusion rather than silently claiming the stronger one."""
        return not self.induced or bool(self.induced_games)

    def naive(self, move: "TracedMove") -> bool:
        """Whether induction has seen neither this move nor any other move of its game."""
        return move.uid not in self.induced and move.game_uid not in self.induced_games

    @property
    def version(self) -> str:
        """Content hash of the active definitions. Annotations record it, so a later reading can tell
        which revision of the taxonomy a label was made under."""
        payload = sorted((code.id, code.name, code.definition, code.scope, code.parent_id or "") for code in self.active())
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:12]

    def active(self) -> list[Code]:
        return [code for code in self.codes.values() if code.active]

    def applicable(self, move: "TracedMove") -> list[Code]:
        """Active codes whose declared scope includes this move."""
        from plybench.analysis.errors.reasoning.scope import code_applies

        return [code for code in self.active() if code_applies(code, move)]

    def spine(self) -> list[Code]:
        """The universal tier: codes that are not a specialisation of another one."""
        return [code for code in self.active() if code.parent_id is None]

    def children(self, code_id: str) -> list[Code]:
        return [code for code in self.active() if code.parent_id == code_id]

    def add(self, code: Code) -> Code:
        if code.id in self.codes:
            raise ValueError(f"Code id already used: {code.id}")
        if code.parent_id is not None and code.parent_id not in self.codes:
            raise ValueError(f"Unknown parent for {code.id}: {code.parent_id}")
        self.codes[code.id] = code
        return code

    def record_induced(self, moves: Iterable["TracedMove"]) -> None:
        """Note that induction has seen these moves, and the games they came from. Provenance only, so it
        deliberately does not move the version -- what a code means is unchanged by which traces went past
        the judge."""
        for move in moves:
            self.induced.add(move.uid)
            self.induced_games.add(move.game_uid)

    def add_example(self, code_id: str, move_uid: str) -> None:
        """Record which move an assignment came from. Provenance only — it never changes what the code
        means, so it deliberately does not move the version."""
        code = self.resolve(code_id)
        self.codes[code.id] = replace(code, examples=tuple(dict.fromkeys((*code.examples, move_uid))))

    def set_etalon(self, code_id: str, etalon: Etalon) -> None:
        """Provenance only, like `add_example` — it never changes what the code means, so it deliberately
        does not move the version and cannot invalidate annotations already made."""
        code = self.resolve(code_id)
        self.codes[code.id] = replace(code, etalon=etalon)

    def depth(self, code_id: str) -> int:
        """How many parents sit above this code. 0 on the spine, 2 at the deepest a three-level taxonomy
        can go."""
        depth, current, seen = 0, self.resolve(code_id), set()
        while current.parent_id is not None:
            if current.id in seen:
                raise ValueError(f"Parent cycle at {current.id}")
            seen.add(current.id)
            current = self.resolve(current.parent_id)
            depth += 1
        return depth

    def set_parent(self, code_id: str, parent_id: str) -> None:
        """Hang a code under the more general one it specialises: a presentation-level code under its
        family, a family-level code under the universal reasoning step.

        Three tiers at most, matching the three levels. A deeper chain is refused rather than flattened,
        because a taxonomy whose depth is unbounded has no level to count anything over -- and the whole
        argument of the hierarchy is that a code's level is a fact you can report."""
        code, parent = self.resolve(code_id), self.resolve(parent_id)
        if code.id == parent.id:
            raise ValueError(f"A code cannot be its own parent: {code_id}")
        if self._descends_from(parent.id, code.id):
            raise ValueError(f"{parent.id} already sits under {code.id}, so this would make a cycle")
        if self.depth(parent.id) + 1 + self._height(code.id) > len(LEVELS) - 1:
            raise ValueError(f"Parenting {code.id} under {parent.id} would build a chain deeper than the {len(LEVELS)} levels")
        self.codes[code.id] = replace(code, parent_id=parent.id)

    def _descends_from(self, code_id: str, ancestor_id: str) -> bool:
        current, seen = self.resolve(code_id), set()
        while current.parent_id is not None and current.id not in seen:
            seen.add(current.id)
            current = self.resolve(current.parent_id)
            if current.id == ancestor_id:
                return True
        return False

    def _height(self, code_id: str) -> int:
        """How many tiers hang below this code already."""
        children = self.children(code_id)
        return 1 + max((self._height(child.id) for child in children), default=0) if children else 0

    def set_level(self, code_id: str, scope: str) -> None:
        """Move a code to a different tier. This is a restructuring decision -- it is a claim about what
        the code means and where it can occur -- so unlike `add_example` or `set_etalon` it deliberately
        *does* move the version, and annotations made under the old one stay readable as an older revision."""
        code = self.resolve(code_id)
        self.codes[code.id] = replace(code, scope=scope)

    def resolve(self, code_id: str) -> Code:
        """Follow merges to the code a label should be counted under. Annotations made before a merge stay
        readable — that is the whole reason merging never deletes."""
        seen: set[str] = set()
        current = self.codes.get(code_id)
        if current is None:
            raise KeyError(f"Unknown code: {code_id}")
        while current.merged_into is not None:
            if current.id in seen:
                raise ValueError(f"Merge cycle at {current.id}")
            seen.add(current.id)
            current = self.codes[current.merged_into]
        return current

    def merge(self, source_id: str, target_id: str) -> None:
        if source_id == target_id:
            raise ValueError("A code cannot be merged into itself")
        source, target = self.codes[source_id], self.resolve(target_id)
        if target.id == source_id:
            raise ValueError(f"Merging {source_id} into {target_id} would create a cycle")
        inducers = tuple(dict.fromkeys(filter(None, (target.inducer, source.inducer))))
        self.codes[source_id] = replace(source, merged_into=target.id)
        self.codes[target.id] = replace(target, examples=tuple(dict.fromkeys(target.examples + source.examples)), inducer=", ".join(inducers))

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "label": self.label,
            "version": self.version,
            "codes": [code.to_dict() for code in self.codes.values()],
            "induced": sorted(self.induced),
            "induced_games": sorted(self.induced_games),
            "discovery_design": self.discovery_design,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Codebook:
        codes = [Code.from_dict(code) for code in data.get("codes", [])]
        return cls(
            experiment=data["experiment"],
            codes={code.id: code for code in codes},
            label=data.get("label", ""),
            induced=set(data.get("induced", [])),
            induced_games=set(data.get("induced_games", [])),
            discovery_design=data.get("discovery_design", {}),
        )

    def save(self, path: Path | None = None) -> Path:
        path = path if path is not None else ReasoningPathBuilder().codebook(self.experiment, self.label)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, experiment: str, path: Path | None = None, label: str = "") -> Codebook:
        """Load the experiment's codebook, or start an empty one. Never fails on a missing file: the
        codebook is built up over runs, and an absent one simply means nothing has been induced yet."""
        path = path if path is not None else ReasoningPathBuilder().codebook(experiment, label)
        if not path.exists():
            return cls(experiment=experiment, label=label)
        return replace(cls.from_dict(json.loads(path.read_text())), label=label)


def render_codes(codes: Sequence[Code], show_level: bool = False) -> str:
    """The codebook as the judge sees it: id, name, definition, and the parent for a specialisation.

    Nothing about prevalence — a judge told a code is common will find it more often. `show_level` is off
    for annotation for the same reason: a judge told a code is presentation-level will not report it
    elsewhere, and where a code occurs is the thing being measured. Restructuring turns it on, because
    revising the levels is exactly what that call is for."""
    if not codes:
        return "(empty — every error you find is a new code)"
    lines = []
    for code in codes:
        parent = "" if code.parent_id is None else f" [specialises {code.parent_id}]"
        level = f" [{code.level}" + (f", {code.scope}]" if code.level != LEVEL_UNIVERSAL else "]") if show_level else ""
        lines.append(f"- {code.id} — {code.name}: {code.definition}{parent}{level}")
    return "\n".join(lines)
