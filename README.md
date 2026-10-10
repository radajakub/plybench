# PlyBench

PlyBench is a benchmark suite for evaluating the performance of LLMs and LLM agents in simple,
fully-observable game environments. It pits players (LLMs, MCTS, optimal solvers, random, or human)
against each other across a matrix of games and records every step for later analysis.

**Paper:** [Towards Improving Sequential Decision-Making in LLM Agents via Experience
Memory](https://arxiv.org/abs/2608.03420) (arXiv:2608.03420) — see [Citation](#citation).

**Package:** [pypi.org/project/plybench](https://pypi.org/project/plybench/)

**Live results** for a selection of models and games are available at
[plybench.jakubrada.com](https://plybench.jakubrada.com).

[![arXiv](https://img.shields.io/badge/arXiv-2608.03420-b31b1b.svg)](https://arxiv.org/abs/2608.03420)
[![PyPI](https://img.shields.io/pypi/v/plybench.svg)](https://pypi.org/project/plybench/)
[![Python](https://img.shields.io/pypi/pyversions/plybench.svg)](https://pypi.org/project/plybench/)
[![License](https://img.shields.io/pypi/l/plybench.svg)](LICENSE)
[![CI](https://github.com/radajakub/plybench/actions/workflows/ci.yml/badge.svg)](https://github.com/radajakub/plybench/actions/workflows/ci.yml)
[![Publish](https://github.com/radajakub/plybench/actions/workflows/publish.yml/badge.svg)](https://github.com/radajakub/plybench/actions/workflows/publish.yml)

## Installation

PlyBench is on PyPI as [`plybench`](https://pypi.org/project/plybench/):

```bash
pip install plybench
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add plybench
```

Requires Python 3.12+.

Each LLM provider SDK is an optional extra — install only the ones you need (or `all`):

```bash
pip install "plybench[openai]"         # OpenAI
pip install "plybench[gemini]"         # Google Gemini
pip install "plybench[grok]"           # Grok (OpenAI-compatible endpoint)
pip install "plybench[anthropic]"      # Anthropic Claude
pip install "plybench[mistral]"        # Mistral
pip install "plybench[metacentrum]"    # Metacentrum (OpenAI-compatible endpoint)
pip install "plybench[huggingface]"    # local HuggingFace models (torch + transformers)
pip install "plybench[all]"            # everything
```

The extras install the matching [plyllm](https://pypi.org/project/plyllm/) extra. Providers whose SDK is
not installed are simply skipped when building `PlyBench()`.

## Quickstart

Building an `PlyBench` object is the one-stop bootstrap: it creates an instance-scoped registry,
registers the built-in games and players, and wires up the LLM router.

```python
import asyncio

from plybench import PlyBench
from plybench.harness.benchmark import Benchmark

op = PlyBench()  # reads provider keys from the environment (see Configuration)

benchmark = Benchmark(
    experiment="quickstart",
    op=op,
    game_configs=["tic_tac_toe:"],
    player_configs=["random:distribution=uniform"],
    opponent_configs=["optimal:stochastic=True"],
    num_games=10,
)

results = asyncio.run(benchmark.run())
```

Runs are **resumable** and written under `results/benchmarks/<experiment>/` in the current working
directory; re-running skips already-completed rounds.

### Config strings

Games and players are addressed by `name:key=value:key=value` strings, e.g.
`llm:actions:text:openai:gpt-5:thinking_enabled=True` or `random:distribution=uniform`.

**Built-in games:** `tic_tac_toe`, `modified_tic_tac_toe`, `magic_square`, `story_magic_square`,
`nim`, `modified_nim`, `inverse_nim`, `story_nim`, `connect_four`, `breakthrough`.

**Built-in players:** `human`, `random`, `mcts`, `optimal`, `llm`.

Extend either set at runtime via `op.registry.register_game(...)` / `op.registry.register_player(...)`.

## Configuration

Model calls go through [plyllm](https://github.com/radajakub/plyllm), the LLM package PlyBench uses (it
was `plybench.llm` up to PlyBench 2.1.2). Import LLM types from it, e.g. `from plyllm import ModelLimits, Provider`.
Its README covers rate limits, retries, structured output, costs and the HuggingFace provider; this
section lists only what PlyBench sets.

Providers are configured through environment variables (a `.env` file is loaded automatically).
`PlyBench()` self-disables any provider whose key is absent, so bot-only benchmarks run offline.
See [`.env.example`](.env.example):

```bash
OPENAI_API_KEY=...
OPENAI_ORGANIZATION=...
OPENAI_PROJECT=...

GEMINI_API_KEY=...
GEMINI_PROJECT=...

GROK_API_KEY=...
GROK_BASE_URL=...   # optional, defaults to https://api.x.ai/v1

CLAUDE_API_KEY=...

MISTRAL_API_KEY=...

METACENTRUM_BASE_URL=...
METACENTRUM_API_KEY=...

HF_TOKEN=...   # only for gated/private HuggingFace models

NTFY_URL=...     # optional, ntfy server base url -- enables progress notifications (see Notifications)
NTFY_TOPIC=...
NTFY_TOKEN=...
```

`PlyBench()` builds its LLM router from these variables, with three settings of its own:

- `PlyBench(concurrency=...)` caps in-flight requests per provider (`None` keeps each provider's default).
- `PlyBench(hf_models=[...])` names the local HuggingFace models to download and verify at startup.
- Requests use the prompt cache key `PlyBench` (OpenAI, Grok and Mistral), so runs share one provider cache.

Pass `PlyBench(llm_config=LLMConfig(...))` to configure plyllm yourself instead; the three settings
above then come from that config. Per-model quotas are account-specific, so none ship with the package.
Set them with `op.llm.set_model_limits(...)`; the repo-local scripts keep theirs in `LIMITS` in
`scripts/_shared.py`, keyed by provider and model name.

### Notifications

Long benchmark runs can push progress notifications to [ntfy.sh](https://ntfy.sh) (or any
compatible endpoint) through [clankers](https://pypi.org/project/clankers/). Set `NTFY_URL` (the
server base url) and `NTFY_TOPIC`, plus `NTFY_TOKEN` for protected topics, then opt in per run with
`scripts/run.py --notify`. A run then reports:

- its start, so you know the job actually launched;
- every finished matchup, as a neutral message with elapsed time, matchups and rounds completed and
  an ETA derived from round throughput;
- its outcome — the matchups and rounds completed on success, or how far it got plus the exception
  on a crash — with the total duration attached.

PlyBench itself sends nothing; use clankers directly for your own messages:

```python
import clankers

clankers.blastthem("halfway there")  # neutral; rogerroger succeeded, uhoh failed

with clankers.Engage("my experiment", success=lambda: f"my experiment: {len(rows)} rows"):
    ...  # reports the start, the duration and the outcome, a crash included
```

clankers reads its configuration from `.env`, `NTFY_`- and `CLANKERS_`-prefixed environment
variables and `~/.config/clankers/config.toml`. Notifications use neutral labels (`Done`, `Info`,
`Failed`); set `CLANKERS_THEME=starwars` for the Star Wars ones. A missing configuration or an unreachable server is logged as a
warning and never interrupts the run.

## Extending PlyBench

Games and players are **open registries** on `op.registry` — you can add your own from your own code
without modifying the package. Each is registered as a spec that pairs a config-string key with the
classes that implement it.

### Adding a player

Implement two things:

1. A `PlayerParams` subclass (`configs/player_params.py`) — the parsed form of your config string.
   Implement `from_string` / `to_string` / `path_suffix`. Reuse `NoGameParams`-style emptiness if your
   player is parameterless.
2. A `Player` subclass (`player/player.py`) — implement `initialize_policy` (one-time setup per game),
   the async `__call__` (given the game, an `InterfaceObservation`, and the legal `InterfaceAction`s,
   return a `PlayerOutput` — set `action=None` to forfeit), and `format_llm_output`.

Optionally, attach a `PlayerTracker` to persist extra per-step data onto each recorded `GameStep`.

Then register a `PlayerSpec`:

```python
from dataclasses import dataclass

from plybench import PlyBench
from plybench.configs.player_config import PlayerConfig
from plybench.configs.player_params import PlayerParams
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.player.spec import PlayerSpec


@dataclass(frozen=True, eq=True)
class FirstMoveParams(PlayerParams):
    @classmethod
    def from_string(cls, params_string: str) -> "FirstMoveParams":
        return cls()

    def to_string(self) -> str:
        return ""

    @property
    def path_suffix(self) -> str:
        return ""


class FirstMovePlayer(Player):
    def initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        pass

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        return PlayerOutput(action=legal_moves[0] if legal_moves else None)

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""


op = PlyBench()
op.registry.register_player(PlayerSpec("first", FirstMoveParams, lambda game, cfg, pid: FirstMovePlayer(cfg, pid)))
# usable anywhere as the config string "first:"
```

### Adding a game

Games are backed by [OpenSpiel](https://github.com/google-deepmind/open_spiel): the underlying game must
be loadable by `pyspiel.load_game(...)` (a built-in OpenSpiel game or a custom game you register with
OpenSpiel). A new variant implements the same set of classes the built-ins do — use any game under
[`src/plybench/games/`](src/plybench/games/) (e.g. `tic_tac_toe/tic_tac_toe.py`) as a template:

- **`TurnBasedGame`** — binds a registry `game_type` key to an OpenSpiel `game_name`.
- **`InterfaceTransformer`** — renders state/actions for both display and the LLM prompt.
- **`InterfaceAction`** / **`InterfaceObservation`** — convert to and from OpenSpiel (`from_openspiel` /
  `to_openspiel`).
- **`PromptAdapter`** — the game's head prompt and expected action format.
- **`TurnBasedEngine`** — wires all of the above together (`engine_factory: GameConfig -> TurnBasedEngine`).
- Optionally a **`GameParams`** subclass for parameterized variants (or reuse `NoGameParams`).

Then register a `GameSpec`:

```python
from plybench.configs.game_params import NoGameParams
from plybench.games.spec import GameSpec

op.registry.register_game(
    GameSpec(
        key="my_game",
        params_cls=NoGameParams,  # or a custom GameParams subclass
        engine_factory=MyGameEngine,  # GameConfig -> TurnBasedEngine
        solvable=False,  # True enables minimax optimality/regret analysis (small trees only)
    )
)
# usable anywhere as the config string "my_game:"
```

Registered games and players work everywhere the built-ins do — in `Benchmark`, the analysis pipeline,
and the scripts.

## Reproducing the paper results

The experiment scripts under [`scripts/`](scripts/) are **not** part of the installed package — they
are the tooling used to produce and reproduce the results of the
[paper](https://arxiv.org/abs/2608.03420) from a repository checkout. Use them
when you want to re-run the exact benchmarks the paper reports, extend them with new models, or run the
analysis and export pipelines on the resulting transcripts. Everything is resumable, so an interrupted
run continues where it left off.

An experiment is a JSON file under `experiments/benchmarks/<name>.json`. Start from
[`templates/benchmark.json`](templates/benchmark.json): it declares the sweep — the games, the players,
the opponents (`random`, `mcts`, `optimal`), and the number of rounds — with per-item `enabled` toggles
so you can narrow a run without editing the sweep. The template runs offline as it stands.

```bash
git clone https://github.com/radajakub/plybench.git
cd plybench
uv sync

# create an experiment from the template and run it (reads experiments/benchmarks/my_experiment.json)
mkdir -p experiments/benchmarks
cp templates/benchmark.json experiments/benchmarks/my_experiment.json
uv run python scripts/run.py --experiment my_experiment

# or an ad-hoc smoke run
uv run python scripts/run.py --name smoke \
    --games tic_tac_toe: \
    --players random:distribution=uniform \
    --opponents optimal:stochastic=True \
    --num-games 10

# push progress notifications for a long run (needs NTFY_URL and NTFY_TOPIC, see Notifications)
uv run python scripts/run.py --experiment my_experiment --notify
```

`run.py` logs matchup and round progress to the console as the run proceeds; rounds already
completed by an earlier run are skipped and not logged again.

The LLM matchups require the relevant provider API keys (see [Configuration](#configuration)) and will
incur API cost; bot-vs-bot matchups run offline. Results are written under
`results/benchmarks/<experiment>/`.

Then analyze or export the transcripts:

```bash
uv run python scripts/analyze.py --experiment my_experiment   # compute per-matchup statistics + confidence intervals
uv run python scripts/export.py --experiment my_experiment --out my_experiment.tar.gz   # export results for the PlyBench website
uv run python scripts/play.py --game tic_tac_toe: --i human: --o optimal:stochastic=True  # play interactively
```

The full result set is large (tens of thousands of game transcripts) and is not stored in this
repository. <!-- TODO: link the archived dataset (Zenodo DOI / Hugging Face) once published. -->

## Citation

If you use PlyBench in your research, please cite the paper (arXiv preprint for now — this entry will
be updated once the proceedings version is out):

```bibtex
@misc{rada2026plybench,
  title         = {Towards Improving Sequential Decision-Making in LLM Agents via Experience Memory},
  author        = {Rada, Jakub and Lis{\'y}, Viliam},
  year          = {2026},
  eprint        = {2608.03420},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.03420}
}
```

## License

[MIT](LICENSE) © Jakub Rada
