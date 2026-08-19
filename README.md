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

Providers whose SDK is not installed are simply skipped when building `PlyBench()`.

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

LLM providers are configured through environment variables (a `.env` file is loaded automatically).
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

NTFY_URL=...   # optional, enables progress notifications (see Notifications)
NTFY_TOKEN=...
```

### Rate limits

Two independent layers protect against provider throttling:

- **Provider concurrency** (`ProviderSemaphore`) caps in-flight requests per provider. This is the
  only mechanism most providers need — set it with `PlyBench(concurrency=...)` or
  `llm.set_concurrency(provider, n)`.
- **Per-model quotas** (`ModelLimits`) additionally pace a single model by in-flight share, requests
  per second and tokens per minute.

Both layers apply to **every** provider — the gate is enforced in the shared dispatch path each client
routes its API call through, so nothing is provider-specific.

**No model ships with a quota**, because published allowances are account-specific — they depend on
your tier and differ per model. Apply your own when you know them:

```python
from plybench.llm import ModelLimits, Provider

op.llm.set_model_limits(Provider.MISTRAL, "mistral-small-4", ModelLimits(max_concurrent=8, rps=1.67, tpm=100_000))
op.llm.set_model_limits(Provider.MISTRAL, "mistral-small-4", None)  # back to unlimited
```

Every field is optional, so you can pace on tokens alone and leave requests unbounded. Unset limits
mean a model is governed only by the provider semaphore, which is why adding this changed nothing for
existing providers. The repo-local scripts keep their quotas in `LIMITS` in `scripts/_shared.py`, keyed
by provider and model name — a useful pattern to copy, since that file is not part of the installed
package.

Token pacing reserves an estimate before each call (prompt length plus `max_tokens`, or the model's
default output guess) and reconciles it against reported usage once the response lands, so an
inaccurate estimate costs a little throughput rather than correctness. Because quotas are account-wide
while each process paces only itself, pass a `scale` below `1.0` when several runs share an account.
`safe_call`'s backoff remains the backstop for any 429 that slips through.

When per-model shares exceed the provider cap they queue behind it, so set the provider concurrency at
or above their sum (`--concurrency 24` for the two models above) to keep one model from holding slots
the other's quota could use.

### HuggingFace (local models)

The HuggingFace provider runs models locally instead of calling a remote API — it exposes
`embed()` (generation is not supported yet). `hf_models` selects which of the supported models
(see `providers/huggingface/models.py`) this environment uses; they are downloaded/verified into
the local HF cache at bootstrap:

```python
op = PlyBench(hf_models=["sup-simcse-bert"])
resp = await op.llm.embed(Provider.HUGGINGFACE, "sup-simcse-bert", ["hello"])
```

Requesting a supported model that was not part of `hf_models` raises an error telling you to add
it to the bootstrap list. Needs the `huggingface` extra installed.

### Notifications

Long benchmark runs can push progress notifications to [ntfy.sh](https://ntfy.sh) (or any
compatible endpoint) — one message per finished matchup, with elapsed time, rounds completed and an
ETA derived from round throughput, plus a final summary. Set `NTFY_URL` (and `NTFY_TOKEN` for
protected topics) and opt in per run:

```python
op = PlyBench(notif_enabled=True)
op.notif.notify("hello")  # no-op when disabled or unconfigured
```

Notifications are off unless `notif_enabled=True`, and failures are logged as warnings rather than
interrupting the run.

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

The complete experiment definitions from the paper are prepared under
[`experiments/benchmarks/`](experiments/benchmarks/):

| Experiment         | File                | Games                                                                       |
| ------------------ | ------------------- | --------------------------------------------------------------------------- |
| Tic-tac-toe family | `ttt.json`          | `tic_tac_toe`, `modified_tic_tac_toe`, `magic_square`, `story_magic_square` |
| Nim family         | `nim.json`          | `nim`, `modified_nim`, `inverse_nim`, `story_nim`                           |
| Connect Four       | `connect_four.json` | `connect_four`                                                              |

Each file declares the full sweep — the LLM players, the opponents (`random`, `mcts`, `optimal`), and
the number of rounds — with per-item `enabled` toggles so you can narrow a run without editing the sweep.

```bash
git clone https://github.com/radajakub/plybench.git
cd plybench
uv sync

# run a prepared experiment (reads experiments/benchmarks/ttt.json)
uv run python scripts/run.py --experiment ttt

# or an ad-hoc smoke run
uv run python scripts/run.py --name smoke \
    --games tic_tac_toe: \
    --players random:distribution=uniform \
    --opponents optimal:stochastic=True \
    --num-games 10

# push progress notifications for a long run (needs NTFY_URL, see Notifications)
uv run python scripts/run.py --experiment ttt --notify
```

`run.py` logs matchup and round progress to the console as the run proceeds; rounds already
completed by an earlier run are skipped and not logged again.

The LLM matchups require the relevant provider API keys (see [Configuration](#configuration)) and will
incur API cost; bot-vs-bot matchups run offline. Results are written under
`results/benchmarks/<experiment>/`.

Then analyze or export the transcripts:

```bash
uv run python scripts/analyze.py --experiment ttt   # compute per-matchup statistics + confidence intervals
uv run python scripts/export.py --experiment ttt --out ttt.tar.gz   # export results for the PlyBench website
uv run python scripts/play.py --game tic_tac_toe: --i human: --o optimal:stochastic=True  # play interactively
```

For the solver-backed action-error, trace-consistency, and reasoning-mistake pipeline, including the
recommended discovery/evaluation protocol and exact commands, see
[Reasoning and action-error analysis](docs/reasoning-analysis.md).

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
