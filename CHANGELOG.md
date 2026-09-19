# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-09-19

### Changed

- Notifications are now delivered by [clankers](https://pypi.org/project/clankers/) (>= 2.0.0)
  instead of the in-repo `NotificationClient`. `NTFY_URL` is now the _server base_ url (e.g.
  `https://ntfy.sh`) and the new `NTFY_TOPIC` is required; a token still comes from `NTFY_TOKEN` and
  an optional request timeout from `NTFY_TIMEOUT`. clankers also reads
  `~/.config/clankers/config.toml`, and its `.env` values take precedence over the process
  environment (the reverse of the old client).
- `scripts/run.py --notify` wraps the run in `clankers.Engage`, which announces the start and reports
  the duration and outcome at the end. Both closing messages are built from the live progress, so a
  success reports the matchups and rounds completed and a crash reports how far the run got before
  the exception.
- Per-matchup progress is sent as a neutral `blastthem` message rather than a success one; only the
  finished run reports success.

### Removed

- `plybench.observability.notifications` (`NotificationClient`), the `PlyBench.notif` property and
  the `PlyBench(notif_enabled=...)` argument. Send notifications with `clankers` directly.
- `requests` as a direct dependency; it now arrives through clankers.

## [1.2.0]

### Added

- Mistral provider behind the `mistral` extra (also included in `all`), reading `MISTRAL_API_KEY`.
  Ships `mistral-medium-3.5` and `mistral-small-4`.
- Per-model rate limits: `ModelLimits(max_concurrent, rps, tpm)` installed with
  `LLM.set_model_limits(provider, model_name, limits)` (and `set_embedding_model_limits` for embedding
  endpoints). The gate sits in the shared dispatch path every client routes its API call through, so
  it applies to all providers. Token pacing reserves an estimate before each call (prompt plus
  `max_tokens`, or the model's `default_output_estimate`) and reconciles it against reported usage;
  `ModelLimits.scaled(factor)` leaves headroom when several processes share an account. No shipped
  model carries a quota — published allowances are account-specific — and the repo-local scripts keep
  theirs in `LIMITS` in `scripts/_shared.py`.
- Embeddings as a first-class capability next to generation: `EmbeddingModel`, `EmbeddingModelConfig`,
  `EmbeddingTask`, `EmbeddingBatch`, `EmbeddingTokens`, plus `LLM.embed`,
  `LLM.get_available_embedding_models`, `LLM.resolve_embedding_model` and
  `LLM.calculate_embedding_cost`. `LLMClient.embed` now owns task formatting, the context guard,
  batching to the model's `max_batch_size` and merging the batches; a provider only implements
  `_embed_batch`.
- Gemini embeddings (`gemini-embedding-2`, with the documented `task:`/`title:` input prefixes per
  `EmbeddingTask`) and OpenAI embeddings (`text-embedding-3-small`, `text-embedding-3-large`).
- Move-level analysis. `MoveRecord`/`collect_moves` pair every judged move with its verdict, branching
  factor and reasoning trace; `MoveMetric` and `MoveFeature` name what is measured and what it is
  binned by; `BenchmarkAnalysis.analyze_partition(partitioner)` and `.analyze_recognition()` split a
  matchup's moves into groups and compare each group against the baseline.
- Recognition analysis: `plybench.analysis.recognition` detects whether a reasoning trace names the
  real game behind an obfuscated variant, exposed as the `recognition_rate` metric and as the
  `by_recognition()` partitioner.
- Studies over a whole experiment: `analyze_scaling` (does token spend slope up with tactical
  sharpness, and does spending more buy accuracy within a difficulty bin) and `compare_games` (per
  model and per opponent, game A minus game B on move quality).
- Statistics: `two_proportion_test`, `mean_difference_test`, `compare_for_family`,
  `combine_independent`/`combine_comparisons`, `linear_fit` and `fit_difference`, plus
  `bundle_for_family` and observation-level pooling of a metric across matchups (`MetricPool`,
  `pooled_bundle`).
- Token accounting, re-exported from `plybench.analysis`: `benchmark_usage`, `matchup_usage`,
  `total_usage`, `group_by`, `entry_cost`, `total_cost` and `model_label` sum what an experiment spent
  from its recorded results, with USD for the providers this process has credentials for.
- Plotting behind the new `viz` extra (matplotlib, also included in `all`):
  `plybench.analysis.visual` with a domain-free core (`Figure`/`Panel`/`Layer`, axes, ticks, legend,
  palette, `render`) and the benchmark glue that feeds it (`build_series`, `StyleEncoder`, metric and
  player/game labels).
- Per-move progress: `BenchmarkCallbacks.move_complete_callback` (bridged to the game loop by
  `BenchmarkCallbacks.for_round`) and `GameTracker.steps_of`. The console callbacks now print which
  rounds are in flight versus queued and a running move count, throttled to one line per matchup every
  `MOVE_LOG_INTERVAL` seconds.
- `NotificationClient.wrap` sends a notification when the wrapped call raises, so a crashed run is
  reported and not only a finished one; `scripts/run.py --notify` wraps the benchmark with it.
- `gemini-3.7-flash`, and `qwen-3.8-27b` on the Metacentrum provider (low/medium/xhigh reasoning).
  The Mistral models and `qwen-3.8-27b` are enabled in the prepared `ttt`, `nim` and `connect_four`
  experiments.
- Scripts: `plot.py` (a metric across games, one line per model), `tokens.py` (token/USD totals for an
  experiment), `import.py` (restore an archive produced by `export.py`), `analyze_scaling.py` and
  `compare_games.py` (the two studies). `analyze.py` gained `--partition` and `--bins`, and `run.py`
  gained `--rounds-concurrency` alongside a `--concurrency` that now means in-flight requests per
  provider.
- README: the paper (arXiv) link, a citation block, PyPI/arXiv/license badges and a section on the two
  rate-limit layers; `Paper` was added to the project URLs.

### Changed

- **Breaking:** `LLMClient.__init__` takes `(models, embedding_models, concurrency)`, and `embed` is now a base-class
  template method — a custom client implements `_embed_batch(model, texts)` returning an
  `EmbeddingBatch` instead of overriding `embed`.
- **Breaking:** `LLM.embed(provider, model_name, texts)` became `LLM.embed(model_config, texts, task)`, taking an
  `EmbeddingModelConfig` and an `EmbeddingTask`.
- **Breaking:** The HuggingFace provider is embeddings-only: `HuggingFaceLLMModel` is now
  `HuggingFaceEmbeddingModel` and `huggingface_models()` is `huggingface_embedding_models()`. The
  supported alias (`sup-simcse-bert`) and the `hf_models=[...]` bootstrap are unchanged.
- `PlyBench(concurrency=...)` and `LLMConfig.from_env(default_concurrency=...)` set the per-provider
  ceiling, with `DEFAULT_CONCURRENCY` (10) exported from `plybench.llm`. The Metacentrum client
  defaults to 4.
- `safe_call` accepts `retry_if` to narrow `retry_errors` for SDKs that funnel every HTTP failure into
  one exception type.
- Public names that analysis callers need: `z_score` (was `_z`), `options_to_string` (was
  `_options_to_string`), and `JudgedStep` (was `_JudgedStep`), which now also carries the branching
  factor and the size of the optimal-action set and is reachable via
  `TurnBasedReplayer.replay_judged`.
- The per-matchup extractor suite moved behind `matchup_suite(tracker, registry, include_fails)`, which
  adds the extractor groups whose preconditions a matchup meets (recognition, optimality/regret).
- The website export writes every metric under its enum value instead of a hard-coded allow-list, so a
  newly added metric reaches the site without touching the exporter; only `moves_per_game` is renamed
  (to `player_moves_per_game`).
- The prepared `nim` experiment plays with the `state` observation type instead of `actions`, and the
  `ttt`/`nim` player lists are ordered by reasoning effort.

### Fixed

- `--concurrency` never reached the providers: it only paced rounds within a matchup, so provider
  semaphores kept their default. Rounds are now paced by `--rounds-concurrency` and `--concurrency`
  configures every provider.
- The Inverse Nim and Story Nim engines built their underlying game from inverse-space pile sizes
  instead of complementing them as `reset()` does, so replaying a recorded game (and therefore
  optimality/regret) started from the wrong position. `build_replayer` also resets the engine before
  the solver sees it.
- Metacentrum structured-output calls looked for a `reasoningreasoning` output item, so reasoning
  traces were silently dropped whenever an output schema was used.

## [1.1.0]

### Added

- Grok provider (xAI) behind the `grok` extra, reusing the OpenAI-compatible endpoint. Reads
  `GROK_API_KEY` and the optional `GROK_BASE_URL` (defaults to `https://api.x.ai/v1`). Ships
  `grok-4.5`, `grok-4.3` and `grok-4.20-reasoning`.
- Anthropic Claude provider behind the `anthropic` extra (also included in `all`), reading
  `CLAUDE_API_KEY`. Ships `claude-fable-5`, `claude-opus-5`, `claude-opus-4.8`, `claude-sonnet-5`,
  `claude-sonnet-4.6` and `claude-haiku-4.5`, mapping reasoning effort onto Anthropic's thinking
  budget (with a numeric budget for models that take no effort parameter).
- Progress notifications through [ntfy.sh](https://ntfy.sh): `NotificationClient` reads `NTFY_URL`
  and `NTFY_TOKEN`, is exposed as `PlyBench.notif` and enabled with `PlyBench(notif_enabled=True)`.
  `scripts/run.py --notify` pushes a message per finished matchup — elapsed time, rounds completed
  and an ETA derived from round throughput — plus a final summary. Send failures are logged as
  warnings instead of interrupting the run.
- Console progress logging during a benchmark: per-matchup start/end plus a running `done/total`
  round counter that skips rounds resumed from an earlier run.
- `max` reasoning effort, and `kimi-k3` on the Metacentrum provider (replacing `kimi-k2.5`).
- The Grok, Claude and Kimi K3 models in the prepared `ttt`, `nim` and `connect_four` experiments.
- `scripts/play.py` prints the player's reasoning trace when one is available.

### Changed

- `console_benchmark_callbacks` moved from `plybench.callbacks.benchmark_callbacks` to
  `plybench.callbacks.console_callbacks`.
- `requests` is now a runtime dependency (used by the notification client).

### Removed

- `plybench.harness` no longer re-exports `Benchmark`, `run_matchup`, `single_game` and
  `order_players_for_game`; import them from `plybench.harness.benchmark` and
  `plybench.harness.matchup` instead.

## [1.0.0]

Initial release of PlyBench.

- Benchmark harness for LLMs, MCTS, optimal solvers, random, and human players.
- Built-in games: tic-tac-toe family, nim family, connect four, breakthrough.
- Analysis pipeline with confidence intervals and minimax-based optimality/regret.
- Open registries for adding custom games and players.
- LLM providers as optional extras (`openai`, `gemini`, `metacentrum`, `huggingface`, and
  `all`); the core install pulls in no provider SDKs, and the router wires up only the providers
  whose dependency is installed (and configured).
- HuggingFace provider that runs models locally (downloading them to the HuggingFace cache) and
  exposes embeddings through `LLM.embed`. Declare the environment's models with
  `PlyBench(hf_models=[...])`; they are downloaded/verified at bootstrap, and requesting a model
  that was not bootstrapped raises a helpful error. Reads `HF_TOKEN` for gated/private models.
  Ships one supported model: `sup-simcse-bert` (`princeton-nlp/sup-simcse-bert-base-uncased`).

[Unreleased]: https://github.com/radajakub/plybench/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/radajakub/plybench/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/radajakub/plybench/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/radajakub/plybench/releases/tag/v1.0.0
