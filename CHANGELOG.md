# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.0] - 2026-10-10

### Added

- `LLM.calculate_cost(..., batch=True)`: the cost at the provider's Batch API price. Each model carries a
  `batch_ratio` (0.5 for OpenAI, Claude, Gemini and Mistral; 0.8 for `grok-4.3` and `grok-4.20-reasoning`;
  none for `grok-4.5` and Metacentrum, which raise), and Gemini models whose batch cache-read price
  differs from half the standard one carry it as `batch_cached_input_cost`. Nothing sends batch requests
  yet; standard prices are unchanged.
- `LLM.aclose()` closes every provider's connection pool. Await it at the end of the `asyncio.run()` that
  made the calls; the live tests do, which removes their "Event loop is closed" log noise.
- `claude-sonnet-5.5` on the Claude provider ($2.00 / $10.00, cache read $0.10; all five effort
  levels). Thinking cannot be turned off: the API rejects `"disabled"`, and its `"between_tools"`
  mode is not supported by the harness.
- `claude-haiku-5.5` on the Claude provider ($0.10 / $0.50, cache read $0.01, for prompts up to 100k
  tokens; prompts over 100k cost $0.50 / $2.50 and are not modelled). All five effort levels; thinking
  can be turned off at `high` or below. The live tests use it instead of `claude-sonnet-5`.
- `gpt-6.1-sol` on the OpenAI provider ($2.00 / $10.00, cached input $0.10; `low` to `max`).
- `mistral-large-4` on the Mistral provider, at the sale price of $0.68 / $2.09 (cached $0.07).
  Mistral gives no end date for the sale; the list price is $1.36 / $4.18.
- A request timeout for every remote provider: 600 s by default, set per provider with
  `OPENAI_TIMEOUT`, `GROK_TIMEOUT`, `CLAUDE_TIMEOUT`, `GEMINI_TIMEOUT` or `MISTRAL_TIMEOUT` (seconds).
  OpenAI, Grok and Claude used to wait without limit.
- `FailureKind.REFUSAL`, raised by every provider that signals a declined request: a refusal item
  (OpenAI, Grok, Metacentrum), `stop_reason: "refusal"` (Claude) or a safety block (Gemini). Mistral
  has no such signal.
- `LLMCallError.response`: for `unparseable` and `refusal` failures, the answer that did arrive, with
  its text, reasoning and tokens.
- `LLM.served_models(provider)`: the model ids a provider's endpoint lists right now (free; every
  remote provider), to catch a retired or mistyped model before a run.
- `retired=True` on a registry model: it still resolves and costs, so recorded results load, but a new
  call fails with a clear message instead of the provider's 400. The five models e-INFRA no longer
  serves (`deepseek-v4-flash`, `deepseek-v3.2-thinking`, `qwen-3.5-122b`, `glm-5.2`,
  `mistral-small-4`) are marked retired.

### Changed

- **Breaking.** The LLM layer moved to its own package, [plyllm](https://pypi.org/project/plyllm/)
  (1.0.0), which PlyBench now depends on. Replace `from plybench.llm import ...` (and its submodules) with
  `from plyllm import ...`; the API is otherwise the same. The provider extras (`plybench[openai]`, ...,
  `plybench[all]`) install the matching plyllm extras. The LLM entries in this section describe the code
  as it moved, so they hold for plyllm 1.0.0. `make test-models` and `make test-live` moved with it.
- Metacentrum is configured only through `METACENTRUM_API_KEY` and `METACENTRUM_BASE_URL` (plyllm reads
  no `OS_API_KEY` / `OS_BASE_URL`).
- `PlyBench()` passes the prompt cache key `PlyBench` to plyllm (whose default is `PlyLLM`), so OpenAI,
  Grok and Mistral requests keep sharing the cache they used before.

- **Affects results.** `LLMPlayer` records an answer that does not fit the schema, or a refusal, as a
  failed move (`Wrong action format (...)` or `Refusal (...)`) instead of aborting the run. Before, a
  resumed run replayed the game until the model answered, so these failures never reached the results.
  This only applies to the `structured` output strategy and to refusals; no released benchmark uses
  either. With the `text` strategy, an OpenAI, Grok, Metacentrum or Gemini refusal used to be recorded
  as `Wrong action format ()`; it is now `Refusal (...)`.
- Structured output is checked in one place for every provider, and a mismatch raises
  `LLMCallError(unparseable)` with the raw answer attached. Providers send the schema themselves
  instead of using the SDKs' parse helpers; the requests are unchanged.
- Without an explicit concurrency, each provider now uses its own default: 4 in-flight requests for
  Metacentrum (was 10; the shared endpoint answered 429 to 10 parallel calls) and 10 for the rest.
  `LLMConfig.default_concurrency` is now `int | None`, with `None` as the default.
  `--concurrency N` and `PlyBench(concurrency=N)` work as before. Speed only; answers are unchanged.

- `claude-haiku-4.5` sends the pinned snapshot `claude-haiku-4-5-20251001` instead of the
  `claude-haiku-4-5` alias. It is the same model; the models endpoint lists only the snapshot.
- `grok-4.20-reasoning` no longer accepts `reasoning_effort`. xAI documents the parameter only for
  `grok-4.7`, `grok-4.6`, `grok-4.5` and `grok-4.3`; a config that sets it now fails before the request.

- One retry rule for every provider: only 408, 409, 429 and 5xx responses and transport failures are
  retried. OpenAI, Grok, Claude and Metacentrum used to retry every error, so a rejected request
  (400, 401, ...) was sent ten times before it failed.
- Claude sends one request instead of streaming, like every other provider. The timeout now caps the
  whole answer, so raise `CLAUDE_TIMEOUT` for long max-effort runs.
- An option a model cannot honour raises `ValueError` before the request instead of being dropped:
  - `temperature` on a model that rejects it: Grok, the effort-based Claude models, and OpenAI models
    while reasoning (OpenAI and Claude Sonnet 4.6 / Haiku 4.5 take it only with thinking off);
  - `reasoning_effort` with thinking off, except on the Claude models with an effort parameter, where
    it also sets answer length;
  - thinking on without an effort on models that would then not reason (`gpt-5.4`, `gpt-5.4-mini`,
    `gpt-5.4-nano`, whose default effort is `none`, and Metacentrum `gemma-4`);
  - thinking off on models that always reason: `gpt-6-astra`, `gpt-6.1-sol`, the Pro models,
    `gpt-5-mini`, `gpt-5-nano`, `grok-4.5`, `grok-4.20-reasoning`, Metacentrum `gpt-oss-120b` and
    `deepseek-v4.1-flash`.
    No benchmark config is affected: their request parameters are unchanged.
- Metacentrum `mistral-medium-3.5` no longer accepts `thinking_enabled=True`: with it, the model
  returned no reasoning (probe, 2026-10-09), so the flag claimed reasoning that did not happen.
- OpenAI: thinking on asks for the reasoning summary even without an explicit effort; before, the
  trace came back empty in that case.

### Fixed

- Claude: `claude-haiku-4.5` sends `max_tokens` 32000 by default, like the other Claude models, instead
  of 16000. At `reasoning_effort=high` the 16000 cap cut the thinking budget from 16384 to 14976 and left
  1024 tokens for the answer. **Affects results** of `claude-haiku-4.5` at `high`; it has not been run.
- Claude: cache writes are costed at 1.25x the input price, the 5-minute cache-write rate, instead of
  the plain input price. The system prompt is cached on every call, so the first call on a prompt was
  under-charged. `LLMTokens` has a new `cache_write_tokens` field (part of `input_tokens`), and
  `LLMModel` a `cache_write_cost` (`None` = the input price, as on every other provider). Recorded
  results carry no cache split, so their costs do not change.
- Claude: a refusal under a schema is reported as `refusal`. The SDK's parse helper used to fail on
  the refusal text first, so it surfaced as a schema failure.
- Metacentrum: the inline `<think>` block is stripped from schema-enforced answers too.
- Metacentrum: models that get the schema in the prompt (`gemma-4`, `glm-5.3`, `qwen-3.8-27b`) are told
  to fill in an instance of it. **Affects results** of the `structured` strategy on these models: with
  the old wording, `gemma-4` copied the schema back in 5 of 6 game prompts (2026-10-09, 10 prompts
  per wording); the new one answered all 10.
- Mistral: `reasoning_effort` accepts only `high`. The API rejects the other levels the SDK lists
  (`mistral-small-2603`: "Must be one of (none, high)"), so such a config now fails before the request.
- Gemini: connection failures and timeouts raised by the HTTP layer are retried and classified as
  `connection` or `timeout` instead of failing at once as `other`. The client now always uses httpx;
  genai used to switch to aiohttp whenever it was installed (the `anthropic` extra pulls it in).

- Thinking off now turns reasoning off. Before, these models kept reasoning at their default: OpenAI
  (sends effort `none`), `grok-4.3` (effort `none`), `gemini-2.5-flash` and `gemini-2.5-flash-lite`
  (`thinking_budget=0`), and Metacentrum `qwen-3.5`, `qwen-3.8-27b`, `qwen-3.8-flash-next`
  (`enable_thinking: false`; the `thinking: false` sent before had no effect) and `kimi-k3`
  (`thinking: false`). Checked on Metacentrum with reasoning-token counts on 2026-10-09. **Affects
  results** only of configs with thinking off; no benchmark config has one.
- Metacentrum: `temperature` is sent. It was dropped on every model.
- Gemini: cached input tokens are priced at the context-caching rate (for example $0.03 on
  `gemini-2.5-flash`). They were costed at $0. 3.7 Flash and 3.5 Flash stay at $0: Google's pricing page
  lists no price for them. Recorded results are not affected: their steps carry no cached-token split.

### Removed

- The `new_api` flag on OpenAI, Grok and Metacentrum models. It was set on every model, so the only
  thing it did was drop `temperature`.
- `grok-4.7` and `grok-4.6` from the Grok provider. xAI's docs still list them, but the API answers
  404 ("does not exist or your team does not have access to it") for this account, and no run used them.

### Deprecated

- `gpt-5-mini` and `gpt-5-nano` shut down on 2026-12-11; `gpt-5.4-nano` on 2027-04-01.
- `gemini-3.1-flash-lite` shuts down on 2027-05-07. Google recommends `gemini-3.5-flash-lite`.

## [2.1.2] - 2026-10-05

### Changed

- Require clankers >= 3.0.0. Notifications now use neutral labels (`Done`, `Info`, `Failed`); set
  `CLANKERS_THEME=starwars` to restore `Roger, roger`, `Blast them!` and `Uh-oh`.

## [2.1.1] - 2026-10-02

### Changed

- The paper's experiment definitions (`experiments/benchmarks/{ttt,nim,connect_four,connect_four_long}.json`)
  are no longer in the repository; `templates/benchmark.json` is the starting point for a new experiment.
  `experiments/` and `results/` are ignored.

## [2.1.0] - 2026-10-02

### Added

- `ModelConfig.from_string` / `to_string` / `slug`, using the `<provider>:<model>[:<options>]` tail of a
  player config, and `parse_options` / `options_to_string` exported from `plybench.llm`. An unknown
  `reasoning_effort` now fails when the config is parsed rather than at the provider.
- `ReplayedStep` carries the legal and optimal move strings of each replayed move; `TurnBasedReplayer`
  gains `reward_range` and `probe`, and `ReplayerCache` builds one replayer per game config.
- `CIBundle.fmt`, and `TurnBasedState.child` / `children`.

### Changed

- **Breaking:** `JudgedStep` is renamed to `ReplayedStep`; `TurnBasedReplayer.replay_steps` (which
  returned `StepStats`) is renamed to `replay_stats`, and `replay_judged` to `replay_steps`.
  `MoveRecord.from_replayed` replaces `collect_moves`.
- A pydantic `ValidationError` raised inside a provider SDK is now classified as
  `FailureKind.UNPARSEABLE`. New benchmark runs may therefore count some failures under a different
  kind than runs recorded before this release.
- `qwen-3.8-27b` on the Metacentrum provider is marked `weak_structured_output`, and the proxy's
  repeated pydantic serializer warnings are silenced.

### Removed

- **Breaking:** partitioned analysis (`stats/partition.py`, `stats/move_features.py`,
  `BenchmarkAnalysis.analyze_partition` / `analyze_recognition`, `--partition` / `--bins` in
  `scripts/analyze.py`), the `analysis/studies/` package (scaling and cross-game comparison) with
  `scripts/analyze_scaling.py` and `scripts/compare_games.py`, `statistics/regression.py`,
  `CombinedEstimate` / `combine_independent` / `combine_comparisons`, and `recognition.step_recognized`.
- The reasoning-trace and action-error analysis is no longer part of PlyBench; it lives in the
  separate `plybench_analysis` package, which depends on this one.

## [2.0.1] - 2026-09-24

### Added

- `claude-opus-5.5` and `claude-fable-5.1` on the Claude provider, `grok-4.7` and `grok-4.6` on the
  Grok provider (all four take the `xhigh` effort level; `grok-4.7`/`grok-4.6` are the first Grok
  models to do so), and `deepseek-v4-flash` on the Metacentrum provider.
- `max` reasoning effort on the `gpt-5.6-sol`, `gpt-5.6-terra` and `gpt-5.6-luna` models, which
  OpenAI documents but the repository did not accept.

### Changed

- `claude-sonnet-5` pricing corrected to $2.00 / $10.00 per 1M input / output tokens with a $0.20
  cache-read rate. Anthropic made the introductory price permanent instead of raising it to
  $3.00 / $15.00 on 2026-09-01, so recomputed costs for Sonnet 5 runs will drop.
- `gemini-3.6-flash` pricing corrected to $0.75 / $3.75 per 1M input / output tokens, the rate in
  force through 2026-12-31 (it rises to $1.50 / $7.50 on 2027-01-01).

### Removed

- `gemini-3-pro` (`gemini-3-pro-preview`), which Google shut down on 2026-03-09. Use `gemini-3.1-pro`.

### Deprecated

- `gemini-3-flash` (`gemini-3-flash-preview`) is deprecated by Google in favour of `gemini-3.6-flash`;
  it is still served and no shutdown date has been announced.
- `deepseek-v3.2-thinking`, `qwen-3.5-122b`, `glm-5.2` and `mistral-small-4` are no longer served by
  e-INFRA on the Metacentrum provider. The definitions are kept so existing experiment configs and
  recorded results still resolve, but requests against them will fail.

## [2.0.0] - 2026-09-22

### Added

- Training harness for learnable players: run combinations of games, trainees, trainers, training
  schedules and replicates, with an untrained baseline (epoch 0) and frozen evaluation against each
  tester after every training epoch. Checkpoints and game records are saved under `results/training/`
  so interrupted runs can resume.
- `LearnablePlayer` and `CheckpointedParams` extension points for custom players. Training games run
  sequentially on one learner, with per-game `observe` and end-of-epoch `update` hooks; evaluation
  games use frozen checkpoints and can run concurrently.
- `scripts/train.py` accepts a JSON experiment from `experiments/training/` or an inline run, with
  controls for per-provider requests, evaluation rounds and concurrent runs. It prints learning curves
  and supports console progress and optional epoch notifications.
- `TrainingAnalysis` and `LearningCurve` compute per-epoch matchup metrics and the change from the
  untrained baseline. `StepData` provides structured access to recorded player output, including a
  `thinking` fallback for reasoning stored in the full response.
- Pyright type checking in CI and through `make typecheck` (also included in `make lint`).

### Changed

- **Breaking:** Import `Benchmark` and `BenchmarkResults` from `plybench.harness.benchmark.benchmark`
  and `plybench.harness.benchmark.results`, respectively, instead of
  `plybench.harness.benchmark` and `plybench.harness.results`. The `BenchmarkResults` re-export from
  `plybench.harness` remains.
- **Breaking:** The matchup runner is now `run_matchup_concurrent` (with `matchup_callbacks` in place
  of `benchmark_callbacks`). `run_matchup_sequential` supports a shared player and an after-game hook
  for training. `BenchmarkCallbacks` inherits the shared matchup and round hooks from
  `MatchupCallbacks`.
- The LLM router imports provider clients only when their provider is configured. Game interfaces,
  engines and provider clients have more specific type annotations.

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

[Unreleased]: https://github.com/radajakub/plybench/compare/1.3.0...HEAD
[1.3.0]: https://github.com/radajakub/plybench/compare/1.2.0...1.3.0
[1.2.0]: https://github.com/radajakub/plybench/compare/v1.1.0...1.2.0
[1.1.0]: https://github.com/radajakub/plybench/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/radajakub/plybench/releases/tag/v1.0.0
