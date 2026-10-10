from __future__ import annotations

from plyllm import LLM, LLMConfig

from plybench.games.builtins import register_builtin_games
from plybench.player.builtins import register_builtin_players
from plybench.registry import Registry

# sent to the providers that route by a prompt cache key (OpenAI, Grok, Mistral); kept from before the LLM
# layer moved to plyllm, so plybench's requests keep sharing one cache
PROMPT_CACHE_KEY = "PlyBench"


class PlyBench:
    def __init__(self, llm_config: LLMConfig | None = None, hf_models: list[str] | None = None, concurrency: int | None = None) -> None:
        self._registry = Registry()
        # hf_models declares which local HuggingFace models this environment uses, and concurrency caps
        # each provider's in-flight requests; both feed the default env config only (an explicit
        # llm_config already carries its own huggingface and concurrency setup)
        config = llm_config if llm_config is not None else LLMConfig.from_env(default_concurrency=concurrency, huggingface_models=hf_models, prompt_cache_key=PROMPT_CACHE_KEY)
        self._llm = LLM(config)
        # download/verify any provider-local resources (e.g. HuggingFace models) up front
        self._llm.bootstrap()

        register_builtin_games(self._registry)
        register_builtin_players(self._registry, self._llm)

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def llm(self) -> LLM:
        return self._llm
