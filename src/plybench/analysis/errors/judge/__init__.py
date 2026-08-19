"""Shared machinery for the stages that call an LLM: prompt rendering, the cached judge runner, the
per-move pass built on it, stratified sampling, the verdict store and the cost ledger. No stage-specific
vocabulary lives here -- a pass supplies its own schema, prompt and record type."""
