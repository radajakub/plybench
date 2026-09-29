"""The metacentrum client's structured-output routing.

Some hosted models support JSON schema and still cannot have one enforced: under grammar-constrained
decoding gemma-4 loops forever, because the grammar forbids the EOS token and its repetition bias has
nowhere else to go (vllm-project/vllm#40080). `weak_structured_output` marks those, and this is what the
flag does -- until now it was stored on the model and read nowhere at all.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from plybench.llm.providers.metacentrum.client import _json_body, _schema_instructions
from plybench.llm.providers.metacentrum.models import metacentrum_models


class Label(BaseModel):
    code_id: str = Field(description="id of the code, or empty when none covers it")


class Answer(BaseModel):
    labels: list[Label]
    notes: str


def _model(name: str):
    return next(model for model in metacentrum_models() if model.model_name == name)


def test_the_models_that_cannot_have_a_schema_enforced_are_marked():
    assert _model("gemma-4").weak_structured_output
    assert not _model("qwen-3.8-27b").weak_structured_output, "qwen returns valid JSON under enforcement; only the affected models are routed around it"
    assert _model("gemma-4").can_use_json_schema, "the model does support structured output -- enforcing it is what breaks"


def test_the_schema_is_asked_for_in_the_prompt_with_its_field_descriptions():
    """vLLM's own guide: under guided decoding "the model does not see the schema or its field
    descriptions". Asking in the prompt is what puts them back, so the descriptions must survive."""
    instructions = _schema_instructions("You label reasoning mistakes.", Answer)

    assert instructions.startswith("You label reasoning mistakes.")
    assert "id of the code, or empty when none covers it" in instructions
    assert json.loads(instructions.split("schema, and nothing else -- no prose, no code fence:\n")[1])["$defs"]


def test_a_fenced_answer_is_unwrapped_before_anything_reads_it_as_json():
    assert _json_body('```json\n{"labels": [], "notes": ""}\n```') == '{"labels": [], "notes": ""}'
    assert _json_body('```\n{"labels": []}\n```') == '{"labels": []}'
    assert _json_body('  {"labels": []}  ') == '{"labels": []}'
    assert Answer.model_validate_json(_json_body('```json\n{"labels": [], "notes": "x"}\n```')).notes == "x"


def test_the_registry_only_names_models_the_endpoint_serves():
    """`/v1/models` on 2026-09-28 had no deepseek-v4-flash; it serves deepseek-v4.1-flash. A stale name in
    the active block fails with a 400 that reads like an outage rather than a typo."""
    served = {model.model_string for model in metacentrum_models()}
    assert "deepseek-v4.1-flash" in served
    assert _model("gemma-4").model_string == "gemma4"
