"""The command-line tools: plybench-run and plybench-train are installed with the package, the repo-local scripts still run, and
the account's per-model limits come from its own JSON file, never from the package."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from plyllm import ModelLimits, Provider

from plybench.app import PlyBench
from plybench.cli import run, train
from plybench.cli.limits import LIMITS_ENV, limits_path, load_limits
from plybench.cli.shared import apply_model_limits

REPO = Path(__file__).parent.parent


def test_the_tools_are_installed_as_console_scripts():
    scripts = {entry_point.name: entry_point.value for entry_point in importlib.metadata.entry_points(group="console_scripts")}
    assert scripts["plybench-run"] == "plybench.cli.run:main"
    assert scripts["plybench-train"] == "plybench.cli.train:main"


@pytest.mark.parametrize("tool", [run, train])
def test_each_tool_takes_the_limits_and_the_quiet_flag(tool, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["tool", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        tool.main()
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--limits" in out and "--quiet" in out


@pytest.mark.parametrize("script", ["run.py", "train.py", "analyze.py"])
def test_the_repo_local_scripts_still_run(script):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--help"], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert result.returncode == 0, result.stderr


def test_the_example_limits_load():
    limits = load_limits(REPO / "limits.example.json")
    assert limits[Provider.MISTRAL]["mistral-medium-3.5"] == ModelLimits(max_concurrent=16, rps=16.67, tpm=500_000)


def test_limits_name_a_bad_provider_or_field(tmp_path):
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({"nowhere": {"m": {"rps": 1}}}))
    with pytest.raises(ValueError, match="unknown provider 'nowhere'"):
        load_limits(path)
    path.write_text(json.dumps({"mistral": {"m": {"requests": 1}}}))
    with pytest.raises(ValueError, match="fields are max_concurrent, rps, tpm"):
        load_limits(path)


def test_the_limits_file_is_the_flag_then_the_environment_then_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(LIMITS_ENV, raising=False)
    assert limits_path() is None

    Path("limits.json").write_text("{}")
    assert limits_path() == Path("limits.json")
    monkeypatch.setenv(LIMITS_ENV, "/elsewhere/limits.json")
    assert limits_path() == Path("/elsewhere/limits.json")
    assert limits_path(Path("given.json")) == Path("given.json")


def test_limits_apply_to_available_providers_only_and_scale():
    calls = []
    llm = SimpleNamespace(available_providers=[Provider.MISTRAL], set_model_limits=lambda *call: calls.append(call))
    limits = {Provider.MISTRAL: {"m": ModelLimits(max_concurrent=4, rps=2.0)}, Provider.OPENAI: {"g": ModelLimits(rps=1.0)}}
    apply_model_limits(cast(PlyBench, SimpleNamespace(llm=llm)), limits, scale=0.5)
    assert calls == [(Provider.MISTRAL, "m", ModelLimits(max_concurrent=4, rps=1.0))]
