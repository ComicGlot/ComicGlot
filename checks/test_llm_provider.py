"""LLM provider selection (comicglot/utils/llm_utils.py); no network calls."""
import asyncio

import pytest

from comicglot.utils.llm_utils import LLMProcessor


def test_default_provider_is_anthropic_with_published_model(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    p = LLMProcessor(model_name="translation", api_key="x")
    assert p.provider == "anthropic" and p.model_name == "claude-3-7-sonnet-20250219"


def test_openrouter_takes_slugs_and_task_names_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5")
    assert LLMProcessor(model_name="google/gemini-2.5-pro", provider="openrouter", api_key="x").model_name == "google/gemini-2.5-pro"
    assert LLMProcessor(model_name="translation", provider="openrouter", api_key="x").model_name == "openai/gpt-5"
    monkeypatch.delenv("OPENROUTER_MODEL")
    with pytest.raises(ValueError):
        LLMProcessor(model_name="translation", provider="openrouter", api_key="x")


def test_openrouter_batch_runs_requests_and_keeps_order(monkeypatch):
    p = LLMProcessor(model_name="m/x", provider="openrouter", api_key="x")
    monkeypatch.setattr(p, "_openrouter_chat", lambda prompt, model, mt, t: f"{model}:{prompt}:{t}")
    out = asyncio.run(p.batch_process_with_claude(["a", "b", "c"], temperature=0.0))
    assert out == ["m/x:a:0.0", "m/x:b:0.0", "m/x:c:0.0"]


def test_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        LLMProcessor(model_name="x", provider="nope", api_key="x")
