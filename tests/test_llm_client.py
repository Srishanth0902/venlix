"""LLM client: provider chain, request shape, fallbacks, streaming and telemetry."""
import asyncio

import pytest

from delivery_agent.llm_manager import client
from delivery_agent.llm_manager.client import call_llm, call_llm_detailed, stream_llm, capture_llm_calls


def test_gemini_request_shape(fake_llm):
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}]
    result = call_llm_detailed("What is the capital of France?", history=history, max_tokens=321)

    assert result["text"] == "Hello from Gemini"
    assert result["provider"] == "gemini"
    assert result["model"] == client.DEFAULT_GEMINI_MODEL
    body = fake_llm.last("gemini")["body"]
    # Thinking disabled so tokens aren't burnt before the answer; token cap forwarded
    thinking = body["generationConfig"]["thinkingConfig"]
    assert thinking.get("thinkingBudget", thinking.get("thinking_budget")) == 0
    assert body["generationConfig"]["maxOutputTokens"] == 321
    # Chat history is sent as proper turns (assistant -> model)
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert "Operations Copilot" in body["systemInstruction"]["parts"][0]["text"]


def test_model_that_cannot_disable_thinking_is_retried(fake_llm):
    fake_llm.gemini_reject_thinking = True
    assert call_llm("hello") == "Hello from Gemini"
    gemini_calls = [r for r in fake_llm.requests if "Content" in r["path"]]
    assert len(gemini_calls) == 2
    assert "thinkingConfig" not in gemini_calls[-1]["body"]["generationConfig"]


def test_gemini_failure_falls_back_to_openrouter(fake_llm):
    fake_llm.gemini_status = 503
    result = call_llm_detailed("hello")
    assert result["provider"] == "openrouter"
    assert result["text"] == "Hello from OpenRouter"
    assert result["errors"] and result["errors"][0].startswith("gemini")
    metrics = client.get_llm_metrics()
    assert metrics["providers"]["gemini"]["errors"] == 1
    assert metrics["providers"]["openrouter"]["calls"] == 1


def test_openrouter_tries_each_configured_model(fake_llm, monkeypatch):
    fake_llm.gemini_status = 503
    monkeypatch.setenv("OPENROUTER_MODEL", "model/a:free, model/b:free")
    calls = []
    original = fake_llm.httpd.RequestHandlerClass._openai

    def flaky(handler, body):
        calls.append(body["model"])
        fake_llm.openai_status = 429 if body["model"] == "model/a:free" else 200
        return original(handler, body)

    monkeypatch.setattr(fake_llm.httpd.RequestHandlerClass, "_openai", flaky)
    result = call_llm_detailed("hello")
    assert calls == ["model/a:free", "model/b:free"]
    assert result["model"] == "model/b:free"


def test_all_providers_down_uses_offline_engine(fake_llm):
    fake_llm.gemini_status = 500
    fake_llm.openai_status = 500
    result = call_llm_detailed("What is 6*7?")
    assert result["provider"] == "offline"
    assert "42" in result["text"]
    assert len(result["errors"]) == 2


def test_offline_disabled_raises_clear_error(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_FALLBACK", "false")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        call_llm("hello")


def test_force_primary_failure_skips_gemini(fake_llm):
    client.set_force_primary_failure(True)
    assert call_llm_detailed("hello")["provider"] == "openrouter"


def test_specialized_tasks_do_not_fetch_copilot_data(fake_llm):
    # A caller-supplied system prompt means a specialized task: no backend data injected
    result = call_llm_detailed("Show failed deliveries today", system="You write SMS messages.")
    assert result["apis_called"] == []
    assert fake_llm.last("gemini")["body"]["contents"][-1]["parts"][0]["text"] == "Show failed deliveries today"

    # The copilot (no system prompt) does attach data
    result = call_llm_detailed("Show failed deliveries today")
    assert "GET /deliveries" in result["apis_called"]
    assert "HERO-001" in fake_llm.last("gemini")["body"]["contents"][-1]["parts"][0]["text"]


@pytest.mark.parametrize("fail_gemini, provider, expected", [
    (False, "gemini", "Hello from Gemini"),
    (True, "openrouter", "Hello from OpenRouter "),
])
def test_streaming(fake_llm, fail_gemini, provider, expected):
    if fail_gemini:
        fake_llm.gemini_status = 503
    events = list(stream_llm("hi there"))
    assert events[0]["type"] == "meta"
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert text == expected
    done = events[-1]
    assert done["type"] == "done" and done["provider"] == provider
    assert done["first_token_ms"] is not None


def test_streaming_offline():
    events = list(stream_llm("hello"))
    assert events[-1]["provider"] == "offline"
    assert "Copilot" in "".join(e["text"] for e in events if e["type"] == "delta")


def test_capture_llm_calls_sees_calls_made_in_worker_threads():
    async def run():
        with capture_llm_calls() as calls:
            await asyncio.to_thread(call_llm, "hi", "You are terse.", 50, 5.0, None, None, "sms_draft")
        return calls

    calls = asyncio.run(run())
    assert len(calls) == 1 and calls[0]["task"] == "sms_draft" and calls[0]["ok"]


def test_provider_status_reports_chain(fake_llm):
    status = client.get_provider_status()
    assert status["chain"] == ["gemini", "openrouter", "offline"]
    assert status["active_provider"] == "gemini"
