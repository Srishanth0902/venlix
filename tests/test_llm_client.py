"""LLM client: provider chain, request shape, fallbacks, streaming and telemetry."""
import asyncio

import pytest

from delivery_agent.llm_manager import client
from delivery_agent.llm_manager.client import call_llm, call_llm_detailed, stream_llm, capture_llm_calls


def test_gemini_request_shape(fake_llm):
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}]
    result = call_llm_detailed("What is the capital of France?", history=history, max_tokens=3210)

    assert result["text"] == "Hello from Gemini"
    assert result["provider"] == "gemini"
    assert result["model"] == client.DEFAULT_GEMINI_MODEL
    body = fake_llm.last("gemini")["body"]
    # Thinking disabled so tokens aren't burnt before the answer; token cap forwarded
    thinking = body["generationConfig"]["thinkingConfig"]
    assert thinking.get("thinkingBudget", thinking.get("thinking_budget")) == 0
    assert body["generationConfig"]["maxOutputTokens"] == 3210
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


def test_google_api_key_works_as_the_gemini_key(fake_llm, monkeypatch):
    # Hosts are often given the key under Google's SDK name.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    assert client.get_provider_status()["gemini"]["configured"]
    assert call_llm_detailed("What is the capital of France?")["provider"] == "gemini"
    headers = {k.lower(): v for k, v in fake_llm.last("gemini")["headers"].items()}
    assert headers["x-goog-api-key"] == "test-google-key"


# --- Behaviour seen against the live Gemini API -------------------------------------------

def _gemini_models_called(fake_llm):
    return [r["path"].split("/models/")[1].split(":")[0] for r in fake_llm.requests if "Content" in r["path"]]


def test_short_timeouts_are_raised_to_gemini_minimum(fake_llm):
    # Agent tasks use 6-8 s budgets; the API rejects deadlines under 10 s
    assert call_llm("hi", system="Be brief.", timeout=6.0, task="reply_parse") == "Hello from Gemini"
    assert int(fake_llm.last("gemini")["headers"]["X-Server-Timeout"]) >= 10


def test_small_token_caps_get_headroom_for_hidden_thinking(fake_llm):
    call_llm("hi", max_tokens=100)
    assert fake_llm.last("gemini")["body"]["generationConfig"]["maxOutputTokens"] >= client.GEMINI_MIN_OUTPUT_TOKENS


def test_agent_tasks_use_the_fast_model_and_chat_the_main_model(fake_llm):
    call_llm("hi", system="Draft an SMS.", task="sms_draft")
    call_llm("hi")
    assert _gemini_models_called(fake_llm) == [client.DEFAULT_GEMINI_TASK_MODEL, client.DEFAULT_GEMINI_MODEL]


def test_retired_model_falls_back_and_is_skipped_afterwards(fake_llm, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    fake_llm.gemini_model_errors["gemini-2.5-flash"] = (404, "NOT_FOUND", "This model is no longer available to new users.")
    first = call_llm_detailed("hi")
    second = call_llm_detailed("hi again")
    assert first["model"] == second["model"] == client.DEFAULT_GEMINI_MODEL
    assert _gemini_models_called(fake_llm) == ["gemini-2.5-flash", client.DEFAULT_GEMINI_MODEL, client.DEFAULT_GEMINI_MODEL]
    assert "gemini-2.5-flash" in client.get_provider_status()["gemini"]["unavailable_models"]


def test_quota_exhausted_model_cools_down(fake_llm):
    lite = client.DEFAULT_GEMINI_TASK_MODEL
    fake_llm.gemini_model_errors[lite] = (429, "RESOURCE_EXHAUSTED", "You exceeded your current quota. Please retry in 30s.")
    assert call_llm_detailed("hi", system="SMS", task="sms_draft")["model"] == client.DEFAULT_GEMINI_MODEL
    assert call_llm_detailed("hi", system="SMS", task="sms_draft")["model"] == client.DEFAULT_GEMINI_MODEL
    # The exhausted model was asked once, not on every call
    assert _gemini_models_called(fake_llm).count(lite) == 1


def test_overloaded_models_get_one_more_round(fake_llm, monkeypatch):
    for model in (client.DEFAULT_GEMINI_MODEL, client.DEFAULT_GEMINI_TASK_MODEL):
        fake_llm.gemini_model_errors[model] = (503, "UNAVAILABLE", "This model is currently experiencing high demand.")
    fake_llm.openai_status = 500
    result = call_llm_detailed("What is 2+2?")
    assert result["provider"] == "offline"
    assert len(_gemini_models_called(fake_llm)) == 4  # two models x two rounds
    assert "Answered by the offline engine because Gemini is overloaded" in result["text"]  # says why


def test_flash_lite_generic_thinking_error_is_retried_and_remembered(fake_llm):
    fake_llm.gemini_generic_thinking_error = True
    assert call_llm("hi", system="SMS", task="sms_draft") == "Hello from Gemini"
    assert call_llm("hi", system="SMS", task="sms_draft") == "Hello from Gemini"
    bodies = [r["body"]["generationConfig"] for r in fake_llm.requests if "Content" in r["path"]]
    # first call: rejected with thinking config, retried without; second call skips straight to it
    assert ["thinkingConfig" in b for b in bodies] == [True, False, False]


def test_streaming_falls_back_across_gemini_models(fake_llm):
    fake_llm.gemini_model_errors[client.DEFAULT_GEMINI_MODEL] = (503, "UNAVAILABLE", "high demand")
    events = list(stream_llm("hi there"))
    assert events[-1]["type"] == "done" and events[-1]["model"] == client.DEFAULT_GEMINI_TASK_MODEL
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Hello from Gemini"


def test_offline_reply_explains_quota_instead_of_asking_for_a_key(fake_llm):
    for model in (client.DEFAULT_GEMINI_MODEL, client.DEFAULT_GEMINI_TASK_MODEL):
        fake_llm.gemini_model_errors[model] = (429, "RESOURCE_EXHAUSTED", "You exceeded your current quota.")
    fake_llm.openai_status = 500
    text = call_llm_detailed("Why is the sky blue?")["text"]
    assert "rate limit" in text and "add `GEMINI_API_KEY`" not in text
    assert client.get_provider_status()["gemini"]["cooling_down"]


def test_rejected_key_stops_after_first_model(fake_llm):
    for model in client._gemini_models("chat"):
        fake_llm.gemini_model_errors[model] = (401, "UNAUTHENTICATED", "Request had invalid authentication credentials.")
    fake_llm.openai_status = 500
    text = call_llm_detailed("Why is the sky blue?")["text"]
    assert len(_gemini_models_called(fake_llm)) == 1
    assert "API key was rejected" in text
