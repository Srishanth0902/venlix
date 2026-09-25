"""
Core LLM Client and Fallback Plumbing for LLM Manager.

call_llm(prompt, system=None, max_tokens=200, timeout=5, ...) is the single
gateway for all LLM interactions across the system; stream_llm() is its
streaming twin used by the dashboard chat.

Provider chain (the first provider that answers wins):
  1. Google Gemini via the google-genai SDK        -> GEMINI_API_KEY
  2. OpenRouter or any OpenAI-compatible endpoint  -> OPENROUTER_API_KEY (+ OPENROUTER_BASE_URL)
  3. Offline synthesizer (data-grounded, no key)   -> ALLOW_MOCK_FALLBACK=true (default)

Every call is timed and recorded so the dashboard can show latency and
provider health (see get_llm_metrics()).
"""
import os
import json
import time
import logging
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from . import offline
from .copilot import process_copilot_request
from .prompts import COPILOT_SYSTEM_PROMPT, COPILOT_DATA_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Flag for testing fallback behavior programmatically
FORCE_PRIMARY_FAILURE = False


def set_force_primary_failure(value: bool):
    """Utility to test fallback path deterministically during tests or demos."""
    global FORCE_PRIMARY_FAILURE
    FORCE_PRIMARY_FAILURE = value


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read configuration at call time so .env / environment changes are picked up."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _is_valid_key(key: Optional[str]) -> bool:
    """Helper to check if API key is present and not a placeholder."""
    if not key or not key.strip():
        return False
    placeholder_terms = ["your_", "placeholder", "here", "xxx", "change_me"]
    key_lower = key.strip().lower()
    return not any(term in key_lower for term in placeholder_terms)


def _allow_offline() -> bool:
    return FORCE_PRIMARY_FAILURE or _env("ALLOW_MOCK_FALLBACK", "true").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

_calls: deque = deque(maxlen=500)
_calls_lock = threading.Lock()
_call_sink: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("llm_call_sink", default=None)


@contextmanager
def capture_llm_calls():
    """Collect the telemetry of every LLM call made inside this block (threads included)."""
    sink: List[Dict[str, Any]] = []
    token = _call_sink.set(sink)
    try:
        yield sink
    finally:
        _call_sink.reset(token)


def _record(provider: str, model: str, task: str, started: float, ok: bool,
            error: Optional[str] = None, streamed: bool = False) -> Dict[str, Any]:
    entry = {
        "ts": time.time(),
        "provider": provider,
        "model": model,
        "task": task,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "ok": ok,
        "streamed": streamed,
    }
    if error:
        entry["error"] = error[:300]
    with _calls_lock:
        _calls.append(entry)
    sink = _call_sink.get()
    if sink is not None:
        sink.append(entry)
    return entry


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def get_llm_metrics(recent: int = 25) -> Dict[str, Any]:
    """Aggregate latency / error statistics per provider plus the most recent calls."""
    with _calls_lock:
        calls = list(_calls)
    providers: Dict[str, Dict[str, Any]] = {}
    for call in calls:
        stats = providers.setdefault(call["provider"], {"calls": 0, "errors": 0, "_lat": []})
        stats["calls"] += 1
        if call["ok"]:
            stats["_lat"].append(call["latency_ms"])
        else:
            stats["errors"] += 1
    for stats in providers.values():
        latencies = stats.pop("_lat")
        stats["avg_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        stats["p50_ms"] = _percentile(latencies, 50)
        stats["p95_ms"] = _percentile(latencies, 95)
    return {
        "total_calls": len(calls),
        "errors": sum(1 for c in calls if not c["ok"]),
        "providers": providers,
        "recent": list(reversed(calls[-recent:])),
    }


def reset_llm_metrics() -> None:
    with _calls_lock:
        _calls.clear()


def get_provider_status() -> Dict[str, Any]:
    """Which providers are configured, and which one will answer first."""
    gemini_ok = _is_valid_key(_env("GEMINI_API_KEY"))
    openrouter_ok = _is_valid_key(_env("OPENROUTER_API_KEY"))
    chain = [name for name, _, _ in _provider_chain()]
    return {
        "gemini": {"configured": gemini_ok, "model": _env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
                   "forced_failure": FORCE_PRIMARY_FAILURE},
        "openrouter": {"configured": openrouter_ok, "models": _openrouter_models(),
                       "base_url": _env("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)},
        "offline_fallback": _allow_offline(),
        "chain": chain + (["offline"] if _allow_offline() else []),
        "active_provider": chain[0] if chain else ("offline" if _allow_offline() else None),
    }


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def _build_request(prompt: str, system: Optional[str], use_tools: bool) -> Dict[str, Any]:
    """Attach backend data for copilot questions; pass specialized tasks through untouched."""
    if not use_tools:
        return {"system": system, "user": prompt, "apis_called": [], "api_context": {}}

    copilot_result = process_copilot_request(prompt)
    apis_called = copilot_result["apis_called"]
    api_context = copilot_result["context_data"]
    copilot_system = f"{system}\n\n{COPILOT_SYSTEM_PROMPT}" if system else COPILOT_SYSTEM_PROMPT
    if apis_called:
        user = COPILOT_DATA_PROMPT.format(
            question=prompt,
            apis=", ".join(apis_called),
            data=json.dumps(api_context, indent=1, default=str),
        )
    else:
        user = prompt
    return {"system": copilot_system, "user": user, "apis_called": apis_called, "api_context": api_context}


def _trim_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    limit = int(_env("LLM_MAX_HISTORY", "12"))
    cleaned = [
        {"role": "assistant" if m.get("role") in ("assistant", "model") else "user", "content": str(m.get("content", ""))}
        for m in (history or [])
        if str(m.get("content", "")).strip()
    ]
    return cleaned[-limit:] if limit > 0 else []


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _gemini_client(api_key: str, base_url: Optional[str] = None):
    from google import genai
    from google.genai import types
    # GEMINI_BASE_URL is optional (API gateways / proxies); the SDK default is used otherwise.
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(base_url=base_url) if base_url else None)


def _gemini_config(system: Optional[str], max_tokens: int, timeout: float, temperature: float, thinking: bool):
    from google.genai import types
    kwargs: Dict[str, Any] = {
        "system_instruction": system or None,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        "http_options": types.HttpOptions(timeout=int(timeout * 1000)),
        # No tools are passed; disabling AFC avoids its per-call warning and bookkeeping.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    # Gemini 2.5+ "thinks" by default; thinking tokens count against max_output_tokens
    # and add seconds of latency, which truncated or emptied short answers. Disable it
    # unless the operator opts in (GEMINI_THINKING_BUDGET=auto or a token budget).
    budget = _env("GEMINI_THINKING_BUDGET", "0")
    if thinking and budget.lower() not in ("auto", "default", "none"):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(budget))
    return types.GenerateContentConfig(**kwargs)


def _gemini_contents(history: List[Dict[str, str]], user: str) -> List[Dict[str, Any]]:
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in history
    ]
    contents.append({"role": "user", "parts": [{"text": user}]})
    return contents


def _is_thinking_config_error(err: Exception) -> bool:
    message = str(err).lower()
    return "thinking" in message or "budget" in message


def _call_gemini(user: str, system: Optional[str], max_tokens: int, timeout: float,
                 history: List[Dict[str, str]], temperature: float, meta: Dict[str, Any]) -> str:
    client = _gemini_client(_env("GEMINI_API_KEY"), _env("GEMINI_BASE_URL"))
    model = meta["model"] = _env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    contents = _gemini_contents(history, user)
    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=_gemini_config(system, max_tokens, timeout, temperature, True))
    except Exception as err:
        # Some models (e.g. Pro) cannot disable thinking: retry with the model default.
        if not _is_thinking_config_error(err):
            raise
        response = client.models.generate_content(
            model=model, contents=contents, config=_gemini_config(system, max_tokens, timeout, temperature, False))
    text = (response.text or "").strip()
    if not text:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        raise RuntimeError(f"Empty response from Gemini model (finish_reason={reason}).")
    return text


def _stream_gemini(user: str, system: Optional[str], max_tokens: int, timeout: float,
                   history: List[Dict[str, str]], temperature: float, meta: Dict[str, Any]) -> Iterator[str]:
    client = _gemini_client(_env("GEMINI_API_KEY"), _env("GEMINI_BASE_URL"))
    model = meta["model"] = _env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    contents = _gemini_contents(history, user)
    emitted = False
    for thinking in (True, False):
        try:
            config = _gemini_config(system, max_tokens, timeout, temperature, thinking)
            for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
                if chunk.text:
                    emitted = True
                    yield chunk.text
            if not emitted:
                raise RuntimeError("Empty streamed response from Gemini model.")
            return
        except Exception as err:
            if emitted or not thinking or not _is_thinking_config_error(err):
                raise


def _openrouter_models() -> List[str]:
    """OPENROUTER_MODEL may be a comma-separated list; models are tried in order."""
    raw = _env("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    return [m.strip() for m in raw.split(",") if m.strip()]


@lru_cache(maxsize=4)
def _openai_client(api_key: str, base_url: str):
    from openai import OpenAI
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=0,  # the provider chain handles fallback; SDK retries only add latency
        default_headers={"HTTP-Referer": "https://github.com", "X-Title": "Venlix LLM Manager"},
    )


def _openai_messages(system: Optional[str], history: List[Dict[str, str]], user: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


def _call_openrouter(user: str, system: Optional[str], max_tokens: int, timeout: float,
                     history: List[Dict[str, str]], temperature: float, meta: Dict[str, Any]) -> str:
    """Call OpenRouter (or any OpenAI-compatible API) trying each configured model in order."""
    client = _openai_client(_env("OPENROUTER_API_KEY"), _env("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL))
    messages = _openai_messages(system, history, user)
    errors = []
    for model in _openrouter_models():
        meta["model"] = model
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
            text = (response.choices[0].message.content or "").strip() if response.choices else ""
            if text:
                return text
            errors.append(f"{model}: empty response")
        except Exception as err:
            errors.append(f"{model}: {err}")
    raise RuntimeError("; ".join(errors) or "No OpenRouter model configured.")


def _stream_openrouter(user: str, system: Optional[str], max_tokens: int, timeout: float,
                       history: List[Dict[str, str]], temperature: float, meta: Dict[str, Any]) -> Iterator[str]:
    client = _openai_client(_env("OPENROUTER_API_KEY"), _env("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL))
    messages = _openai_messages(system, history, user)
    errors = []
    for model in _openrouter_models():
        meta["model"] = model
        emitted = False
        try:
            stream = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                timeout=timeout, stream=True)
            for chunk in stream:
                piece = chunk.choices[0].delta.content if chunk.choices else None
                if piece:
                    emitted = True
                    yield piece
            if emitted:
                return
            errors.append(f"{model}: empty response")
        except Exception as err:
            if emitted:
                raise
            errors.append(f"{model}: {err}")
    raise RuntimeError("; ".join(errors) or "No OpenRouter model configured.")


ProviderFn = Callable[..., str]
StreamFn = Callable[..., Iterator[str]]


def _provider_chain() -> List[Tuple[str, ProviderFn, StreamFn]]:
    chain: List[Tuple[str, ProviderFn, StreamFn]] = []
    if not FORCE_PRIMARY_FAILURE and _is_valid_key(_env("GEMINI_API_KEY")):
        chain.append(("gemini", _call_gemini, _stream_gemini))
    if _is_valid_key(_env("OPENROUTER_API_KEY")):
        chain.append(("openrouter", _call_openrouter, _stream_openrouter))
    return chain


def _no_provider_error(errors: List[str]) -> RuntimeError:
    if errors:
        return RuntimeError("All LLM providers failed: " + " | ".join(errors))
    return RuntimeError(
        "LLM API Key Configuration Error: Neither GEMINI_API_KEY nor OPENROUTER_API_KEY is configured in .env file. "
        "Please provide a valid API key (GEMINI_API_KEY or OPENROUTER_API_KEY) in .env to invoke live models."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm_detailed(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 200,
    timeout: float = 5.0,
    history: Optional[List[Dict[str, str]]] = None,
    use_tools: Optional[bool] = None,
    task: Optional[str] = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Like call_llm() but returns {"text", "provider", "model", "latency_ms", "apis_called", "errors"}.

    use_tools: attach backend/agent data for copilot questions. Defaults to True only when
    no system prompt is given - callers with their own system prompt are specialized tasks
    (SMS drafting, reply parsing, ...) that must not be polluted with copilot data.
    """
    task = task or offline.infer_task(system, prompt)
    if use_tools is None:
        use_tools = system is None
    request = _build_request(prompt, system, use_tools)
    history_msgs = _trim_history(history)

    logger.debug("LLM request task=%s apis=%s prompt=%r", task, request["apis_called"], request["user"][:500])

    errors: List[str] = []
    for name, call_fn, _ in _provider_chain():
        meta: Dict[str, Any] = {"model": ""}
        started = time.perf_counter()
        try:
            text = call_fn(request["user"], request["system"], max_tokens, timeout, history_msgs, temperature, meta)
        except Exception as err:
            _record(name, meta["model"], task, started, ok=False, error=str(err))
            logger.warning(f"LLM provider {name} failed for task {task}: {err}")
            errors.append(f"{name}: {err}")
            continue
        entry = _record(name, meta["model"], task, started, ok=True)
        return {"text": text, "provider": name, "model": meta["model"], "latency_ms": entry["latency_ms"],
                "apis_called": request["apis_called"], "errors": errors}

    if not _allow_offline():
        raise _no_provider_error(errors)

    started = time.perf_counter()
    text = offline.synthesize(task, prompt, request["api_context"])
    entry = _record("offline", "synthesizer", task, started, ok=True)
    return {"text": text, "provider": "offline", "model": "synthesizer", "latency_ms": entry["latency_ms"],
            "apis_called": request["apis_called"], "errors": errors}


def call_llm(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 200,
    timeout: float = 5.0,
    history: Optional[List[Dict[str, str]]] = None,
    use_tools: Optional[bool] = None,
    task: Optional[str] = None,
    temperature: float = 0.1,
) -> str:
    """
    Unified entry point for all LLM calls.
    Decides when to fetch backend APIs and routes queries to Gemini/OpenRouter or the offline fallback.
    """
    return call_llm_detailed(prompt, system, max_tokens, timeout, history, use_tools, task, temperature)["text"]


def stream_llm(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 1024,
    timeout: float = 30.0,
    history: Optional[List[Dict[str, str]]] = None,
    use_tools: Optional[bool] = None,
    task: str = "chat",
    temperature: float = 0.3,
) -> Iterator[Dict[str, Any]]:
    """
    Streaming variant of call_llm. Yields events:
      {"type": "meta", "apis_called": [...]}
      {"type": "delta", "text": "..."}            (repeated)
      {"type": "done", "provider", "model", "latency_ms", "first_token_ms"}
      {"type": "error", "message": "..."}          (only if nothing could answer)
    Falls through to the next provider if one fails before producing output.
    """
    if use_tools is None:
        use_tools = system is None
    request = _build_request(prompt, system, use_tools)
    history_msgs = _trim_history(history)
    yield {"type": "meta", "apis_called": request["apis_called"]}

    errors: List[str] = []
    for name, _, stream_fn in _provider_chain():
        meta: Dict[str, Any] = {"model": ""}
        started = time.perf_counter()
        first_token_ms = None
        try:
            for piece in stream_fn(request["user"], request["system"], max_tokens, timeout, history_msgs, temperature, meta):
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000.0, 1)
                yield {"type": "delta", "text": piece}
        except Exception as err:
            _record(name, meta["model"], task, started, ok=False, error=str(err), streamed=True)
            logger.warning(f"LLM provider {name} failed while streaming: {err}")
            if first_token_ms is not None:
                yield {"type": "error", "message": f"{name} stopped mid-answer: {err}"}
                return
            errors.append(f"{name}: {err}")
            continue
        entry = _record(name, meta["model"], task, started, ok=True, streamed=True)
        yield {"type": "done", "provider": name, "model": meta["model"], "latency_ms": entry["latency_ms"],
               "first_token_ms": first_token_ms, "errors": errors}
        return

    if not _allow_offline():
        yield {"type": "error", "message": str(_no_provider_error(errors))}
        return

    started = time.perf_counter()
    text = offline.synthesize(task, prompt, request["api_context"])
    first_token_ms = round((time.perf_counter() - started) * 1000.0, 1)
    for i in range(0, len(text), 48):
        yield {"type": "delta", "text": text[i:i + 48]}
    entry = _record("offline", "synthesizer", task, started, ok=True, streamed=True)
    yield {"type": "done", "provider": "offline", "model": "synthesizer", "latency_ms": entry["latency_ms"],
           "first_token_ms": first_token_ms, "errors": errors}
