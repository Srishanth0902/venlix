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
import re
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
from ..env_utils import env_float, env_int

logger = logging.getLogger(__name__)

# Google-maintained aliases that always point at the current Flash / Flash-Lite models, so a
# retired model version can never take the agent down again (gemini-1.5-flash and then
# gemini-2.5-flash were both closed to new API keys). They are also appended after any
# configured models as fallbacks.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"            # Copilot chat: best quality
DEFAULT_GEMINI_TASK_MODEL = "gemini-flash-lite-latest"  # short agent tasks: ~4x faster
# Agent-internal tasks with short, structured outputs; they run on GEMINI_TASK_MODEL.
FAST_TASKS = {"sms_draft", "customer_reply_sim", "reply_parse", "decision_summary", "exception_analysis"}
# The Gemini API rejects request deadlines under 10 s ("Minimum allowed deadline is 10s"), and the
# SDK forwards the timeout as that deadline; shorter agent-step limits are enforced by the agent.
GEMINI_MIN_TIMEOUT = 10.0
# Gemini 3.x models may still spend hidden "thought" tokens; a small cap then ends the answer
# before any text (finish_reason=MAX_TOKENS). Prompts, not the cap, keep answers short.
GEMINI_MIN_OUTPUT_TOKENS = 1024
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
        "gemini": {"configured": gemini_ok, "model": _gemini_models("chat")[0],
                   "models": _gemini_models("chat"), "task_models": _gemini_models("sms_draft"),
                   "unavailable_models": sorted(_unavailable_models),
                   "cooling_down": {m: round(t - time.monotonic()) for m, t in _quota_cooldown_until.items()
                                    if t > time.monotonic()},
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
    limit = env_int("LLM_MAX_HISTORY", 12)
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


def _split_models(raw: Optional[str]) -> List[str]:
    return [m.strip() for m in (raw or "").split(",") if m.strip()]


# Learned at runtime so the same failing request is never repeated:
_unavailable_models: set = set()             # 404: retired / not enabled for this key
_models_without_thinking_off: set = set()     # reject thinking_budget=0 (e.g. Flash-Lite, Pro)
_quota_cooldown_until: Dict[str, float] = {}  # 429: skip the model until its quota refills


def reset_model_state() -> None:
    _unavailable_models.clear()
    _models_without_thinking_off.clear()
    _quota_cooldown_until.clear()


def _gemini_models(task: Optional[str]) -> List[str]:
    """
    Models to try in order: the configured list for this kind of task and its 'latest' alias,
    then the other list and alias. Free-tier quotas are per model, so every extra model adds capacity.
    """
    chat_models = _split_models(_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    task_models = _split_models(_env("GEMINI_TASK_MODEL", DEFAULT_GEMINI_TASK_MODEL))
    if task in FAST_TASKS:
        candidates = task_models + [DEFAULT_GEMINI_TASK_MODEL] + chat_models + [DEFAULT_GEMINI_MODEL]
    else:
        candidates = chat_models + [DEFAULT_GEMINI_MODEL] + task_models + [DEFAULT_GEMINI_TASK_MODEL]
    ordered = [m for m in dict.fromkeys(candidates) if m not in _unavailable_models]
    now = time.monotonic()
    ready = [m for m in ordered if _quota_cooldown_until.get(m, 0.0) <= now]
    # If every model is cooling down, still try them (soonest refill first) rather than give up.
    return ready or sorted(ordered, key=lambda m: _quota_cooldown_until.get(m, 0.0))


def _thinking_off_requested() -> bool:
    return _env("GEMINI_THINKING_BUDGET", "0").lower() not in ("auto", "default", "none")


def _gemini_config(system: Optional[str], max_tokens: int, timeout: float, temperature: float, thinking: bool):
    from google.genai import types
    kwargs: Dict[str, Any] = {
        "system_instruction": system or None,
        "max_output_tokens": max(max_tokens, GEMINI_MIN_OUTPUT_TOKENS),
        "temperature": temperature,
        "http_options": types.HttpOptions(timeout=int(max(timeout, GEMINI_MIN_TIMEOUT) * 1000)),
        # No tools are passed; disabling AFC avoids its per-call warning and bookkeeping.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    # Gemini 2.5+ "thinks" by default; thinking tokens count against max_output_tokens
    # and add seconds of latency, which truncated or emptied short answers. Disable it
    # unless the operator opts in (GEMINI_THINKING_BUDGET=auto or a token budget).
    if thinking and _thinking_off_requested():
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=env_int("GEMINI_THINKING_BUDGET", 0))
    return types.GenerateContentConfig(**kwargs)


def _gemini_contents(history: List[Dict[str, str]], user: str) -> List[Dict[str, Any]]:
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in history
    ]
    contents.append({"role": "user", "parts": [{"text": user}]})
    return contents


def _error_code(err: Exception) -> Optional[int]:
    code = getattr(err, "code", None)
    return code if isinstance(code, int) else None


def _is_thinking_config_error(err: Exception) -> bool:
    # Most models name the setting in the error; Flash-Lite only returns the generic
    # "Request contains an invalid argument." (other 400s, e.g. a too-short deadline, are real errors).
    message = str(err).lower()
    return "thinking" in message or "budget" in message or (
        _error_code(err) == 400 and "request contains an invalid argument" in message)


def _is_overloaded(err: Exception) -> bool:
    """Server-side overload / timeouts: worth one retry after a short pause (unlike quota 429s)."""
    message = str(err).lower()
    return _error_code(err) in (500, 502, 503, 504) or "timed out" in message or "timeout" in message


def _quota_retry_seconds(err: Exception) -> float:
    """Seconds until a 429'd model may be used again (from Google's RetryInfo / message)."""
    match = re.search(r"retry(?:_?delay)?['\"]?\s*(?:in|:)\s*['\"]?(\d+(?:\.\d+)?)s", str(err), re.IGNORECASE)
    seconds = float(match.group(1)) if match else 20.0
    return min(max(seconds, 1.0), 120.0)


def _short_error(err: Exception) -> str:
    code, status = _error_code(err), getattr(err, "status", None)
    message = getattr(err, "message", None) or str(err)
    return f"{code} {status}: {message}"[:160] if code else str(err)[:160]


def _thinking_attempts(model: str) -> Tuple[bool, ...]:
    if not _thinking_off_requested() or model in _models_without_thinking_off:
        return (False,)
    return (True, False)


def _gemini_try_models(task: Optional[str], meta: Dict[str, Any], attempt_fn: Callable[[str, bool], Any]) -> Any:
    """
    Run attempt_fn(model, thinking) over the model list: a model that rejects thinking_budget=0 is
    retried without it; a retired model (404) is skipped from then on; a model out of quota (429) is
    skipped until Google says it refills; overload (5xx) moves to the next model, and if every model
    was overloaded the list is tried once more after a short pause.
    """
    errors: List[str] = []
    for round_no in range(2):
        all_overloaded = True
        for model in _gemini_models(task):
            meta["model"] = model
            for thinking in _thinking_attempts(model):
                try:
                    result = attempt_fn(model, thinking)
                    if not thinking and _thinking_off_requested():
                        _models_without_thinking_off.add(model)
                    return result
                except Exception as err:
                    if thinking and _is_thinking_config_error(err):
                        continue  # same model, default thinking
                    if _error_code(err) == 404:
                        _unavailable_models.add(model)
                        logger.warning(f"Gemini model {model} is not available to this key; skipping it from now on.")
                    elif _error_code(err) == 429:
                        wait = _quota_retry_seconds(err)
                        _quota_cooldown_until[model] = time.monotonic() + wait
                        logger.warning(f"Gemini model {model} hit its quota; skipping it for {wait:.0f}s.")
                    all_overloaded = all_overloaded and _is_overloaded(err)
                    errors.append(f"{model}: {_short_error(err)}")
                    if _error_code(err) in (401, 403):
                        # A rejected / revoked key fails on every model: don't try the rest
                        raise RuntimeError("; ".join(errors)) from err
                    break
        if round_no == 0 and errors and all_overloaded:
            time.sleep(env_float("GEMINI_RETRY_DELAY", 0.8))
            continue
        break
    raise RuntimeError("; ".join(errors) or "No Gemini model available.")


def _call_gemini(user: str, system: Optional[str], max_tokens: int, timeout: float,
                 history: List[Dict[str, str]], temperature: float, meta: Dict[str, Any]) -> str:
    client = _gemini_client(_env("GEMINI_API_KEY"), _env("GEMINI_BASE_URL"))
    contents = _gemini_contents(history, user)

    def attempt(model: str, thinking: bool) -> str:
        response = client.models.generate_content(
            model=model, contents=contents, config=_gemini_config(system, max_tokens, timeout, temperature, thinking))
        text = (response.text or "").strip()
        if not text:
            reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
            raise RuntimeError(f"Empty response from Gemini model (finish_reason={reason}).")
        return text

    return _gemini_try_models(meta.get("task"), meta, attempt)


def _stream_gemini(user: str, system: Optional[str], max_tokens: int, timeout: float,
                   history: List[Dict[str, str]], temperature: float, meta: Dict[str, Any]) -> Iterator[str]:
    client = _gemini_client(_env("GEMINI_API_KEY"), _env("GEMINI_BASE_URL"))
    contents = _gemini_contents(history, user)

    def first_chunk(model: str, thinking: bool):
        # Open the stream and pull the first text chunk inside the retry loop, so a model that
        # fails up front falls through to the next one; later chunks are yielded below.
        stream = iter(client.models.generate_content_stream(
            model=model, contents=contents, config=_gemini_config(system, max_tokens, timeout, temperature, thinking)))
        for chunk in stream:
            if chunk.text:
                return chunk.text, stream
        raise RuntimeError("Empty streamed response from Gemini model.")

    text, stream = _gemini_try_models(meta.get("task"), meta, first_chunk)
    yield text
    for chunk in stream:
        if chunk.text:
            yield chunk.text


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


def _friendly_reason(errors: List[str]) -> str:
    text = " ".join(errors).lower()
    if "429" in text or "resource_exhausted" in text or "quota" in text:
        return ("your Gemini API key hit its rate limit (the free tier allows about 5 requests per minute "
                "per model); it resets within a minute")
    if "503" in text or "high demand" in text or "overloaded" in text or "504" in text:
        return "Gemini is overloaded right now"
    if "401" in text or "403" in text or "api key not valid" in text or "permission" in text:
        return "the API key was rejected"
    if "404" in text:
        return "the configured models are not available to this API key"
    return errors[0][:140]


def _explain_offline(task: str, text: str, errors: List[str]) -> str:
    """When a configured provider failed, say why in the chat (agent tasks stay clean)."""
    if task != "chat" or not errors:
        return text
    reason = _friendly_reason(errors)
    if text == offline.OFFLINE_NOTICE:
        return (f"The live model is unavailable right now: {reason}. Open-ended questions need it, so please "
                "try again shortly. Meanwhile I can still answer from the platform's data, arithmetic and dates.")
    return f"{text}\n\n_Answered by the offline engine because {reason}._"


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
        meta: Dict[str, Any] = {"model": "", "task": task}
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
    text = _explain_offline(task, offline.synthesize(task, prompt, request["api_context"]), errors)
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
        meta: Dict[str, Any] = {"model": "", "task": task}
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
    text = _explain_offline(task, offline.synthesize(task, prompt, request["api_context"]), errors)
    first_token_ms = round((time.perf_counter() - started) * 1000.0, 1)
    for i in range(0, len(text), 48):
        yield {"type": "delta", "text": text[i:i + 48]}
    entry = _record("offline", "synthesizer", task, started, ok=True, streamed=True)
    yield {"type": "done", "provider": "offline", "model": "synthesizer", "latency_ms": entry["latency_ms"],
           "first_token_ms": first_token_ms, "errors": errors}
