# Venlix delivery exception agent

A LangGraph agent that takes at-risk deliveries flagged by the Venlix XGBoost model,
works out what is going wrong, and resolves it: it reassigns the driver, contacts the
customer by SMS, or escalates to a human. It also includes an **Operations Copilot**
that answers questions about live operations (and general questions), and a **web
dashboard** for watching and driving all of it.

```
ML prediction ─► risk_detection ─┬─► deterministic_resolution (driver_delay: reassign / re-route, no LLM)
 (risk_factors)   weighs factors  ├─► llm_resolution (access / address / customer issues:
                  by ML impact    │     LLM drafts SMS → customer reply → LLM parses intent)
                                  ├─► escalation (fraud → human review queue)
                                  └─► manager (errors) ──► manager: savings, SQLite, live WebSocket push
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # Windows  (macOS/Linux: cp .env.example .env)
# put your GEMINI_API_KEY (and/or OPENROUTER_API_KEY) in .env

python -m dashboard               # open http://127.0.0.1:8050
```

Command-line demos still work: `python test_graph.py` runs the ML sample predictions, and
`python pipeline.py` runs the backend's `/deliveries/` feed through the agent.

> **Without an API key** everything still runs on the built-in *offline engine*. It
> answers from platform data, arithmetic and dates, and drafts template SMS messages.
> It says plainly when a question needs a real model. To get answers to **any**
> question, set `GEMINI_API_KEY` or `OPENROUTER_API_KEY`.

## Dashboard

| Tab | What it does |
|---|---|
| **Overview** | Run the sample ML predictions or the backend deliveries through the agent (concurrently), live activity feed, KPIs (cases, resolution rate, escalations, ₹ and minutes saved, latency), and charts by failure type and outcome |
| **Cases** | Every processed case: status, failure type, path, outcome. Click one for the full trace with per-step timings, the SMS conversation, parsed intent, ML risk factors, the LLM calls made (provider, model, latency), the exact prompt, and an *Explain this decision* summary. **New case** runs a custom scenario |
| **Copilot** | Chat with streaming answers and conversation memory. It automatically pulls reports, deliveries, predictions, the digital twin and the agent's own cases when a question needs them |
| **Exception analyzer** | Tier-2 RAG: paste a messy driver note to get the root cause, a fix and similar historical cases |
| **System** | Provider chain and configuration, LLM call count, errors, latency per provider, recent calls |

Live updates arrive over a WebSocket (`/ws`); the REST API is listed in `dashboard/server.py`.

## What was broken and what changed

| Problem | Effect | Fix |
|---|---|---|
| Used the deprecated `google-generativeai` SDK and the retired `gemini-1.5-flash` model | Every Gemini call failed and silently fell back to canned mock text, so the agent "didn't answer" | Moved to the `google-genai` SDK. Defaults are Google's `gemini-flash-latest` / `gemini-flash-lite-latest` aliases, which never retire (even `gemini-2.5-flash` is now closed to new keys), and a model that returns 404 is skipped automatically |
| Agent calls used 5–8 s timeouts | The Gemini API rejects deadlines under 10 s, so every agent LLM call failed | Gemini requests use a deadline of at least 10 s; the per-step `AGENT_LLM_TIMEOUT` still bounds each step |
| No handling of rate limits or overload | Free-tier keys allow about 5 requests per minute per model, and Gemini often returns 503 "high demand"; either error meant no answer | Each request type tries a list of models (quotas are per model). A 429 puts that model on cooldown for the time Google specifies, a 5xx moves to the next model and gets one more round, and a 401 stops immediately |
| Gemini 2.5 "thinking" enabled with small token caps | Thinking tokens used up `max_output_tokens`: empty or truncated answers, plus several seconds of extra latency | Thinking budget 0 by default (`GEMINI_THINKING_BUDGET`); automatic retry for models that can't disable thinking |
| Offline mock echoed the prompt back | The simulated customer reply was literally `"and"` (a regex matched the `<reply> and </reply>` example inside the echo), and every SMS was the canned fallback | New offline engine produces real, context-aware drafts and replies, and honest data-grounded chat answers |
| Every agent prompt went through the Copilot tool router | Drafting an SMS fetched `/deliveries` and injected JSON into the prompt, replacing the SMS system prompt | Tools only run for Copilot questions (no system prompt), or explicitly with `use_tools=True` |
| Risk classifier read a `severity` key that the ML payload never sends | Every case was classified `driver_delay` | Factors are weighted by ML `impact`/`severity`, positive factors are ignored, backend `failure_type` text counts as a signal, and the heaviest category wins |
| `deterministic_resolution` node unreachable | Driver problems always took the slow LLM path | `driver_delay` routes to reassignment, with no LLM call |
| Outcome always `resolved_customer` | Customers who declined were counted as resolved, with savings | Declined or unclear replies escalate; savings only count for resolved cases |
| Reply parser matched substrings (`"no"` in *know*, *now*, *No problem*) | Wrong intents | Word-boundary parser with negation handling and slot extraction ("not today, tomorrow 2pm" gives Tomorrow 2:00 PM) |
| Blocking SDK calls inside async nodes; `asyncio.wait_for` could not time them out | One slow call froze every case | LLM calls run in worker threads; per-step timeouts actually fire |
| `write_agent_log` / `broadcast_ws` were `sleep()` stubs | Nothing persisted or reached a UI | SQLite case store and WebSocket event bus |
| ChromaDB initialised at import; debug `print`s | Slow startup | Lazy initialisation, logging, better keyword fallback |
| Duplicate mappers in `pipeline.py` / `test_graph.py` (`reason` vs `factor`, 0–100 vs 0–1 scores) | Lost risk reasons, inconsistent filtering | One shared `delivery_agent/mapping.py`, scores normalised to 0–1 |
| Literal `{{ }}` in two system prompts | The model saw malformed JSON examples | Fixed |
| No chat memory | Follow-up questions lost context | History is sent to both providers (last `LLM_MAX_HISTORY` turns) |

### Performance and responsiveness

- **Concurrent cases.** `run_cases()` processes a batch in parallel with a semaphore
  (`AGENT_CONCURRENCY`). In the test suite, 4 cases × 3 LLM calls × 200 ms took about
  0.6 s instead of 2.4 s.
- **Streaming chat.** The first words appear as soon as the model produces them, and
  the dashboard shows time-to-first-token.
- **No thinking tokens** by default, with enough output-token headroom that a model which
  still "thinks" can't return an empty answer. Timeouts are enforced on every provider,
  and SDK clients are cached, not rebuilt per call.
- **Right-sized models.** Short agent tasks (SMS drafts, summaries) run on Flash-Lite,
  about 0.5 s per call in live tests; the Copilot uses Flash.
- **Fewer LLM calls.** Clear customer replies ("Yes, 5 PM works") are parsed locally;
  only ambiguous ones go to the LLM. That is 2 calls per case instead of 3.
- **Backend calls** are cached (`BACKEND_CACHE_TTL`), and a down backend is skipped
  for 30 s rather than timing out on every question.
- **Provider chain.** Gemini → OpenRouter (several models, tried in order) → offline.
  An outage degrades an answer instead of failing it.
- **The compiled graph is reused.** ChromaDB loads only on first use.

## Deploying to Vercel

Import the repository in Vercel. It detects FastAPI through `main.py`, so no build settings are needed. Optionally add
`GEMINI_API_KEY` and/or `OPENROUTER_API_KEY` as environment variables; without them the offline engine answers.

Serverless functions differ from a normal server, and the app adapts when Vercel's `VERCEL` variable is set:

- **No WebSockets:** the dashboard polls for results instead, and the "Live" indicator stays red.
- **Runs finish inside the request**, because a function can be paused after it responds.
- **Cases live in SQLite under `/tmp`**, per function instance, so they reset when Vercel recycles the instance.

For the full experience (live feed, persistent cases), run it on a host with a long-running process, such as Render or
Railway: `uvicorn dashboard.server:app --host 0.0.0.0 --port $PORT`.

## Configuration

See [`.env.example`](.env.example). The main settings:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Primary provider. Keep it in `.env` or the environment, never in a committed file: Google disables keys it finds in public repos |
| `GEMINI_MODEL` / `GEMINI_TASK_MODEL` | `gemini-flash-latest` / `gemini-flash-lite-latest` | Comma-separated models for the Copilot and for agent tasks. List several on a free-tier key to multiply its per-model quota (see `.env.example`) |
| `GEMINI_THINKING_BUDGET` | `0` | `auto` lets the model think (slower) |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` / `OPENROUTER_BASE_URL` | – / Llama 3.3 70B free / OpenRouter | Fallback; any OpenAI-compatible API works (Ollama, vLLM, …) |
| `ALLOW_MOCK_FALLBACK` | `true` | Use the offline engine when no provider answers |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Venlix backend (`/reports/`, `/deliveries/`, `/twin/`, `/prediction/`) |
| `AGENT_CONCURRENCY` / `AGENT_LLM_TIMEOUT` | `4` / `20` s | Batch parallelism and per-step timeout |
| `VENLIX_DB_PATH` | `venlix_agent.db` | SQLite case store |
| `DASHBOARD_HOST` / `DASHBOARD_PORT` | `127.0.0.1` / `8050` | Dashboard address (8000 is the backend's) |

## Project layout

```
delivery_agent/
  graph.py, nodes.py, state.py   LangGraph workflow
  interfaces.py                  risk classifier, persistence, LLM adapters
  mapping.py, sample_data.py     payload ingest and demo data
  runner.py, store.py            concurrent runs, SQLite store and event bus
  llm_manager/
    client.py                    provider chain, streaming, telemetry
    copilot.py                   backend tools, intent routing, caching
    offline.py                   offline engine
    customer_comm.py, decision.py, exception_analyzer.py, prompts.py
dashboard/                       FastAPI server and static web UI
tests/                           pytest suite (fake Gemini/OpenAI servers, no keys needed)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs fully offline. A local fake server speaks the Gemini and OpenAI wire
protocols, so the real SDK code paths are exercised: request shape, thinking retry,
fallback order, multi-model fallback, and streaming.
