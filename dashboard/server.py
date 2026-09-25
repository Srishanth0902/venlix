"""
Venlix Agent Dashboard - FastAPI server.

Run:  python -m dashboard              -> http://127.0.0.1:8050

Endpoints
  GET  /api/health                  provider chain, backend status
  GET  /api/stats                   agent KPIs + LLM latency metrics
  GET  /api/llm/metrics             LLM metrics with recent calls
  GET  /api/cases                   processed cases (newest first)
  GET  /api/cases/{id}              one case with full trace
  POST /api/cases                   run a custom case through the agent
  POST /api/cases/{id}/summary      plain-English explanation of a decision
  DELETE /api/cases                 clear the case store
  POST /api/runs                    run sample / backend deliveries in the background
                                    (on Vercel the run finishes before the response)
  GET  /api/runs/{run_id}           run progress
  POST /api/chat                    ask the Operations Copilot anything
  POST /api/chat/stream             same, streamed as Server-Sent Events
  POST /api/exceptions/analyze      Tier-2 RAG analysis of a driver exception note
  WS   /ws                          live events (case_started, case_completed, run_*)
"""
import os
import time
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from delivery_agent.mapping import map_to_delivery_case, is_at_risk
from delivery_agent.runner import run_case, run_cases, DEFAULT_CONCURRENCY
from delivery_agent.sample_data import sample_deliveries
from delivery_agent.store import get_store, get_event_bus
from delivery_agent.llm_manager import copilot
from delivery_agent.llm_manager.client import call_llm_detailed, stream_llm, get_llm_metrics, get_provider_status
from delivery_agent.llm_manager.decision import summarize_decision
from delivery_agent.llm_manager.exception_analyzer import analyze_exception_note

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Venlix Agent Dashboard", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_runs: Dict[str, Dict[str, Any]] = {}
_background_tasks: Set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Copilot access to the agent's own cases
# ---------------------------------------------------------------------------

def _compact_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "delivery_id": case.get("delivery_id"),
        "customer": (case.get("customer") or {}).get("name"),
        "status": case.get("status"),
        "final_outcome": case.get("final_outcome"),
        "failure_type": case.get("failure_type"),
        "risk_score": case.get("risk_score"),
        "resolution_path": case.get("resolution_path"),
        "resolution_detail": case.get("resolution_detail"),
        "customer_message": case.get("customer_message"),
        "customer_reply": case.get("customer_reply"),
        "savings": case.get("savings"),
        "duration_ms": case.get("duration_ms"),
    }


def agent_cases_context() -> Dict[str, Any]:
    store = get_store()
    stats = store.stats()
    return {
        "totals": {**stats["totals"], "savings": stats["savings"]},
        "by_failure_type": stats["by_failure_type"],
        "by_outcome": stats["by_outcome"],
        "cases": [_compact_case(c) for c in store.list(limit=50)],
    }


copilot.register_context_provider(
    "agent_cases_api", "Agent case store",
    ["case", "escalat", "resolved", "saving", "agent", "sms", "reschedul", "outcome", "declin", "processed"],
    agent_cases_context,
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    source: Literal["sample", "backend"] = "sample"
    concurrency: int = Field(DEFAULT_CONCURRENCY, ge=1, le=16)


class RiskFactor(BaseModel):
    factor: str = Field(..., min_length=1, max_length=120)
    impact: float = Field(50, ge=0, le=100)


class CaseRequest(BaseModel):
    delivery_id: Optional[str] = Field(None, max_length=60)
    customer_name: str = Field("Customer", max_length=80)
    phone: Optional[str] = Field(None, max_length=30)
    address: Optional[str] = Field(None, max_length=200)
    risk_score: float = Field(90, ge=0, le=100)
    risk_factors: List[RiskFactor] = Field(default_factory=list, max_length=12)
    recommended_actions: List[str] = Field(default_factory=list, max_length=8)
    proposed_slot: Optional[str] = Field(None, max_length=60)
    driver_status: Optional[str] = Field(None, max_length=80)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=20000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=40)


class NoteRequest(BaseModel):
    note: str = Field(..., min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# Pages & status
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "llm": get_provider_status(),
        "backend": copilot.get_backend_status(),
        "db_path": get_store().path,
        "live_clients": get_event_bus().subscriber_count,
    }


@app.get("/api/stats")
async def stats() -> Dict[str, Any]:
    agent_stats = await asyncio.to_thread(get_store().stats)
    return {"agent": agent_stats, "llm": get_llm_metrics(recent=0)}


@app.get("/api/llm/metrics")
async def llm_metrics(recent: int = 50) -> Dict[str, Any]:
    return get_llm_metrics(recent=max(0, min(recent, 500)))


# ---------------------------------------------------------------------------
# Cases & runs
# ---------------------------------------------------------------------------

@app.get("/api/cases")
async def list_cases(limit: int = 200) -> Dict[str, Any]:
    cases = await asyncio.to_thread(get_store().list, max(1, min(limit, 1000)))
    return {"cases": cases}


@app.get("/api/cases/{delivery_id}")
async def get_case(delivery_id: str) -> Dict[str, Any]:
    case = await asyncio.to_thread(get_store().get, delivery_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {delivery_id} not found")
    return case


@app.delete("/api/cases")
async def clear_cases() -> Dict[str, Any]:
    deleted = await asyncio.to_thread(get_store().clear)
    await get_event_bus().publish({"event": "cases_cleared"})
    return {"deleted": deleted}


@app.post("/api/cases")
async def create_case(req: CaseRequest) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "delivery_id": req.delivery_id or f"DEL-CUSTOM-{uuid.uuid4().hex[:6].upper()}",
        "customer": {"name": req.customer_name, "phone": req.phone, "address": req.address},
        "driver": {"status": req.driver_status} if req.driver_status else None,
        "risk_score": req.risk_score,
        "risk_factors": [rf.model_dump() for rf in req.risk_factors],
        "recommended_actions": [{"action": a} for a in req.recommended_actions if a.strip()],
        "proposed_slot": req.proposed_slot,
    }
    case = map_to_delivery_case(payload, source="custom")
    return await run_case(case)


@app.post("/api/cases/{delivery_id}/summary")
async def explain_case(delivery_id: str) -> Dict[str, Any]:
    store = get_store()
    case = await asyncio.to_thread(store.get, delivery_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {delivery_id} not found")
    try:
        summary = await asyncio.to_thread(summarize_decision, case.get("trace") or [])
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"Could not summarise: {err}")
    case["decision_summary"] = summary
    await asyncio.to_thread(store.save, case)
    return {"delivery_id": delivery_id, "summary": summary}


@app.post("/api/runs")
async def start_run(req: RunRequest) -> Dict[str, Any]:
    bus = get_event_bus()
    raw = sample_deliveries() if req.source == "sample" else await asyncio.to_thread(copilot.get_deliveries)
    at_risk = [d for d in raw if is_at_risk(d)]
    cases = [map_to_delivery_case(d, i, source=req.source) for i, d in enumerate(at_risk)]

    run: Dict[str, Any] = {
        "run_id": uuid.uuid4().hex[:8],
        "source": req.source,
        "concurrency": req.concurrency,
        "total": len(cases),
        "skipped_low_risk": len(raw) - len(at_risk),
        "completed": 0,
        "status": "running",
        "started_at": time.time(),
    }
    if req.source == "backend":
        run["backend_live"] = bool(copilot.get_backend_status().get("live"))
    _runs[run["run_id"]] = run
    for stale in list(_runs)[:-50]:  # keep the 50 most recent runs
        _runs.pop(stale, None)

    async def on_result(_: Dict[str, Any]) -> None:
        run["completed"] += 1
        await bus.publish({"event": "run_progress", "run": dict(run)})

    async def execute() -> None:
        await bus.publish({"event": "run_started", "run": dict(run)})
        started = time.perf_counter()
        try:
            await run_cases(cases, concurrency=req.concurrency, on_result=on_result)
            run["status"] = "completed"
        except Exception as err:
            logger.exception("Run failed")
            run.update(status="failed", error=str(err))
        run["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        await bus.publish({"event": "run_completed", "run": dict(run)})

    task = asyncio.create_task(execute())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    if _sync_runs():
        await task
    return run


def _sync_runs() -> bool:
    """Serverless hosts (Vercel) may pause work after the response and have no WebSockets,
    so there the run completes inside the request. VENLIX_SYNC_RUNS=1 forces this anywhere."""
    return bool(os.getenv("VERCEL")) or os.getenv("VENLIX_SYNC_RUNS") == "1"


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return _runs[run_id]


# ---------------------------------------------------------------------------
# Copilot chat & exception analysis
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    history = [m.model_dump() for m in req.history]
    try:
        result = await asyncio.to_thread(
            call_llm_detailed, req.message, None, 2048, 45.0, history, True, "chat", 0.3)
    except Exception as err:
        raise HTTPException(status_code=503, detail=str(err))
    return {
        "answer": result["text"],
        "provider": result["provider"],
        "model": result["model"],
        "latency_ms": result["latency_ms"],
        "apis_called": result["apis_called"],
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    history = [m.model_dump() for m in req.history]

    def events():
        try:
            for event in stream_llm(req.message, None, 2048, 45.0, history, True, "chat", 0.3):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as err:  # never leave the browser hanging
            yield f"data: {json.dumps({'type': 'error', 'message': str(err)})}\n\n"

    # Starlette iterates a sync generator in a worker thread, so blocking SDK
    # streams never stall the event loop.
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/exceptions/analyze")
async def analyze_exception(req: NoteRequest) -> Dict[str, Any]:
    return await asyncio.to_thread(analyze_exception_note, req.note)


# ---------------------------------------------------------------------------
# Live updates
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_events(websocket: WebSocket) -> None:
    await websocket.accept()
    bus = get_event_bus()
    queue = bus.subscribe()

    async def sender() -> None:
        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event, default=str))

    async def receiver() -> None:
        while True:
            await websocket.receive_text()  # raises on disconnect

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        await websocket.send_text(json.dumps({"event": "hello", "ts": time.time()}))
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        bus.unsubscribe(queue)
