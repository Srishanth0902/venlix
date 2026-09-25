"""Dashboard HTTP / WebSocket API."""
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from dashboard.server import app


@pytest.fixture
def api():
    with TestClient(app) as test_client:
        yield test_client


def _wait_for_run(api, run_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = api.get(f"/api/runs/{run_id}").json()
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_index_and_health(api):
    assert "Venlix Agent Console" in api.get("/").text
    assert api.get("/static/app.js").status_code == 200
    health = api.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["llm"]["active_provider"] == "offline"


def test_sample_run_populates_cases_and_stats(api):
    run = api.post("/api/runs", json={"source": "sample", "concurrency": 3}).json()
    assert run["total"] == 6 and run["skipped_low_risk"] == 1
    assert _wait_for_run(api, run["run_id"])["status"] == "completed"

    cases = api.get("/api/cases").json()["cases"]
    assert len(cases) == 6
    stats = api.get("/api/stats").json()
    assert stats["agent"]["totals"]["cases"] == 6
    assert stats["agent"]["by_failure_type"]["fraud"] == 1
    assert stats["llm"]["total_calls"] > 0

    case = api.get("/api/cases/DEL-TEST-2").json()
    assert case["trace"]
    summary = api.post("/api/cases/DEL-TEST-2/summary").json()["summary"]
    assert summary and api.get("/api/cases/DEL-TEST-2").json()["decision_summary"] == summary

    assert api.delete("/api/cases").json()["deleted"] == 6
    assert api.get("/api/cases").json()["cases"] == []


def test_runs_finish_within_the_request_on_vercel(api, monkeypatch):
    # Serverless functions have no WebSockets and may pause after responding.
    monkeypatch.setenv("VERCEL", "1")
    run = api.post("/api/runs", json={"source": "sample"}).json()
    assert run["status"] == "completed" and run["completed"] == 6
    assert len(api.get("/api/cases").json()["cases"]) == 6


def test_case_store_uses_tmp_on_vercel(monkeypatch):
    import tempfile
    from delivery_agent import store

    monkeypatch.delenv("VENLIX_DB_PATH", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    store.reset_singletons()
    assert store.get_store().path == os.path.join(tempfile.gettempdir(), "venlix_agent.db")


def test_backend_run_uses_sample_data_when_backend_is_down(api):
    run = api.post("/api/runs", json={"source": "backend"}).json()
    assert run["total"] == 3 and run["backend_live"] is False
    assert _wait_for_run(api, run["run_id"])["completed"] == 3


def test_custom_case(api):
    res = api.post("/api/cases", json={
        "customer_name": "Test User", "risk_score": 95,
        "risk_factors": [{"factor": "Low Address Confidence", "impact": 90}],
        "recommended_actions": ["Verify Address"],
    })
    assert res.status_code == 200
    case = res.json()
    assert case["failure_type"] == "address_issue"
    assert case["customer_message"].startswith("Hi Test User")
    assert api.get(f"/api/cases/{case['delivery_id']}").status_code == 200


def test_chat_uses_agent_cases_and_backend_data(api):
    api.post("/api/cases", json={"customer_name": "Zed", "risk_factors": [{"factor": "Suspicious Order Pattern", "impact": 95}]})
    res = api.post("/api/chat", json={"message": "Which cases were escalated?", "history": []}).json()
    assert "Agent case store" in res["apis_called"]
    assert "fraud" in res["answer"]

    res = api.post("/api/chat", json={"message": "What's today's failure rate?"}).json()
    assert "GET /reports" in res["apis_called"] and "Failure rate" in res["answer"]


def test_chat_stream_is_server_sent_events(api):
    with api.stream("POST", "/api/chat/stream", json={"message": "what is 9*9", "history": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}) as res:
        assert res.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in res.iter_lines() if line.startswith("data: ")]
    assert events[0]["type"] == "meta" and events[-1]["type"] == "done"
    assert "81" in "".join(e["text"] for e in events if e["type"] == "delta")


def test_exception_analysis(api):
    res = api.post("/api/exceptions/analyze", json={"note": "Package crushed and leaking"}).json()
    assert res["references"] and res["references"][0]["category"] == "Package Damage"
    assert 0 <= res["confidence"] <= 1


def test_validation_errors(api):
    assert api.post("/api/chat", json={"message": ""}).status_code == 422
    assert api.post("/api/runs", json={"concurrency": 99}).status_code == 422
    assert api.get("/api/cases/NOPE").status_code == 404


def test_websocket_streams_case_events(api):
    with api.websocket_connect("/ws") as ws:
        assert ws.receive_json()["event"] == "hello"
        api.post("/api/cases", json={"customer_name": "Live", "risk_factors": [{"factor": "Heavy Traffic", "impact": 80}]})
        events = [ws.receive_json()["event"] for _ in range(2)]
    assert events == ["case_started", "case_completed"]
