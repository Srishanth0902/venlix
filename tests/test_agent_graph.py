"""End-to-end agent graph behaviour (offline engine + stubbed LLM responses)."""
import asyncio
import time

import pytest

from delivery_agent.graph import get_delivery_graph
from delivery_agent.mapping import is_at_risk, map_to_delivery_case
from delivery_agent.runner import run_cases
from delivery_agent.sample_data import sample_deliveries
from delivery_agent.store import get_store
from delivery_agent.llm_manager import customer_comm


def _cases():
    rows = [d for d in sample_deliveries() if is_at_risk(d)]
    return [map_to_delivery_case(d, i, source="sample") for i, d in enumerate(rows)]


def test_sample_run_covers_every_resolution_path():
    results = {r["delivery_id"]: r for r in asyncio.run(run_cases(_cases()))}
    assert len(results) == 6

    fraud = results["DEL-DEMO-FRAUD"]
    assert (fraud["resolution_path"], fraud["status"]) == ("escalation", "escalated")
    assert fraud["savings"] == {"time_min": 0.0, "fuel_inr": 0.0, "cost_inr": 0.0}

    traffic = results["DEL-DEMO-TRAFFIC"]
    assert (traffic["resolution_path"], traffic["final_outcome"], traffic["status"]) == ("auto_route", "resolved_auto", "resolved")
    assert traffic["savings"]["cost_inr"] == 75.0
    assert not traffic.get("customer_message")  # deterministic path: no LLM

    gate = results["DEL-TEST-2"]
    assert gate["resolution_path"] == "customer_contact"
    assert "Arjun Mehta" in gate["customer_message"]
    assert gate["customer_intent"] is not None
    # Clear replies are parsed locally, so a case costs two LLM calls, not three
    assert {c["task"] for c in gate["llm_metadata"]["calls"]} == {"sms_draft", "customer_reply_sim"}

    for result in results.values():
        assert result["duration_ms"] is not None and result["completed_at"]
        assert result["trace"][-1]["node"] == "manager"

    stored = get_store().list()
    assert len(stored) == 6
    assert get_store().get("DEL-TEST-2")["customer_message"] == gate["customer_message"]


def _llm_node_case():
    row = next(d for d in sample_deliveries() if d["delivery_id"] == "DEL-TEST-3")
    return map_to_delivery_case(row, source="sample")


def test_declined_reply_is_escalated_not_resolved(monkeypatch):
    def fake_llm(prompt, system=None, max_tokens=200, timeout=5.0, history=None, use_tools=None, task=None, temperature=0.1):
        return {
            "sms_draft": "<sms>Hi Kavya, can we come at 5 PM?</sms>",
            "customer_reply_sim": "<reply>No, cancel it.</reply>",
            "reply_parse": '```json\n{"wants_reschedule": false, "new_slot": null, "declined": true}\n```',
        }[task]

    monkeypatch.setattr(customer_comm, "call_llm", fake_llm)
    result = asyncio.run(get_delivery_graph().ainvoke(_llm_node_case()))
    assert result["final_outcome"] == "customer_declined"
    assert result["status"] == "escalated"
    assert result["savings"]["cost_inr"] == 0.0


def test_rambling_llm_output_is_contained(monkeypatch):
    ramble = "Okay, let me think step by step.\n\nFirst the customer...\n\n" + "blah " * 80

    def fake_llm(prompt, system=None, max_tokens=200, timeout=5.0, history=None, use_tools=None, task=None, temperature=0.1):
        if task == "reply_parse":
            return "not json at all"
        return ramble

    monkeypatch.setattr(customer_comm, "call_llm", fake_llm)
    result = asyncio.run(get_delivery_graph().ainvoke(_llm_node_case()))
    assert result["customer_message"].startswith("Hi Kavya Reddy")  # template, not the ramble
    assert len(result["customer_message"].split()) <= customer_comm.SMS_WORD_LIMIT
    assert result["customer_reply"] == "Sure, that works for me."   # safe default reply
    assert result["final_outcome"] == "rescheduled"                   # keyword fallback parsed it


def test_llm_errors_escalate_instead_of_crashing(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(customer_comm, "call_llm", broken)
    result = asyncio.run(get_delivery_graph().ainvoke(_llm_node_case()))
    assert result["status"] in ("resolved", "escalated")
    assert result["customer_message"]  # template SMS still produced
    assert result["trace"][-1]["node"] == "manager"


def test_cases_run_concurrently(monkeypatch):
    def slow_llm(prompt, system=None, max_tokens=200, timeout=5.0, history=None, use_tools=None, task=None, temperature=0.1):
        time.sleep(0.2)
        return {"sms_draft": "<sms>Hi, 5 PM ok?</sms>", "customer_reply_sim": "<reply>Yes</reply>",
                "reply_parse": '{"wants_reschedule": true, "new_slot": null, "declined": false}'}[task]

    monkeypatch.setattr(customer_comm, "call_llm", slow_llm)
    cases = [c for c in _cases() if c["delivery_id"].startswith("DEL-TEST")]  # 4 LLM-path cases, 3 calls each
    started = time.perf_counter()
    results = asyncio.run(run_cases(cases, concurrency=4))
    elapsed = time.perf_counter() - started
    assert all(r["final_outcome"] == "rescheduled" for r in results)
    assert elapsed < 4 * 3 * 0.2 * 0.6, f"expected concurrent execution, took {elapsed:.2f}s"
