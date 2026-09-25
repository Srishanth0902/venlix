"""Risk classification, ingest mapping, reply parsing and the offline engine."""
import pytest

from delivery_agent.interfaces import classify_failure
from delivery_agent.mapping import is_at_risk, map_to_delivery_case, normalize_risk_score
from delivery_agent.sample_data import TEST_DATA, sample_deliveries
from delivery_agent.llm_manager.copilot import SEEDED_DELIVERIES
from delivery_agent.llm_manager.customer_comm import parse_customer_reply
from delivery_agent.llm_manager.offline import answer_question, parse_reply_keywords, _try_math


@pytest.mark.parametrize("delivery_id, expected", [
    ("DEL-TEST-2", "access_issue"),          # High Gate Wait Time (95) dominates
    ("DEL-TEST-3", "customer_unavailable"),  # Customer Response Time (93)
    ("DEL-TEST-4", "access_issue"),
    ("DEL-TEST-5", "customer_unavailable"),
    ("DEL-DEMO-FRAUD", "fraud"),
    ("DEL-DEMO-TRAFFIC", "driver_delay"),
])
def test_ml_payloads_are_classified_by_weighted_factors(delivery_id, expected):
    row = next(d for d in sample_deliveries() if d["delivery_id"] == delivery_id)
    assert classify_failure(map_to_delivery_case(row))["failure_type"] == expected


@pytest.mark.parametrize("case_id, expected", [
    ("HERO-001", "access_issue"),   # "Gated Access Code Missing"
    ("HERO-002", "address_issue"),  # "Vacant Suite / Wrong Address"
    ("DEL-104", "driver_delay"),    # "Traffic / Heavy Weather"
])
def test_backend_failure_type_is_used_as_a_signal(case_id, expected):
    row = next(d for d in SEEDED_DELIVERIES if d["case_id"] == case_id)
    assert classify_failure(map_to_delivery_case(row))["failure_type"] == expected


def test_positive_factors_are_ignored():
    low_risk = map_to_delivery_case(TEST_DATA[0])  # "High Address Confidence", "Customer Reachable", ...
    assert classify_failure(low_risk)["factors"] == []


def test_risk_scores_are_normalised_and_filtered():
    assert normalize_risk_score(99) == pytest.approx(0.99)
    assert normalize_risk_score(0.88) == pytest.approx(0.88)
    assert normalize_risk_score("bad") == 0.0
    assert [is_at_risk(d) for d in TEST_DATA] == [False, True, True, True, True]
    assert [is_at_risk(d) for d in SEEDED_DELIVERIES] == [True, True, False, True, False]


def test_mapping_handles_all_payload_shapes():
    ml = map_to_delivery_case(TEST_DATA[1], 1, source="sample")
    assert ml["risk_reason"][0] == "High Gate Wait Time"
    assert ml["ai_recommendation"]["recommended_actions"][0]["action"] == "Notify Security Gate"
    backend = map_to_delivery_case(SEEDED_DELIVERIES[0], 0, source="backend")
    assert backend["delivery_id"] == "HERO-001"
    assert backend["customer"]["name"] == "Alex Rivera"
    assert backend["driver"]["name"] == "Dave Miller"
    assert backend["risk_reason"] == ["Gated Access Code Missing"]


@pytest.mark.parametrize("reply, wants, declined, slot", [
    ("No problem, 5 PM works", True, False, "5:00 PM"),
    ("I know, sure", True, False, None),               # "know" is not "no"
    ("Right now is fine", True, False, None),          # "now" is not "no"
    ("No.", False, True, None),
    ("no, please cancel the order", False, True, None),
    ("I am not home today, come tomorrow at 2pm", True, False, "Tomorrow 2:00 PM"),
    ("Can you come tomorrow at 2 PM instead?", True, False, "Tomorrow 2:00 PM"),
    ("hmm", False, False, None),
])
def test_keyword_reply_parser(reply, wants, declined, slot):
    parsed = parse_reply_keywords(reply)
    assert parsed["wants_reschedule"] is wants
    assert parsed["declined"] is declined
    assert parsed["new_slot"] == slot


def test_parse_customer_reply_offline_returns_schema():
    assert parse_customer_reply("Yes, Friday 10am") == {"wants_reschedule": True, "new_slot": "Friday 10:00 AM", "declined": False}
    assert parse_customer_reply("") == {"wants_reschedule": False, "new_slot": None, "declined": False}


@pytest.mark.parametrize("question, answer", [
    ("what is 12*7+3?", "87"),
    ("What is 17% of 2400?", "408"),
    ("whats 5 times 3", "15"),
    ("sqrt(144)", "12"),
])
def test_offline_math(question, answer):
    assert f"**{answer}**" in _try_math(question)


@pytest.mark.parametrize("question", ["deliveries with risk 0.7-0.9", "status of DEL-104", "what happened on 2026-09-25"])
def test_offline_math_does_not_hijack_data_questions(question):
    assert _try_math(question) is None


def test_offline_answers_are_grounded_and_honest():
    ctx = {"deliveries_api": SEEDED_DELIVERIES}
    assert "HERO-002" in answer_question("Show high-risk deliveries", ctx)
    assert "Robert Chen" in answer_question("What happened with DEL-104?", ctx)
    assert "Dave Miller" in answer_question("Which deliveries did Dave handle?", ctx)
    # "this" contains "hi" but is not a greeting; open questions get an honest notice
    general = answer_question("Explain this: why is the sky blue?", {})
    assert "offline mode" in general and "Hello" not in general
