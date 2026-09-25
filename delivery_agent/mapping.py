"""
Ingest mapping: turn any upstream payload into the agent's DeliveryCase state.

Accepts the three shapes seen in this project:
- ML prediction output  (risk_factors[{factor, impact}], recommended_actions, risk_score 0..100)
- Backend /deliveries/  (case_id, customer_name, failure_type, driver_name, risk_score 0..1)
- Enriched payloads     (delivery_id, customer{...}, driver{...}, store, drop, ai_recommendation)
"""
from typing import Any, Dict, List

from .state import DeliveryCase

AT_RISK_PREDICTIONS = {"delivery failure", "delivery failure likely", "critical risk", "high risk"}
AT_RISK_THRESHOLD = 0.8


def normalize_risk_score(value: Any) -> float:
    """ML service sends 0..100, the backend sends 0..1; the agent works in 0..1."""
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score /= 100.0
    return max(0.0, min(1.0, score))


def is_at_risk(data: Dict[str, Any], threshold: float = AT_RISK_THRESHOLD) -> bool:
    prediction = str(data.get("prediction", "")).strip().lower()
    return (
        prediction in AT_RISK_PREDICTIONS
        or data.get("delivery_failed") == 1
        or normalize_risk_score(data.get("risk_score")) > threshold
    )


def _clean_factors(raw: Any) -> List[Dict[str, Any]]:
    factors = []
    for rf in raw or []:
        if isinstance(rf, str):
            rf = {"factor": rf}
        if not isinstance(rf, dict):
            continue
        name = rf.get("factor") or rf.get("reason") or rf.get("name") or ""
        if not name:
            continue
        factors.append({**rf, "factor": name})
    return factors


def map_to_delivery_case(data: Dict[str, Any], index: int = 0, source: str = "custom") -> DeliveryCase:
    ident = data.get("id")
    delivery_id = (
        data.get("delivery_id")
        or data.get("case_id")
        or (f"DEL-{ident}" if ident is not None else f"DEL-{source.upper()}-{index + 1}")
    )

    customer = dict(data.get("customer") or {})
    if data.get("customer_name"):
        customer.setdefault("name", data["customer_name"])
    if data.get("address"):
        customer.setdefault("address", data["address"])
    if data.get("phone"):
        customer.setdefault("phone", data["phone"])

    driver = dict(data.get("driver") or {})
    if data.get("driver_name"):
        driver.setdefault("name", data["driver_name"])
    if data.get("driver_rating") is not None:
        driver.setdefault("rating", data["driver_rating"])

    risk_factors = _clean_factors(data.get("risk_factors"))
    risk_reason = [rf.get("reason") or rf["factor"] for rf in risk_factors]
    backend_failure = str(data.get("failure_type") or "").strip()
    if backend_failure and backend_failure.lower() != "none":
        risk_reason.insert(0, backend_failure)

    ai_recommendation = dict(data.get("ai_recommendation") or {})
    if data.get("recommended_actions"):
        ai_recommendation.setdefault("recommended_actions", data["recommended_actions"])
    for key in ("estimated_success_after_action", "estimated_time_saved_minutes",
                "estimated_cost_saved_rupees", "estimated_fuel_saved_liters", "proposed_slot"):
        if data.get(key) is not None:
            ai_recommendation.setdefault(key, data[key])

    suffix = ident if ident is not None else index + 1
    return {
        "delivery_id": str(delivery_id),
        "customer_id": customer.get("customer_id") or f"CUST-{suffix:0>3}",
        "driver_id": driver.get("driver_id") or f"DRV-{suffix:0>3}",
        "risk_score": normalize_risk_score(data.get("risk_score")),
        "risk_reason": risk_reason,
        "failure_type": None,
        "failure_confidence": None,
        "store_location": data.get("store") or data.get("store_location"),
        "drop_location": data.get("drop") or data.get("drop_location"),
        "driver": driver or None,
        "customer": customer or None,
        "environment": data.get("environment"),
        "risk_factors": risk_factors,
        "ai_recommendation": ai_recommendation or None,
        "customer_context": {},
        "driver_context": {},
        "resolution_path": "",
        "resolution_detail": None,
        "problem_prompt": None,
        "customer_message": None,
        "customer_reply": None,
        "customer_intent": None,
        "final_outcome": None,
        "savings": {},
        "status": "pending",
        "source": source,
        "started_at": None,
        "completed_at": None,
        "duration_ms": None,
        "llm_metadata": None,
        "trace": []
    }
