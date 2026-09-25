"""
Operations Copilot Tool Integration for LLM Manager.

Connects the LLM Assistant to backend logistics APIs:
- GET  /reports/
- GET  /deliveries/
- GET  /twin/
- POST /prediction/

Responses are cached for a few seconds and a down backend is remembered briefly
(circuit breaker), so a chat question never waits on repeated connection timeouts.
When the backend is unreachable, seeded sample data is returned and flagged via
get_backend_status() so answers can say the data is not live.
"""

import re
import copy
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Iterable
import requests
from ..env_utils import env_float, env_str

logger = logging.getLogger(__name__)

BACKEND_BASE_URL = env_str("BACKEND_URL", "http://127.0.0.1:8000")
BACKEND_TIMEOUT = env_float("BACKEND_TIMEOUT", 3.0)
BACKEND_CACHE_TTL = env_float("BACKEND_CACHE_TTL", 15.0)
BACKEND_RETRY_AFTER = env_float("BACKEND_RETRY_AFTER", 30.0)

_session = requests.Session()
_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
_backend_state: Dict[str, Any] = {"live": None, "last_error": None, "down_until": 0.0, "checked_at": None}


def _fetch_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Call the backend with caching and a circuit breaker. Returns None when unavailable."""
    cache_key = f"{method} {path}"
    now = time.monotonic()
    if payload is None:
        with _cache_lock:
            hit = _cache.get(cache_key)
            if hit and now - hit[0] < BACKEND_CACHE_TTL:
                return copy.deepcopy(hit[1])

    if now < _backend_state["down_until"]:
        return None

    url = f"{BACKEND_BASE_URL.rstrip('/')}{path}"
    try:
        res = _session.request(method, url, json=payload, timeout=BACKEND_TIMEOUT)
        res.raise_for_status()
        data = res.json()
    except Exception as err:
        _backend_state.update(live=False, last_error=str(err), down_until=now + BACKEND_RETRY_AFTER, checked_at=time.time())
        logger.info(f"Backend {method} {path} unavailable ({err}); using sample data for {BACKEND_RETRY_AFTER:.0f}s.")
        return None

    _backend_state.update(live=True, last_error=None, down_until=0.0, checked_at=time.time())
    if payload is None:
        with _cache_lock:
            _cache[cache_key] = (now, copy.deepcopy(data))
    return data


def get_backend_status() -> Dict[str, Any]:
    """Whether the last backend call succeeded (None = not contacted yet)."""
    return {"url": BACKEND_BASE_URL, **{k: v for k, v in _backend_state.items() if k != "down_until"}}


def clear_backend_cache() -> None:
    with _cache_lock:
        _cache.clear()
    _backend_state.update(down_until=0.0)


# ---------------------------------------------------------
# Tool Callers (Fetching live backend data)
# ---------------------------------------------------------
def get_reports() -> Dict[str, Any]:
    """Fetch dashboard statistics from GET /reports/."""
    data = _fetch_json("GET", "/reports/")
    if isinstance(data, dict):
        return data

    # Seeded fallback matching backend schema, derived from the seeded deliveries
    deliveries = get_deliveries()
    failures = sum(1 for d in deliveries if d.get("delivery_failed") == 1)
    total = len(deliveries)
    return {
        "total_predictions": total,
        "delivery_failures": failures,
        "delivery_success": total - failures,
        "failure_rate": f"{(failures / total * 100) if total else 0:.1f}%",
        "average_confidence": round(sum(d.get("confidence", 0) for d in deliveries) / total, 2) if total else 0,
        "high_risk_count": sum(1 for d in deliveries if d.get("risk_score", 0) >= 0.7),
    }


def get_deliveries() -> List[Dict[str, Any]]:
    """Fetch delivery history from GET /deliveries/."""
    data = _fetch_json("GET", "/deliveries/")
    if isinstance(data, list) and data:
        return data
    return copy.deepcopy(SEEDED_DELIVERIES)


SEEDED_DELIVERIES: List[Dict[str, Any]] = [
    {
        "id": 101,
        "case_id": "HERO-001",
        "customer_name": "Alex Rivera",
        "address": "742 Evergreen Terrace, Gate 4",
        "failure_type": "Gated Access Code Missing",
        "delivery_failed": 1,
        "status": "Failed - Reschedule Required",
        "risk_score": 0.88,
        "confidence": 0.94,
        "driver_name": "Dave Miller",
        "driver_rating": 4.8,
        "created_at": "Today 09:30 AM"
    },
    {
        "id": 102,
        "case_id": "HERO-002",
        "customer_name": "Marcus Vance",
        "address": "100 Innovation Way, Suite 402",
        "failure_type": "Vacant Suite / Wrong Address",
        "delivery_failed": 1,
        "status": "Failed - Cancel Requested",
        "risk_score": 0.91,
        "confidence": 0.89,
        "driver_name": "Sarah Connor",
        "driver_rating": 4.6,
        "created_at": "Today 10:15 AM"
    },
    {
        "id": 103,
        "case_id": "DEL-103",
        "customer_name": "Elena Rostova",
        "address": "450 Ocean Drive",
        "failure_type": "None",
        "delivery_failed": 0,
        "status": "Delivered",
        "risk_score": 0.12,
        "confidence": 0.98,
        "driver_name": "Dave Miller",
        "driver_rating": 4.8,
        "created_at": "Today 08:45 AM"
    },
    {
        "id": 104,
        "case_id": "DEL-104",
        "customer_name": "Robert Chen",
        "address": "88 Industrial Pkwy",
        "failure_type": "Traffic / Heavy Weather",
        "delivery_failed": 1,
        "status": "Failed - Road Blockage",
        "risk_score": 0.82,
        "confidence": 0.85,
        "driver_name": "Carlos Gomez",
        "driver_rating": 4.5,
        "created_at": "Today 11:00 AM"
    },
    {
        "id": 105,
        "case_id": "DEL-105",
        "customer_name": "Sophia Martinez",
        "address": "120 Pine Street",
        "failure_type": "None",
        "delivery_failed": 0,
        "status": "Delivered",
        "risk_score": 0.05,
        "confidence": 0.99,
        "driver_name": "Sarah Connor",
        "driver_rating": 4.6,
        "created_at": "Today 07:30 AM"
    }
]


def get_twin() -> Dict[str, Any]:
    """Fetch Digital Twin map visualization data from GET /twin/."""
    data = _fetch_json("GET", "/twin/")
    if isinstance(data, dict):
        return data

    nodes = []
    for item in get_deliveries():
        item_id = item.get("id", 0)
        nodes.append({
            "id": item_id,
            "label": item.get("customer_name", "Recipient"),
            "lat": 12.9716 + (item_id * 0.01),
            "lng": 77.5946 + (item_id * 0.01),
            "risk_score": item.get("risk_score", 0.5),
            "status": item.get("status", "Active")
        })
    return {
        "active_nodes": len(nodes),
        "digital_twin_nodes": nodes,
        "high_risk_alert_count": sum(1 for n in nodes if n["risk_score"] > 0.7)
    }


DEFAULT_PREDICTION_PAYLOAD: Dict[str, Any] = {
    "Agent_Age": 28,
    "Agent_Rating": 4.8,
    "Store_Latitude": 12.9716,
    "Store_Longitude": 77.5946,
    "Drop_Latitude": 12.9352,
    "Drop_Longitude": 77.6245,
    "Weather": 1,
    "Traffic": 2,
    "Vehicle": 1,
    "Area": 1,
    "Category": 0,
    "Delivery_Time": 30,
    "pin_code": 560001,
    "driver_on_time_rate": 0.92,
    "customer_unavailability_history": 0.12,
    "address_failure_history_rate": 0.04,
    "order_value": 750,
    "slot_width_minutes": 30,
    "distance_km": 4.6,
    "risk_score": 0.35,
    "day_of_week": 3,
    "month": 7,
    "is_weekend": 0,
    "pickup_delay_minutes": 3,
    "hour_of_day": 15
}


def post_prediction(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute delivery failure prediction via POST /prediction/."""
    input_data = payload or DEFAULT_PREDICTION_PAYLOAD
    data = _fetch_json("POST", "/prediction/", payload=input_data)
    if isinstance(data, dict):
        return data

    # Fallback response format matching XGBoost backend model output
    risk = input_data.get("risk_score", 0.35)
    is_failed = 1 if risk > 0.6 else 0
    return {
        "id": 1,
        "delivery_failed": is_failed,
        "prediction": "Delivery Failure Likely" if is_failed else "Delivery Successful",
        "confidence": 0.94,
        "input_summary": {
            "distance_km": input_data.get("distance_km", 4.6),
            "traffic": input_data.get("Traffic", 2),
            "delivery_time_min": input_data.get("Delivery_Time", 30),
            "risk_score": risk
        }
    }


# ---------------------------------------------------------
# Intent Router & Copilot Dispatcher
# ---------------------------------------------------------
_CASE_ID_RE = re.compile(r"\b(?:HERO|DEL|CASE)-[A-Z0-9-]+\b", re.IGNORECASE)

# (context key, API label, keyword regex, fetcher). Keywords match on word starts,
# so "risk" matches "risky" but "map" does not match "bitmap".
_TOOLS: List[Dict[str, Any]] = [
    {"key": "reports_api", "label": "GET /reports",
     "pattern": r"\b(report|statistic|stats|failure rate|success rate|dashboard|summary|overview|kpi|how many|total|count)",
     "fetch": get_reports},
    {"key": "deliveries_api", "label": "GET /deliveries",
     "pattern": r"\b(fail|deliver|history|risk|high-risk|today|driver|customer|address|order|shipment|parcel|package|status|late|delay)",
     "fetch": get_deliveries},
    {"key": "twin_api", "label": "GET /twin",
     "pattern": r"\b(twin|map|coordinate|location|lat|lng|longitude|latitude|visuali[sz]ation|geo)",
     "fetch": get_twin},
    {"key": "prediction_api", "label": "POST /prediction",
     "pattern": r"\b(predict|xgboost|model|forecast|eta|how long|duration)",
     "fetch": post_prediction},
]

# Extra context providers registered by the host application (e.g. the dashboard
# registers the agent's processed cases).
_EXTRA_PROVIDERS: List[Dict[str, Any]] = []


def register_context_provider(key: str, label: str, keywords: Iterable[str], fetch: Callable[[], Any]) -> None:
    """Register an additional data source the copilot can attach to questions."""
    pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + ")"
    _EXTRA_PROVIDERS[:] = [p for p in _EXTRA_PROVIDERS if p["key"] != key]
    _EXTRA_PROVIDERS.append({"key": key, "label": label, "pattern": pattern, "fetch": fetch})


def process_copilot_request(prompt: str) -> Dict[str, Any]:
    """
    Decides which backend API(s) to invoke based on user prompt intent,
    retrieves live data, and returns structured context payload.
    """
    prompt_lower = prompt.lower()
    retrieved_context: Dict[str, Any] = {}
    apis_called: List[str] = []

    mentions_case_id = bool(_CASE_ID_RE.search(prompt))
    for tool in _TOOLS + _EXTRA_PROVIDERS:
        wanted = re.search(tool["pattern"], prompt_lower) is not None
        if mentions_case_id and tool["key"] in ("deliveries_api", "agent_cases_api"):
            wanted = True
        if not wanted:
            continue
        try:
            retrieved_context[tool["key"]] = tool["fetch"]()
            apis_called.append(tool["label"])
        except Exception as err:
            logger.warning(f"Copilot tool {tool['label']} failed: {err}")

    backend_keys = {"reports_api", "deliveries_api", "twin_api", "prediction_api"}
    if backend_keys & retrieved_context.keys() and _backend_state.get("live") is False:
        retrieved_context["data_source"] = "sample data (backend unreachable at " + BACKEND_BASE_URL + ")"

    # If no logistics keywords matched, apis_called remains [] and retrieved_context remains {}
    return {
        "apis_called": apis_called,
        "context_data": retrieved_context
    }
