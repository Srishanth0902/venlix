import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from .state import DeliveryCase
from .store import get_store, get_event_bus

# LLM Manager (all calls are synchronous SDK calls, so they run in worker threads
# to keep the event loop - and every other case in flight - responsive).
from .llm_manager import customer_comm
from .llm_manager.client import call_llm
from .llm_manager.exception_analyzer import analyze_exception_note as _analyze_exception_note

DEFAULT_PROPOSED_SLOT = "today at 5 PM"


async def get_delivery_context(delivery_id: str, case: Optional[DeliveryCase] = None) -> Dict[str, Any]:
    """Stub for Backend DB read to fetch context (uses what the payload already knows)."""
    case = case or {}
    customer = case.get("customer") or {}
    driver = case.get("driver") or {}
    return {
        "customer_context": {
            "unavailability_rate": customer.get("unavailability_rate", 0.15),
            "past_notes": customer.get("notes", ["gate code 1234"]),
        },
        "driver_context": {
            "on_time_rate": driver.get("on_time_rate", 0.88),
            "lat": driver.get("lat", 12.9716),
            "lng": driver.get("lng", 77.5946),
            "idle_driver_nearby": driver.get("idle_driver_nearby", True),
        },
    }


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

# Checked in order; the first category whose keywords match a factor claims it.
FAILURE_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("fraud", ("fraud", "suspicious", "anomal", "spoof", "chargeback")),
    ("access_issue", ("gate", "visitor", "security", "access", "entry", "intercom", "pass", "keypad")),
    ("address_issue", ("address", "location", "wrong", "vacant", "pin code", "geocod", "landmark")),
    ("customer_unavailable", ("customer", "availab", "reachab", "response", "contact", "recipient", "phone")),
    ("driver_delay", ("traffic", "weather", "driver", "road", "delay", "vehicle", "rain", "congestion", "pickup")),
]

# Factors the ML model reports as *reducing* risk (seen in low-risk predictions).
_POSITIVE_FACTOR = re.compile(r"\b(approved|excellent|quick|reachable|good|verified|high address confidence)\b", re.I)

_SEVERITY_WEIGHT = {"critical": 100.0, "high": 80.0, "medium": 50.0, "low": 20.0}


def _factor_weight(rf: Dict[str, Any], position: int) -> float:
    severity = str(rf.get("severity", "")).lower()
    if severity in _SEVERITY_WEIGHT:
        return _SEVERITY_WEIGHT[severity]
    try:
        return float(rf["impact"])
    except (KeyError, TypeError, ValueError):
        return max(10.0, 60.0 - 10.0 * position)  # unscored factors: earlier = more important


def _categorize(text: str) -> Optional[str]:
    lowered = text.lower()
    for failure_type, keywords in FAILURE_KEYWORDS:
        if any(re.search(rf"\b{re.escape(k)}", lowered) for k in keywords):
            return failure_type
    return None


def classify_failure(state: DeliveryCase) -> Dict[str, Any]:
    """
    Weigh every risk factor by its ML impact/severity, bucket it into a failure type
    and pick the heaviest bucket. Returns failure_type, confidence and the ranked factors.
    """
    weights: Dict[str, float] = {}
    ranked: List[Dict[str, Any]] = []

    factors = list(state.get("risk_factors") or [])
    known = {str(rf.get("factor", "")).lower() for rf in factors}
    # Free-text reasons (e.g. backend failure_type) that aren't already factors
    for reason in state.get("risk_reason") or []:
        if reason and reason.lower() not in known:
            factors.append({"factor": reason, "impact": 60.0})

    for position, rf in enumerate(factors):
        name = str(rf.get("factor") or rf.get("reason") or "")
        if not name or _POSITIVE_FACTOR.search(name):
            continue
        category = _categorize(name)
        weight = _factor_weight(rf, position)
        ranked.append({"factor": name, "weight": weight, "category": category})
        if category:
            weights[category] = weights.get(category, 0.0) + weight

    ranked.sort(key=lambda r: r["weight"], reverse=True)
    if not weights:
        return {"failure_type": "driver_delay", "confidence": 0.3, "factors": ranked}

    failure_type = max(weights, key=weights.get)
    confidence = round(weights[failure_type] / sum(weights.values()), 2)
    return {"failure_type": failure_type, "confidence": confidence, "factors": ranked}


async def analyze_risk_heuristics(state: DeliveryCase) -> str:
    """Reasoning step over the ML payload's risk_factors; returns the inferred failure_type."""
    return classify_failure(state)["failure_type"]


# ---------------------------------------------------------------------------
# Persistence & live updates
# ---------------------------------------------------------------------------

async def write_agent_log(case: DeliveryCase) -> None:
    """Persist the completed case (SQLite)."""
    await asyncio.to_thread(get_store().save, dict(case))


async def broadcast_ws(event: Dict[str, Any]) -> None:
    """Push an event to every connected dashboard (WebSocket)."""
    await get_event_bus().publish(event)

# ---------------------------------------------------------------------------
# LLM Manager Integrated Methods
# ---------------------------------------------------------------------------

async def ask_llm_manager(prompt: str, system_prompt: str = None) -> str:
    """Generic wrapper for an agent node to ask the LLM Manager directly."""
    try:
        return await asyncio.to_thread(call_llm, prompt=prompt, system=system_prompt, max_tokens=600, timeout=20.0)
    except Exception as e:
        return f"Error contacting LLM: {str(e)}"


def build_customer_case(case: DeliveryCase) -> Dict[str, Any]:
    """Convert DeliveryCase state into the dict the LLM manager's customer_comm understands."""
    customer = case.get("customer") or {}
    driver = case.get("driver") or {}
    ai_rec = case.get("ai_recommendation") or {}
    classification = classify_failure(case)
    top_reasons = [f["factor"] for f in classification["factors"][:3]]
    actions = [a.get("action") for a in ai_rec.get("recommended_actions") or [] if isinstance(a, dict) and a.get("action")]
    return {
        "case_id": case.get("delivery_id", "CASE-000"),
        "customer_name": customer.get("name") or "Customer",
        "failure_type": (case.get("failure_type") or "delivery issue").replace("_", " "),
        "driver_context": f"Driver is {driver.get('status', 'delayed')}.",
        "proposed_slot": ai_rec.get("proposed_slot") or DEFAULT_PROPOSED_SLOT,
        "phone": customer.get("phone"),
        "address": customer.get("address"),
        "metadata": {
            "risk_details": ", ".join(top_reasons) or "None",
            "recommended_action": actions[0] if actions else "None",
        },
    }


async def draft_customer_message(case: DeliveryCase) -> str:
    """Uses the llm_manager to draft an SMS."""
    return await asyncio.to_thread(customer_comm.draft_customer_message, build_customer_case(case))


async def simulate_customer_reply(drafted_message: str) -> str:
    """Uses the LLM to generate a fake customer reply."""
    try:
        return await asyncio.to_thread(customer_comm.simulate_customer_reply, drafted_message)
    except Exception:
        return "I am not available today, sorry."


async def parse_customer_reply(text: str) -> Dict[str, Any]:
    """Uses the llm_manager to parse intent (never raises: keyword fallback inside)."""
    return await asyncio.to_thread(customer_comm.parse_customer_reply, text)


async def analyze_exception_note(note: str) -> Dict[str, Any]:
    """Tier 2: RAG analysis of messy driver notes."""
    return await asyncio.to_thread(_analyze_exception_note, note)
