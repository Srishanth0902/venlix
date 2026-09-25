import os
import time
import asyncio
import traceback
from typing import Dict, Any, Literal
from .state import DeliveryCase
from . import interfaces
from .llm_manager.client import capture_llm_calls
from .llm_manager.customer_comm import build_draft_prompt
from .llm_manager.models import DeliveryCase as CommCase

LLM_STEP_TIMEOUT = float(os.getenv("AGENT_LLM_TIMEOUT", "20"))


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


async def risk_detection_node(state: DeliveryCase) -> Dict[str, Any]:
    """Pulls context and infers failure_type from the weighted ML risk factors."""
    started = time.perf_counter()
    try:
        context = await interfaces.get_delivery_context(state["delivery_id"], state)
        classification = interfaces.classify_failure(state)
        inferred_failure = classification["failure_type"]
        top = ", ".join(f"{f['factor']} ({f['weight']:.0f})" for f in classification["factors"][:3]) or "no scored factors"

        return {
            "customer_context": context["customer_context"],
            "driver_context": context["driver_context"],
            "failure_type": inferred_failure,
            "failure_confidence": classification["confidence"],
            "started_at": state.get("started_at") or time.time(),
            "trace": [
                {"node": "risk_detection", "action": "Pulled context from DB"},
                {"node": "risk_detection",
                 "action": f"Weighed risk factors [{top}]. Deduced failure_type: {inferred_failure} "
                           f"(confidence {classification['confidence']:.0%})",
                 "ms": _elapsed_ms(started)},
            ]
        }
    except Exception as e:
        return {
            "status": "escalated",
            "started_at": state.get("started_at") or time.time(),
            "trace": [{"node": "risk_detection", "action": "Failed to pull context", "error": str(e)}]
        }


def route_resolution(state: DeliveryCase) -> Literal["deterministic_resolution_node", "llm_resolution_node", "escalation_node", "manager_node"]:
    """Conditional edge router based on failure_type or status."""
    if state.get("status") == "escalated":
        return "manager_node"

    failure_type = state.get("failure_type")

    if failure_type == "fraud":
        return "escalation_node"
    if failure_type == "driver_delay":
        # Driver-side problems are fixed operationally (reassign / re-route), no customer contact needed
        return "deterministic_resolution_node"
    # access_issue, address_issue, customer_unavailable -> talk to the customer
    return "llm_resolution_node"


async def deterministic_resolution_node(state: DeliveryCase) -> Dict[str, Any]:
    """Handles resolution without LLM, e.g., reassigning driver."""
    started = time.perf_counter()
    try:
        driver_context = state.get("driver_context") or {}
        actions = [a.get("action") for a in (state.get("ai_recommendation") or {}).get("recommended_actions") or []
                   if isinstance(a, dict) and a.get("action")]
        if driver_context.get("idle_driver_nearby", True):
            detail = "Reassigned to nearby idle driver"
            outcome = "resolved_auto"
        else:
            detail = "No idle driver nearby; re-routed current driver around congestion and pushed new ETA"
            outcome = "rerouted"
        trace = [{"node": "deterministic_resolution", "action": f"{detail} without LLM call", "ms": _elapsed_ms(started)}]
        if actions:
            trace.append({"node": "deterministic_resolution", "action": f"Applied ML recommended actions: {', '.join(actions)}"})
        return {
            "resolution_path": "auto_route",
            "resolution_detail": detail,
            "final_outcome": outcome,
            "trace": trace,
        }
    except Exception as e:
        return {
            "status": "escalated",
            "trace": [{"node": "deterministic_resolution", "action": "Error routing", "error": str(e)}]
        }


async def llm_resolution_node(state: DeliveryCase) -> Dict[str, Any]:
    """Contacts the customer: LLM-drafted SMS, (simulated) reply, LLM-parsed intent."""
    comm_case = interfaces.build_customer_case(state)
    problem_prompt = build_draft_prompt(CommCase.from_dict(comm_case))
    customer_name = comm_case["customer_name"]
    slot = comm_case["proposed_slot"]
    trace = []

    with capture_llm_calls() as llm_calls:
        try:
            started = time.perf_counter()
            try:
                message = await asyncio.wait_for(interfaces.draft_customer_message(state), timeout=LLM_STEP_TIMEOUT)
            except asyncio.TimeoutError:
                message = f"Hi {customer_name}, we ran into an issue with your delivery. Please reply if {slot} works for a reschedule."
                return {
                    "problem_prompt": problem_prompt,
                    "customer_message": message,
                    "resolution_path": "customer_contact",
                    "resolution_detail": "LLM timed out drafting the SMS; human agent to follow up",
                    "final_outcome": "escalated_timeout",
                    "status": "escalated",
                    "llm_metadata": {"calls": list(llm_calls)},
                    "trace": [{"node": "llm_resolution", "action": "LLM timed out, used canned fallback string",
                               "ms": _elapsed_ms(started)}]
                }
            trace.append({"node": "llm_resolution", "action": f"Drafted SMS to {customer_name}", "ms": _elapsed_ms(started)})

            # Simulate receiving a reply from the customer dynamically using the LLM!
            started = time.perf_counter()
            simulated_reply = await asyncio.wait_for(interfaces.simulate_customer_reply(message), timeout=LLM_STEP_TIMEOUT)
            trace.append({"node": "llm_resolution", "action": "Received customer reply (simulated)", "ms": _elapsed_ms(started)})

            started = time.perf_counter()
            intent = await asyncio.wait_for(interfaces.parse_customer_reply(simulated_reply), timeout=LLM_STEP_TIMEOUT)
            trace.append({"node": "llm_resolution", "action": f"Parsed reply intent: {intent}", "ms": _elapsed_ms(started)})
        except Exception:
            return {
                "problem_prompt": problem_prompt,
                "status": "escalated",
                "llm_metadata": {"calls": list(llm_calls)},
                "trace": trace + [{"node": "llm_resolution", "action": "Unhandled error during LLM path",
                                   "error": traceback.format_exc()}]
            }

    update: Dict[str, Any] = {
        "problem_prompt": problem_prompt,
        "customer_message": message,
        "customer_reply": simulated_reply,
        "customer_intent": intent,
        "resolution_path": "customer_contact",
        "llm_metadata": {"calls": list(llm_calls)},
    }
    if intent.get("declined"):
        update.update(final_outcome="customer_declined", status="escalated",
                      resolution_detail="Customer declined the reschedule; routed to a human agent")
    elif intent.get("wants_reschedule"):
        new_slot = intent.get("new_slot") or slot
        update.update(final_outcome="rescheduled", resolution_detail=f"Delivery rescheduled to {new_slot}")
    else:
        update.update(final_outcome="awaiting_customer", status="escalated",
                      resolution_detail="Customer reply was unclear; flagged for human follow-up")
    trace.append({"node": "llm_resolution", "action": f"Outcome: {update['final_outcome']} - {update['resolution_detail']}"})
    update["trace"] = trace
    return update


async def escalation_node(state: DeliveryCase) -> Dict[str, Any]:
    """Routes to human escalation queue."""
    return {
        "resolution_path": "escalation",
        "resolution_detail": f"{(state.get('failure_type') or 'risk').replace('_', ' ').capitalize()} signals detected; "
                             "sent to the human review queue",
        "final_outcome": "escalated",
        "status": "escalated",
        "trace": [{"node": "escalation", "action": "Routed directly to human escalation"}]
    }


PATH_SAVINGS = {
    "auto_route": {"time_min": 15.0, "fuel_inr": 50.0, "cost_inr": 75.0},
    "customer_contact": {"time_min": 5.0, "fuel_inr": 10.0, "cost_inr": 20.0},
}


async def manager_node(state: DeliveryCase) -> Dict[str, Any]:
    """Terminal node: Compiles trace, computes savings, writes DB and WS."""
    try:
        # Determine status if not already escalated
        current_status = state.get("status")
        final_status = current_status if current_status == "escalated" else "resolved"

        # A failed delivery attempt is only avoided when the case was actually resolved
        savings = {"time_min": 0.0, "fuel_inr": 0.0, "cost_inr": 0.0}
        if final_status == "resolved":
            savings = dict(PATH_SAVINGS.get(state.get("resolution_path"), savings))

        completed_at = time.time()
        started_at = state.get("started_at") or completed_at
        duration_ms = round((completed_at - started_at) * 1000.0, 1)
        final_trace = [{"node": "manager", "action": f"Compiled trace, calculated savings, set status to {final_status}"}]

        # Create an updated copy of the state manually to pass into interfaces
        state_copy = dict(state)
        state_copy.update(status=final_status, savings=savings, completed_at=completed_at, duration_ms=duration_ms,
                          trace=list(state.get("trace") or []) + final_trace)

        await interfaces.write_agent_log(state_copy)  # type: ignore
        await interfaces.broadcast_ws({"event": "case_completed", "case": state_copy})

        return {
            "status": final_status,
            "savings": savings,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "trace": final_trace,
        }
    except Exception as e:
        # Even manager_node can fail, but since it's terminal, we just append to trace
        return {
            "status": "escalated",
            "trace": [{"node": "manager", "action": "Failed to finalize case", "error": str(e)}]
        }
