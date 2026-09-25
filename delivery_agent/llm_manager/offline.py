"""
Offline synthesizer used when no LLM provider is configured or reachable.

It never pretends to be a general LLM: it answers what it can compute exactly
(arithmetic, dates, the attached backend / agent data), produces well-formed
outputs for the agent's internal tasks (SMS drafts, reply parsing, summaries),
and says clearly when a question needs a live model.
"""
import ast
import json
import math
import operator
import re
import zlib
from datetime import datetime
from typing import Any, Dict, List, Optional

OFFLINE_NOTICE = (
    "I'm running in **offline mode** (no LLM provider is configured or reachable), so I can only "
    "answer from the platform's data, arithmetic and date/time questions.\n\n"
    "To get answers to *any* question, add `GEMINI_API_KEY` (or `OPENROUTER_API_KEY`) to your `.env` "
    "file and restart the dashboard, or on Vercel to the project's environment variables and redeploy."
)

# ---------------------------------------------------------------------------
# Task inference (for legacy callers that don't pass task=...)
# ---------------------------------------------------------------------------

def infer_task(system: Optional[str], prompt: str) -> str:
    sys_lower = (system or "").lower()
    prompt_lower = prompt.lower()
    if "sms-style" in sys_lower or "draft the sms message" in prompt_lower:
        return "sms_draft"
    if "role-play a delivery customer" in sys_lower:
        return "customer_reply_sim"
    if "intent parser" in sys_lower:
        return "reply_parse"
    if "executive summary" in sys_lower:
        return "decision_summary"
    if "root cause analyst" in sys_lower:
        return "exception_analysis"
    return "chat"


def synthesize(task: str, prompt: str, api_context: Optional[Dict[str, Any]] = None) -> str:
    """Produce an offline response for the given task."""
    if task == "sms_draft":
        return _draft_sms(prompt)
    if task == "customer_reply_sim":
        return _simulate_reply(prompt)
    if task == "reply_parse":
        return json.dumps(parse_reply_keywords(_field(prompt, "Customer Reply").strip('"') or prompt))
    if task == "decision_summary":
        return _summarize_trace(prompt)
    if task == "exception_analysis":
        return _analyze_exception(prompt)
    return answer_question(prompt, api_context or {})


def _field(text: str, name: str) -> str:
    """Extract 'Name: value' from a structured prompt."""
    match = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Customer communication
# ---------------------------------------------------------------------------

_ISSUE_PHRASES = {
    "access_issue": "a gate / visitor-pass hold-up at your address",
    "customer_unavailable": "us not being able to reach you",
    "address_issue": "an unconfirmed delivery address",
    "driver_delay": "a delay on your driver's route",
    "fraud": "a verification check on your order",
}

_ACTION_ASKS = {
    "access_issue": " Please approve our visitor pass or share the gate code.",
    "address_issue": " Please confirm your full address.",
}


def _draft_sms(prompt: str) -> str:
    name = _field(prompt, "Customer Name") or "there"
    ftype = (_field(prompt, "Delivery Failure Type") or _field(prompt, "Failure Type") or "delivery issue").strip()
    slot = _field(prompt, "Proposed New Slot") or _field(prompt, "Proposed Slot") or "tomorrow at 2 PM"
    key = ftype.lower().replace(" ", "_")
    issue = _ISSUE_PHRASES.get(key)
    if not issue:
        driver_ctx = _field(prompt, "Driver Context")
        issue = driver_ctx.rstrip(".") if driver_ctx and len(driver_ctx) < 60 else f"a delivery issue ({ftype})"
    sms = f"Hi {name}, your delivery is delayed due to {issue}.{_ACTION_ASKS.get(key, '')} Can we reschedule for {slot}?"
    return f"<sms>{sms}</sms>"


_SIM_REPLIES = [
    "Yes, {slot} works for me, thanks for the heads up.",
    "Sure, I will be home then.",
    "Can you come tomorrow at 2 PM instead?",
    "No, please cancel the order, I do not need it anymore.",
]


def _simulate_reply(prompt: str) -> str:
    match = re.search(r'"(.*)"', prompt, re.DOTALL)
    message = match.group(1) if match else prompt
    slot_match = re.search(r"reschedule for ([^?.!]+)", message, re.IGNORECASE)
    slot = slot_match.group(1).strip() if slot_match else "that time"
    # Deterministic per message, so runs are reproducible but cases differ.
    reply = _SIM_REPLIES[zlib.crc32(message.encode()) % len(_SIM_REPLIES)].format(slot=slot)
    return f"<reply>{reply}</reply>"


_POSITIVE_NEGATIONS = re.compile(r"\bno (problem|worries|issue|issues)\b")
_STRONG_DECLINE = re.compile(r"\b(cancel|refuse|decline|stop|do not want|don'?t want|never ?mind|return it)\b")
_SOFT_DECLINE = re.compile(r"\b(no|nope|not home|not available|won'?t|wont|can'?t|cannot|don'?t|dont)\b")
_POSITIVE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|fine|works?|great|perfect|sounds good|reschedule|alright|please do|good)\b"
)
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b")
_DAY = re.compile(r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b")
_PART_OF_DAY = re.compile(r"\b(morning|afternoon|evening)\b")


def _extract_slot(text: str) -> Optional[str]:
    # The last mention wins: counter-proposals come after the rejection
    # ("not home today, come tomorrow at 2pm").
    days = _DAY.findall(text)
    times = _TIME.findall(text)
    parts = _PART_OF_DAY.findall(text)
    pieces = []
    if days:
        pieces.append(days[-1].capitalize())
    if times:
        hour, minute, meridiem = times[-1]
        pieces.append(f"{int(hour)}:{minute or '00'} {meridiem.upper()}")
    elif parts:
        pieces.append(parts[-1])
    return " ".join(pieces) if pieces else None


def parse_reply_keywords(text: str) -> Dict[str, Any]:
    """Deterministic, word-boundary keyword parser for customer replies."""
    lower = (text or "").lower().strip()
    if not lower:
        return {"wants_reschedule": False, "new_slot": None, "declined": False}

    cleaned = _POSITIVE_NEGATIONS.sub(" ", lower)
    slot = _extract_slot(cleaned)

    if _STRONG_DECLINE.search(cleaned):
        return {"wants_reschedule": False, "new_slot": None, "declined": True}
    if slot:
        # "Can't do 5, but tomorrow 2pm works" -> a counter-proposal, not a decline.
        return {"wants_reschedule": True, "new_slot": slot, "declined": False}
    if _SOFT_DECLINE.search(cleaned) and not _POSITIVE.search(cleaned):
        return {"wants_reschedule": False, "new_slot": None, "declined": True}
    if _POSITIVE.search(cleaned) or _POSITIVE_NEGATIONS.search(lower):
        return {"wants_reschedule": True, "new_slot": None, "declined": False}
    return {"wants_reschedule": False, "new_slot": None, "declined": False}


# ---------------------------------------------------------------------------
# Decision summary & exception analysis
# ---------------------------------------------------------------------------

def _summarize_trace(prompt: str) -> str:
    # [ \t] rather than \s: the pattern must never run across line breaks.
    steps = re.findall(r"^[ \t]*\d+\.[ \t]*\[([^\]\n]+)\][ \t]*(.+?)(?:[ \t]*->[ \t]*(.*))?$", prompt, re.MULTILINE)
    if not steps:
        return "The agent processed the case, but no trace steps were available to summarize."
    actions = [action.strip().rstrip(".") for _, action, _ in steps]
    # Lower-case the leading verb, but keep acronyms ("LLM timed out") intact.
    actions = [a if a[1:2].isupper() else a[0].lower() + a[1:] for a in actions if a]
    body, last = actions[:-1], actions[-1]
    summary = f"The agent handled this case in {len(actions)} steps"
    summary += (": " + "; ".join(body) + ".") if body else "."
    summary += f" Final step: {last}."
    return summary


def _analyze_exception(prompt: str) -> str:
    root = _field(prompt, "Root Cause") or "Delivery exception requiring manual review."
    solution = _field(prompt, "Suggested Solution") or "Contact customer via SMS and escalate to human dispatch."
    return json.dumps({"root_cause": root, "suggested_solution": solution, "confidence": 0.8})


# ---------------------------------------------------------------------------
# Copilot chat (offline)
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "sin": math.sin, "cos": math.cos}
_FUNC_RE = "|".join(_FUNCS)
_MATH_FILLER = {
    "what", "what's", "whats", "is", "calculate", "compute", "evaluate", "solve", "how", "much", "please",
    "the", "value", "answer", "equals", "equal", "to", "tell", "me", "can", "you", "result", "of",
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def _try_math(question: str) -> Optional[str]:
    text = question.lower()
    text = re.sub(r"(\d)\s*[x×]\s*(\d)", r"\1*\2", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", r"(\1/100*\2)", text)
    text = text.replace("^", "**").replace("÷", "/")
    text = re.sub(r"\b(plus)\b", "+", text)
    text = re.sub(r"\b(minus)\b", "-", text)
    text = re.sub(r"\b(times|multiplied by)\b", "*", text)
    text = re.sub(r"\b(divided by|over)\b", "/", text)
    candidates = re.findall(rf"(?:{_FUNC_RE}|[\d.(])(?:{_FUNC_RE}|[\d\s.+\-*/%()])*", text)
    for candidate in sorted(candidates, key=len, reverse=True):
        expr = candidate.strip().rstrip("?.= ")
        has_op = re.search(rf"[+\-*/%]|{_FUNC_RE}", expr)
        if not expr or not re.search(r"\d", expr) or not has_op:
            continue
        # Only treat it as math when the rest of the question is filler ("what is ...?"),
        # so data questions like "deliveries with risk 0.7-0.9" are not evaluated.
        rest = set(re.findall(r"[a-z']+", text.replace(candidate, " "))) - _MATH_FILLER
        if rest:
            continue
        try:
            value = _safe_eval(ast.parse(expr, mode="eval"))
        except Exception:
            continue
        if isinstance(value, float):
            value = round(value, 6)
            if value.is_integer():
                value = int(value)
        return f"The answer is **{value}** (computed as `{expr.replace(' ', '')}`)."
    return None


def _fmt_delivery(d: Dict[str, Any]) -> str:
    case_id = d.get("case_id") or f"DEL-{d.get('id')}"
    risk = d.get("risk_score", 0.0)
    return (
        f"- **{case_id}** ({d.get('customer_name', 'Recipient')}) - {d.get('status', 'Unknown')}; "
        f"risk {risk:.2f}; issue: {d.get('failure_type', 'n/a')}; driver: {d.get('driver_name', 'n/a')}"
    )


def _source_note(ctx: Dict[str, Any]) -> str:
    return f"\n\n_Source: {ctx['data_source']}._" if ctx.get("data_source") else ""


def _fmt_case(c: Dict[str, Any]) -> str:
    return (
        f"- **{c.get('delivery_id')}** - {c.get('status')} ({c.get('final_outcome') or 'n/a'}); "
        f"failure type: {c.get('failure_type') or 'n/a'}; path: {c.get('resolution_path') or 'n/a'}"
        + (f"; reason: {c['resolution_detail']}" if c.get("resolution_detail") else "")
    )


def answer_question(question: str, ctx: Dict[str, Any]) -> str:
    q = question.strip()
    ql = q.lower()
    if not q:
        return "Ask me anything about your deliveries, agent cases, or operations."

    math_answer = _try_math(q)
    if math_answer:
        return math_answer

    if re.search(r"\b(what(?:'s| is) the (time|date)|current (time|date)|today'?s date|what day is it|time now|what time)\b", ql):
        now = datetime.now()
        return f"It is **{now:%A, %d %B %Y, %H:%M}** (server local time)."

    if re.match(r"^\s*(hi|hello|hey|greetings|good (morning|afternoon|evening))\b", ql):
        return (
            "Hello! I'm the Venlix Operations Copilot. Ask me about failure rates, high-risk or failed deliveries, "
            "a specific case ID, predictions, or what the agent did with each case."
        )

    if re.search(r"\b(help|what can you do|capabilities)\b", ql):
        return (
            "I can answer questions about:\n- Today's report / failure rate\n- Failed and high-risk deliveries\n"
            "- A specific delivery or case ID, customer or driver\n- XGBoost predictions and the digital twin map\n"
            "- Cases handled by the autonomous agent (escalations, outcomes, savings)\n\n" + OFFLINE_NOTICE
        )

    deliveries = [d for d in ctx.get("deliveries_api", []) if isinstance(d, dict)]
    cases = ctx.get("agent_cases_api") or {}
    case_list = cases.get("cases", []) if isinstance(cases, dict) else []

    # Specific IDs / names
    ids = {m.upper() for m in re.findall(r"\b(?:HERO|DEL|CASE)-[A-Z0-9-]+\b", q, re.IGNORECASE)}
    if ids:
        lines = []
        for d in deliveries:
            if str(d.get("case_id", "")).upper() in ids or f"DEL-{d.get('id')}" in ids:
                lines.append(_fmt_delivery(d))
        for c in case_list:
            if str(c.get("delivery_id", "")).upper() in ids:
                lines.append(_fmt_case(c) + (f"\n  - SMS sent: \"{c['customer_message']}\"" if c.get("customer_message") else ""))
        if lines:
            return "Here is what I found:\n" + "\n".join(lines) + _source_note(ctx)
        return f"I couldn't find {', '.join(sorted(ids))} in the delivery data or agent cases." + _source_note(ctx)

    named = [d for d in deliveries if any(
        part and re.search(rf"\b{re.escape(part.lower())}\b", ql)
        for part in str(d.get("customer_name", "")).split() + str(d.get("driver_name", "")).split()
    )]
    if named:
        return "Deliveries matching that name:\n" + "\n".join(_fmt_delivery(d) for d in named) + _source_note(ctx)

    if cases and re.search(r"\b(case|escalat|resolved|saving|agent|sms|reschedul|outcome)", ql):
        totals = cases.get("totals", {})
        lines = [
            f"The agent has processed **{totals.get('cases', len(case_list))}** cases: "
            f"{totals.get('resolved', 0)} resolved, {totals.get('escalated', 0)} escalated.",
        ]
        if "saving" in ql:
            sv = totals.get("savings", {})
            lines.append(
                f"Total savings: {sv.get('time_min', 0):.0f} min, Rs {sv.get('cost_inr', 0):.0f} cost, "
                f"Rs {sv.get('fuel_inr', 0):.0f} fuel."
            )
        wanted = case_list
        if "escalat" in ql:
            wanted = [c for c in case_list if c.get("status") == "escalated"]
        elif "resolved" in ql:
            wanted = [c for c in case_list if c.get("status") == "resolved"]
        if wanted:
            lines.append("\n".join(_fmt_case(c) for c in wanted[:15]))
        return "\n".join(lines)

    reports = ctx.get("reports_api")
    if reports and re.search(r"\b(rate|summary|overview|report|statistic|stats|kpi|how many|total|count)", ql):
        return (
            "**Operations summary**\n"
            f"- Total deliveries: {reports.get('total_predictions', 'n/a')}\n"
            f"- Successful: {reports.get('delivery_success', 'n/a')}\n"
            f"- Failed: {reports.get('delivery_failures', 'n/a')}\n"
            f"- Failure rate: {reports.get('failure_rate', 'n/a')}\n"
            f"- High-risk deliveries: {reports.get('high_risk_count', 'n/a')}\n"
            f"- Average model confidence: {reports.get('average_confidence', 'n/a')}" + _source_note(ctx)
        )

    if deliveries and re.search(r"\bfail", ql):
        failed = [d for d in deliveries if d.get("delivery_failed") == 1]
        if not failed:
            return "There are no failed deliveries in the current data." + _source_note(ctx)
        return f"**{len(failed)} failed deliveries:**\n" + "\n".join(_fmt_delivery(d) for d in failed) + _source_note(ctx)

    if deliveries and re.search(r"\brisk", ql):
        risky = sorted((d for d in deliveries if d.get("risk_score", 0) >= 0.7), key=lambda d: -d.get("risk_score", 0))
        if not risky:
            return "No deliveries currently have a risk score of 0.70 or higher." + _source_note(ctx)
        return (f"**{len(risky)} high-risk deliveries (risk >= 0.70):**\n" + "\n".join(_fmt_delivery(d) for d in risky)
                + _source_note(ctx))

    pred = ctx.get("prediction_api")
    if pred:
        summary = pred.get("input_summary", {})
        return (
            "**XGBoost prediction**\n"
            f"- Outcome: {pred.get('prediction', 'n/a')}\n"
            f"- Confidence: {float(pred.get('confidence', 0)) * 100:.0f}%\n"
            + "".join(f"- {k.replace('_', ' ').capitalize()}: {v}\n" for k, v in summary.items())
            + _source_note(ctx)
        ).rstrip()

    twin = ctx.get("twin_api")
    if twin:
        nodes = twin.get("digital_twin_nodes", [])
        lines = [f"- {n.get('label')} @ ({n.get('lat'):.4f}, {n.get('lng'):.4f}) - risk {n.get('risk_score', 0):.2f}, {n.get('status')}"
                 for n in nodes]
        return (f"**Digital twin:** {twin.get('active_nodes', len(nodes))} active nodes, "
                f"{twin.get('high_risk_alert_count', 0)} high-risk alerts.\n" + "\n".join(lines) + _source_note(ctx))

    if deliveries:
        return f"**{len(deliveries)} deliveries:**\n" + "\n".join(_fmt_delivery(d) for d in deliveries) + _source_note(ctx)

    return OFFLINE_NOTICE
