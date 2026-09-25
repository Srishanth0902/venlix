"""
Decision Summarization Module (Tier 1).

Implements:
- summarize_decision(trace: list) -> str
"""
import logging
from typing import List, Union, Dict, Any

from .client import call_llm
from .prompts import (
    DECISION_SUMMARY_SYSTEM_PROMPT,
    DECISION_SUMMARY_USER_PROMPT,
)

logger = logging.getLogger(__name__)

def summarize_decision(trace: List[Union[str, Dict[str, Any]]]) -> str:
    """
    Turns an agent trace into a 2-3 sentence plain-English explanation for judges and dashboards.
    """
    if not trace:
        return "No agent trace steps were provided to summarize."

    # Format trace items cleanly into bullet points
    formatted_steps = []
    for idx, step in enumerate(trace, 1):
        if isinstance(step, dict):
            step_name = step.get("step", step.get("node", f"Step {idx}"))
            action = step.get("action", step.get("description", str(step)))
            result = step.get("result", step.get("output", ""))
            formatted_steps.append(f"{idx}. [{step_name}] {action} -> {result}")
        else:
            formatted_steps.append(f"{idx}. {str(step)}")

    trace_text = "\n".join(formatted_steps)
    user_prompt = DECISION_SUMMARY_USER_PROMPT.format(trace_text=trace_text)

    summary = call_llm(
        prompt=user_prompt,
        system=DECISION_SUMMARY_SYSTEM_PROMPT,
        max_tokens=250,
        timeout=8.0,
        task="decision_summary",
    )

    return summary.strip()
