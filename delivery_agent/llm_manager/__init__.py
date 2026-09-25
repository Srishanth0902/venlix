"""
LLM Manager Module.

Provides the single unified entry point for all LLM calls, prompts, client/fallback plumbing,
customer communication, decision summarization, and Tier-2 RAG exception analysis.
"""

from .models import DeliveryCase
from .client import (
    call_llm,
    call_llm_detailed,
    stream_llm,
    set_force_primary_failure,
    get_llm_metrics,
    get_provider_status,
    capture_llm_calls,
)
from .customer_comm import draft_customer_message, parse_customer_reply, simulate_customer_reply
from .decision import summarize_decision
from .exception_analyzer import analyze_exception_note

__all__ = [
    "call_llm",
    "call_llm_detailed",
    "stream_llm",
    "set_force_primary_failure",
    "get_llm_metrics",
    "get_provider_status",
    "capture_llm_calls",
    "draft_customer_message",
    "parse_customer_reply",
    "simulate_customer_reply",
    "summarize_decision",
    "analyze_exception_note",
    "DeliveryCase",
]
