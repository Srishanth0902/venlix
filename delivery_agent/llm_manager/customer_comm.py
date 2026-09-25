"""
Customer Communication Module (Tier 1).

Implements:
- draft_customer_message(case: DeliveryCase | dict) -> str
- simulate_customer_reply(drafted_message: str) -> str
- parse_customer_reply(text: str) -> dict
"""
import re
import json
import logging
from typing import Union, Dict, Any, Optional

from .models import DeliveryCase
from .client import call_llm
from .offline import parse_reply_keywords
from .prompts import (
    CUSTOMER_DRAFT_SYSTEM_PROMPT,
    CUSTOMER_DRAFT_USER_PROMPT,
    CUSTOMER_REPLY_SIM_SYSTEM_PROMPT,
    CUSTOMER_REPLY_SIM_USER_PROMPT,
    REPLY_PARSE_SYSTEM_PROMPT,
    REPLY_PARSE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

SMS_WORD_LIMIT = 45

_ASCII_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
})


def _to_ascii(text: str) -> str:
    """SMS gateways mangle smart punctuation and emojis; normalise to plain ASCII."""
    text = text.translate(_ASCII_MAP).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def _extract_tagged(raw: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return None


def _usable_untagged(raw: str, max_words: int) -> Optional[str]:
    """Accept an untagged answer only if it looks like the message itself, not a ramble."""
    cleaned = re.sub(r"</?\w+>", "", raw).strip().strip('"\'')
    if cleaned and len(cleaned.split()) <= max_words and "\n\n" not in cleaned:
        return cleaned
    return None


def build_draft_prompt(case_obj: DeliveryCase) -> str:
    """The exact user prompt sent to the LLM for an SMS draft."""
    metadata = case_obj.metadata or {}
    return CUSTOMER_DRAFT_USER_PROMPT.format(
        customer_name=case_obj.customer_name,
        failure_type=case_obj.failure_type,
        driver_context=case_obj.driver_context,
        risk_details=metadata.get("risk_details") or "None",
        recommended_action=metadata.get("recommended_action") or "None",
        proposed_slot=case_obj.proposed_slot,
    )


def draft_customer_message(case: Union[DeliveryCase, Dict[str, Any]], timeout: float = 8.0) -> str:
    """
    Turns a DeliveryCase into a friendly SMS-style message.
    System prompt fixes tone ('friendly, concise, one question at a time') and forces short output (<40 words).
    Optional case.metadata keys: risk_details, recommended_action.
    """
    case_obj = DeliveryCase.from_dict(case) if isinstance(case, dict) else case

    try:
        raw_response = call_llm(
            prompt=build_draft_prompt(case_obj),
            system=CUSTOMER_DRAFT_SYSTEM_PROMPT,
            max_tokens=200,
            timeout=timeout,
            task="sms_draft",
        )
    except Exception as err:
        # A dead LLM must not stop the customer from being contacted
        logger.error(f"SMS draft LLM call failed ({err}); using template.")
        raw_response = ""

    sms = _extract_tagged(raw_response, "sms") or _usable_untagged(raw_response, 60)
    if not sms:
        logger.warning("SMS draft was unusable; using template.")
        sms = (f"Hi {case_obj.customer_name}, we hit an issue with your delivery ({case_obj.failure_type}). "
               f"Can we reschedule for {case_obj.proposed_slot}?")

    sms = _to_ascii(sms)
    words = sms.split()
    if len(words) > SMS_WORD_LIMIT:
        # Gracefully trim to stay within the limit while keeping the question
        sms = " ".join(words[:SMS_WORD_LIMIT - 1]).rstrip(",;:") + ("?" if "?" in sms else ".")
    return sms


def simulate_customer_reply(drafted_message: str, timeout: float = 8.0) -> str:
    """
    Autonomously generates a realistic fake customer reply to a drafted SMS message.
    Used to make the multi-agent test fully dynamic.
    """
    try:
        raw_response = call_llm(
            prompt=CUSTOMER_REPLY_SIM_USER_PROMPT.format(message=drafted_message),
            system=CUSTOMER_REPLY_SIM_SYSTEM_PROMPT,
            max_tokens=120,
            timeout=timeout,
            task="customer_reply_sim",
            temperature=0.9,
        )
        reply = _extract_tagged(raw_response, "reply") or _usable_untagged(raw_response, 40)
        return _to_ascii(reply) if reply else "Sure, that works for me."
    except Exception as e:
        logger.error(f"Failed to simulate reply: {e}")
        return "I can't take it right now, please come tomorrow."


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_customer_reply(text: str, timeout: float = 6.0) -> Dict[str, Any]:
    """
    Turns freeform customer reply text into structured intent dictionary:
    {"wants_reschedule": bool, "new_slot": str|None, "declined": bool}

    Validates JSON; if parsing fails or LLM errors, falls back to keyword matching
    so the pipeline never breaks on malformed LLM responses.
    """
    if not text or not text.strip():
        return {"wants_reschedule": False, "new_slot": None, "declined": False}

    user_prompt = REPLY_PARSE_USER_PROMPT.format(reply_text=text.strip())

    try:
        raw_response = call_llm(
            prompt=user_prompt,
            system=REPLY_PARSE_SYSTEM_PROMPT,
            max_tokens=100,
            timeout=timeout,
            task="reply_parse",
            temperature=0.0,
        )
        data = _extract_json_object(raw_response)
        if data and "wants_reschedule" in data and "declined" in data:
            return {
                "wants_reschedule": bool(data.get("wants_reschedule")),
                "new_slot": data.get("new_slot") or None,
                "declined": bool(data.get("declined"))
            }
        logger.warning("LLM response did not match schema. Executing keyword matching fallback.")
    except Exception as err:
        logger.warning(f"LLM call failed ({err}). Executing keyword matching fallback.")
    return _parse_reply_keyword_fallback(text)


def _parse_reply_keyword_fallback(text: str) -> Dict[str, Any]:
    """
    Deterministic rule-based keyword matcher fallback.
    Prevents pipeline failures on malformed LLM output.
    """
    return parse_reply_keywords(text)
