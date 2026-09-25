"""
Tier 2: RAG Exception Analyzer.

Uses in-process ChromaDB over synthetic delivery exception notes to perform vector retrieval
of past similar cases, grounding the LLM in historical root causes and suggested solutions.
ChromaDB is optional: without it, a keyword-overlap retriever is used.
"""
import re
import time
import logging
import threading
from typing import Dict, Any, List, Optional

from .client import call_llm
from .customer_comm import _extract_json_object
from .prompts import (
    EXCEPTION_ANALYSIS_SYSTEM_PROMPT,
    EXCEPTION_ANALYSIS_USER_PROMPT,
)
from .synthetic_data import SYNTHETIC_EXCEPTION_NOTES

logger = logging.getLogger(__name__)

_chroma_collection = None
_chroma_unavailable = False
_chroma_lock = threading.Lock()


def init_chroma_collection():
    """
    Initialize the ChromaDB in-process collection on first use (not at import time,
    which used to slow down every import of the agent package).
    Seeds collection with 60 synthetic exception notes.
    """
    global _chroma_collection, _chroma_unavailable
    if _chroma_collection is not None or _chroma_unavailable:
        return _chroma_collection

    with _chroma_lock:
        if _chroma_collection is not None or _chroma_unavailable:
            return _chroma_collection
        start_t = time.perf_counter()
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
            collection = client.get_or_create_collection(name="exception_notes")
            if collection.count() == 0:
                collection.add(
                    documents=[item["note"] for item in SYNTHETIC_EXCEPTION_NOTES],
                    metadatas=[
                        {
                            "root_cause": item["root_cause"],
                            "suggested_solution": item["suggested_solution"],
                            "category": item["category"]
                        }
                        for item in SYNTHETIC_EXCEPTION_NOTES
                    ],
                    ids=[item["id"] for item in SYNTHETIC_EXCEPTION_NOTES],
                )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            logger.info(f"ChromaDB vector store ready ({collection.count()} notes) in {elapsed_ms:.0f} ms.")
            _chroma_collection = collection
        except Exception as err:
            _chroma_unavailable = True
            logger.info(f"ChromaDB unavailable ({err}); using keyword-based reference retrieval.")
        return _chroma_collection


def get_chroma_collection():
    return init_chroma_collection()


def analyze_exception_note(note: str) -> Dict[str, Any]:
    """Return {"root_cause", "suggested_solution", "confidence", "references", "retrieval_ms", "llm_ms"}."""
    if not note or not note.strip():
        return {
            "root_cause": "No exception note provided.",
            "suggested_solution": "Verify driver input and re-submit note.",
            "confidence": 0.0,
            "references": [],
        }

    t_retrieval_start = time.perf_counter()
    top_references = _retrieve_top_references(note, k=3)
    retrieval_ms = round((time.perf_counter() - t_retrieval_start) * 1000.0, 1)

    reference_cases_text = "\n".join(
        f"Reference Case {idx}:\n"
        f"  Driver Note: \"{ref['note']}\"\n"
        f"  Root Cause: {ref['root_cause']}\n"
        f"  Suggested Solution: {ref['suggested_solution']}\n"
        for idx, ref in enumerate(top_references, 1)
    )
    user_prompt = EXCEPTION_ANALYSIS_USER_PROMPT.format(
        current_note=note,
        reference_cases_text=reference_cases_text
    )

    t_llm_start = time.perf_counter()
    result: Optional[Dict[str, Any]] = None
    try:
        raw_response = call_llm(
            prompt=user_prompt,
            system=EXCEPTION_ANALYSIS_SYSTEM_PROMPT,
            max_tokens=300,
            timeout=8.0,
            task="exception_analysis",
        )
        data = _extract_json_object(raw_response)
        if data:
            result = {
                "root_cause": str(data.get("root_cause", "Unspecified root cause.")),
                "suggested_solution": str(data.get("suggested_solution", "Contact dispatch for resolution.")),
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.85)))),
            }
        else:
            logger.warning("Exception analysis response was not valid JSON; using top reference.")
    except Exception as err:
        logger.warning(f"Exception analysis LLM call failed ({err}); using top reference.")
    llm_ms = round((time.perf_counter() - t_llm_start) * 1000.0, 1)

    if result is None:
        if top_references:
            top_ref = top_references[0]
            result = {"root_cause": top_ref["root_cause"], "suggested_solution": top_ref["suggested_solution"],
                      "confidence": 0.82}
        else:
            result = {"root_cause": "Delivery exception requiring manual review.",
                      "suggested_solution": "Contact customer via SMS and escalate to human dispatch.",
                      "confidence": 0.75}

    result.update(references=top_references, retrieval_ms=retrieval_ms, llm_ms=llm_ms)
    return result


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "at", "in", "on", "is", "was", "for", "with", "but", "no",
    "not", "it", "by", "be", "as", "from", "that", "this", "has", "had", "are", "were", "driver", "customer",
}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _retrieve_top_references(note: str, k: int = 3) -> List[Dict[str, str]]:
    """Retrieve top-k similar historical notes from ChromaDB or keyword fallback."""
    collection = get_chroma_collection()

    if collection is not None:
        try:
            results = collection.query(query_texts=[note], n_results=k)
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = (results.get("metadatas") or [[]])[0]
                return [
                    {
                        "note": docs[i],
                        "root_cause": (metas[i] if i < len(metas) else {}).get("root_cause", "Historical delivery exception."),
                        "suggested_solution": (metas[i] if i < len(metas) else {}).get("suggested_solution", "Follow standard protocol."),
                        "category": (metas[i] if i < len(metas) else {}).get("category", ""),
                    }
                    for i in range(len(docs))
                ]
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}. Falling back to keyword match.")

    # Keyword-overlap fallback (punctuation-insensitive, stopwords removed, weighted by
    # overlap with the note, root cause and category text).
    query = _tokens(note)
    scored = []
    for item in SYNTHETIC_EXCEPTION_NOTES:
        doc = _tokens(item["note"])
        extra = _tokens(item["root_cause"] + " " + item["category"])
        score = 2 * len(query & doc) + len(query & extra)
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "note": item["note"],
            "root_cause": item["root_cause"],
            "suggested_solution": item["suggested_solution"],
            "category": item["category"],
        }
        for _, item in scored[:k]
    ]
