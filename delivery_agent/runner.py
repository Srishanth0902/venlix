"""
Run many delivery cases through the agent graph concurrently.

Cases spend most of their time waiting on LLM I/O, so running them concurrently
(bounded by a semaphore to respect provider rate limits) cuts batch wall-time
roughly by the concurrency factor.
"""
import time
import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .graph import get_delivery_graph
from .interfaces import broadcast_ws
from .state import DeliveryCase
from .env_utils import env_int

DEFAULT_CONCURRENCY = env_int("AGENT_CONCURRENCY", 4)


async def run_case(case: DeliveryCase) -> Dict[str, Any]:
    case["started_at"] = time.time()
    await broadcast_ws({"event": "case_started", "delivery_id": case["delivery_id"], "risk_score": case.get("risk_score")})
    return await get_delivery_graph().ainvoke(case)


async def run_cases(
    cases: List[DeliveryCase],
    concurrency: int = DEFAULT_CONCURRENCY,
    on_result: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
) -> List[Dict[str, Any]]:
    """Run cases with bounded concurrency; results are returned in input order."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(case: DeliveryCase) -> Dict[str, Any]:
        async with semaphore:
            try:
                result = await run_case(case)
            except Exception as err:  # never let one bad case sink the batch
                result = {**case, "status": "escalated", "final_outcome": "agent_error",
                          "trace": list(case.get("trace") or []) + [{"node": "runner", "action": "Graph failed", "error": str(err)}]}
        if on_result:
            await on_result(result)
        return result

    return await asyncio.gather(*(_one(c) for c in cases))
