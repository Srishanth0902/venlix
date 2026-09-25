import asyncio
import time
from delivery_agent.mapping import map_to_delivery_case, is_at_risk
from delivery_agent.runner import run_cases
from delivery_agent.llm_manager.copilot import get_deliveries
from delivery_agent.llm_manager.client import get_provider_status


def map_ml_to_delivery_case(ml_data: dict, index: int):
    """Backwards-compatible alias for the shared ingest mapper."""
    return map_to_delivery_case(ml_data, index, source="backend")


def print_case(final_state: dict) -> None:
    print(f"--- Delivery {final_state.get('delivery_id')} (Risk Score: {final_state.get('risk_score', 0):.2f}) ---")
    print(f"FINAL STATUS: {final_state.get('status')}")
    print(f"FINAL OUTCOME: {final_state.get('final_outcome')}")
    print(f"FAILURE TYPE: {final_state.get('failure_type')}")
    print(f"RESOLUTION PATH: {final_state.get('resolution_path')} - {final_state.get('resolution_detail')}")

    if final_state.get("customer_message"):
        print(f"[LLM] DRAFTED SMS TO CUSTOMER: '{final_state.get('customer_message')}'")
    if final_state.get("customer_reply"):
        print(f"[LLM] SIMULATED CUSTOMER REPLY: '{final_state.get('customer_reply')}'")
    if final_state.get("customer_intent"):
        print(f"[LLM] PARSED INTENT: {final_state.get('customer_intent')}")

    print("\nTRACE:")
    for t in final_state.get("trace", []):
        timing = f" ({t['ms']:.0f} ms)" if t.get("ms") is not None else ""
        print(f"  [{t.get('node')}] {t.get('action')}{timing}")
        if "error" in t:
            print(f"      ERROR: {t['error']}")
    print(f"Case completed in {final_state.get('duration_ms') or 0:.0f} ms")
    print("-" * 60 + "\n")


async def process_deliveries(concurrency: int = 4):
    print("=" * 60)
    print("STEP 4: Predict + dispatch")
    print("=" * 60)
    print(f"LLM provider chain: {get_provider_status()['chain']}")

    # Dynamically extract data from the backend
    try:
        backend_data = get_deliveries()
        print(f"Successfully fetched {len(backend_data)} deliveries from backend.")
    except Exception as e:
        print(f"Error fetching from backend: {e}")
        return

    # 1. Filter the high-risk deliveries
    at_risk_deliveries = [d for d in backend_data if is_at_risk(d)]
    print(f"Found {len(at_risk_deliveries)} deliveries at risk out of {len(backend_data)} total.\n")

    # 2. Dispatch them to the Multi-agent (concurrently; results print in order)
    cases = [map_to_delivery_case(d, i, source="backend") for i, d in enumerate(at_risk_deliveries)]
    started = time.perf_counter()
    results = await run_cases(cases, concurrency=concurrency)
    for final_state in results:
        print_case(final_state)
    print(f"Processed {len(results)} cases in {(time.perf_counter() - started) * 1000:.0f} ms "
          f"(concurrency={concurrency}).")
    print("STEP 9 & 10: broadcast_ws() -> Live UI Update triggered!\n")

if __name__ == "__main__":
    asyncio.run(process_deliveries())
