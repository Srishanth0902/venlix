"""
Demo: run the ML prediction samples through the agent graph and print each result.
(The automated test suite lives in tests/ - run `pytest`.)
"""
import asyncio
from delivery_agent.mapping import map_to_delivery_case, is_at_risk
from delivery_agent.runner import run_cases
from delivery_agent.sample_data import TEST_DATA, sample_deliveries
from pipeline import print_case


async def main():
    print("Testing dynamic risk factor routing with NEW PREDICTION DATA...\n")

    backend_data = sample_deliveries()
    print(f"Loaded {len(backend_data)} deliveries from mock data ({len(TEST_DATA)} ML samples + demo extras).")

    # Filter the high-risk deliveries
    at_risk_deliveries = [d for d in backend_data if is_at_risk(d)]
    print(f"Found {len(at_risk_deliveries)} deliveries at risk out of {len(backend_data)} total.\n")

    cases = [map_to_delivery_case(d, i, source="sample") for i, d in enumerate(at_risk_deliveries)]
    for final_state in await run_cases(cases):
        print_case(final_state)

if __name__ == "__main__":
    asyncio.run(main())
