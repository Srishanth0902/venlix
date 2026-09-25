import operator
from typing import TypedDict, Annotated, Optional, List, Dict, Any

class DeliveryCase(TypedDict):
    delivery_id: str
    customer_id: str
    driver_id: str
    risk_score: float  # normalised to 0..1 (ML payloads sending 0..100 are scaled on ingest)
    risk_reason: List[str]
    failure_type: Optional[str]
    failure_confidence: Optional[float]
    store_location: Optional[Dict[str, float]]
    drop_location: Optional[Dict[str, float]]

    # New Payload Fields
    driver: Optional[Dict[str, Any]]
    customer: Optional[Dict[str, Any]]
    environment: Optional[Dict[str, Any]]
    risk_factors: Optional[List[Dict[str, Any]]]
    ai_recommendation: Optional[Dict[str, Any]]

    customer_context: Dict[str, Any]
    driver_context: Dict[str, Any]
    resolution_path: str
    resolution_detail: Optional[str]
    problem_prompt: Optional[str]
    customer_message: Optional[str]
    customer_reply: Optional[str]
    customer_intent: Optional[Dict[str, Any]]
    final_outcome: Optional[str]
    savings: Dict[str, float]
    status: str

    # Where the case came from ("sample", "backend", "custom") and timing
    source: Optional[str]
    started_at: Optional[float]
    completed_at: Optional[float]
    duration_ms: Optional[float]

    # LLM calls made for this case (provider, model, latency) - filled by the LLM node
    llm_metadata: Optional[Dict[str, Any]]

    # Annotated with operator.add to ensure it is append-only
    trace: Annotated[List[Dict[str, Any]], operator.add]
