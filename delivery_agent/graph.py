
from functools import lru_cache
from langgraph.graph import StateGraph, END
from .state import DeliveryCase
from .nodes import (
    risk_detection_node,
    route_resolution,
    deterministic_resolution_node,
    llm_resolution_node,
    escalation_node,
    manager_node
)

def create_delivery_graph():
    # Initialize the graph with our state schema
    workflow = StateGraph(DeliveryCase)
    
    # Add nodes
    workflow.add_node("risk_detection", risk_detection_node)
    workflow.add_node("deterministic_resolution", deterministic_resolution_node)
    workflow.add_node("llm_resolution", llm_resolution_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("manager", manager_node)
    
    # Add edges
    workflow.set_entry_point("risk_detection")
    
    # Conditional routing from risk detection to resolution paths
    workflow.add_conditional_edges(
        "risk_detection",
        route_resolution,
        {
            "deterministic_resolution_node": "deterministic_resolution",
            "llm_resolution_node": "llm_resolution",
            "escalation_node": "escalation",
            "manager_node": "manager" # If escalated early
        }
    )
    
    # All resolution paths lead to the manager (terminal node)
    workflow.add_edge("deterministic_resolution", "manager")
    workflow.add_edge("llm_resolution", "manager")
    workflow.add_edge("escalation", "manager")
    
    # Manager ends the graph
    workflow.add_edge("manager", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


@lru_cache(maxsize=1)
def get_delivery_graph():
    """Compiled graph singleton (compiling on every case wastes time)."""
    return create_delivery_graph()
