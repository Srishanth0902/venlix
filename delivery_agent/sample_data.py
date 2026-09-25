"""
Sample payloads for demos, the dashboard's "Run sample cases" button, and tests.

TEST_DATA holds real outputs of the Venlix-XGBoost-v2 prediction service.
DEMO_EXTRA_DATA adds two synthetic cases so every resolution path (driver
reassignment and fraud escalation) is exercised in a demo run.
"""
import copy
from typing import Any, Dict, List

TEST_DATA: List[Dict[str, Any]] = [
    {   "success": True,   "prediction": "Low Risk",   "prediction_class": 0,   "risk_score": 0,   "confidence": 99,   "risk_level": "Low",   "risk_factors": [     {       "factor": "High Address Confidence",       "impact": 90     },     {       "factor": "Approved Visitor Pass",       "impact": 80     },     {       "factor": "Excellent Driver Reliability",       "impact": 73     },     {       "factor": "Customer Reachable",       "impact": 52     },     {       "factor": "Quick Customer Response",       "impact": 14     }   ],   "recommended_actions": [     {       "action": "Proceed Normally",       "priority": "Low",       "expected_improvement": 0     }   ],   "estimated_success_after_action": 99,   "estimated_time_saved_minutes": 0,   "estimated_cost_saved_rupees": 0,   "estimated_fuel_saved_liters": 0,   "model": "Venlix-XGBoost-v2",   "timestamp": "2026-08-07T06:54:46.500102Z" },
    {   "success": True,   "prediction": "Critical Risk",   "prediction_class": 1,   "risk_score": 99,   "confidence": 99,   "risk_level": "Critical",   "risk_factors": [     {       "factor": "High Gate Wait Time",       "impact": 95     },     {       "factor": "Customer Response Time",       "impact": 24     },     {       "factor": "Visitor Pass Pending",       "impact": 10     },     {       "factor": "Driver Reliability Score",       "impact": 10     },     {       "factor": "Previous Failed Deliveries",       "impact": 10     }   ],   "recommended_actions": [     {       "action": "Notify Security Gate",       "priority": "High",       "expected_improvement": 8     },     {       "action": "Request Visitor Approval",       "priority": "Medium",       "expected_improvement": 15     },     {       "action": "Monitor Delivery",       "priority": "Medium",       "expected_improvement": 10     }   ],   "estimated_success_after_action": 34,   "estimated_time_saved_minutes": 0,   "estimated_cost_saved_rupees": 0,   "estimated_fuel_saved_liters": 0,   "model": "Venlix-XGBoost-v2",   "timestamp": "2026-08-07T06:55:22.405336Z" },
    {   "success": True,   "prediction": "Critical Risk",   "prediction_class": 1,   "risk_score": 99,   "confidence": 99,   "risk_level": "Critical",   "risk_factors": [     {       "factor": "Customer Response Time",       "impact": 93     },     {       "factor": "Customer Unavailable",       "impact": 10     },     {       "factor": "Previous Failed Deliveries",       "impact": 10     },     {       "factor": "Low Address Confidence",       "impact": 10     },     {       "factor": "Driver Reliability Score",       "impact": 10     }   ],   "recommended_actions": [     {       "action": "Offer Reschedule",       "priority": "Medium",       "expected_improvement": 11     },     {       "action": "Monitor Delivery",       "priority": "Medium",       "expected_improvement": 8     },     {       "action": "Verify Address",       "priority": "Medium",       "expected_improvement": 12     }   ],   "estimated_success_after_action": 32,   "estimated_time_saved_minutes": 7,   "estimated_cost_saved_rupees": 17,   "estimated_fuel_saved_liters": 0,   "model": "Venlix-XGBoost-v2",   "timestamp": "2026-08-07T06:55:59.084239Z" },
    {   "success": True,   "prediction": "Critical Risk",   "prediction_class": 1,   "risk_score": 99,   "confidence": 99,   "risk_level": "Critical",   "risk_factors": [     {       "factor": "High Gate Wait Time",       "impact": 90     },     {       "factor": "Customer Response Time",       "impact": 47     },     {       "factor": "Previous Failed Deliveries",       "impact": 10     },     {       "factor": "Low Address Confidence",       "impact": 10     },     {       "factor": "Driver Reliability Score",       "impact": 10     }   ],   "recommended_actions": [     {       "action": "Notify Security Gate",       "priority": "High",       "expected_improvement": 10     },     {       "action": "Monitor Delivery",       "priority": "Medium",       "expected_improvement": 10     },     {       "action": "Verify Address",       "priority": "Medium",       "expected_improvement": 9     }   ],   "estimated_success_after_action": 30,   "estimated_time_saved_minutes": 8,   "estimated_cost_saved_rupees": 26,   "estimated_fuel_saved_liters": 0,   "model": "Venlix-XGBoost-v2",   "timestamp": "2026-08-07T06:56:26.284531Z" },
    {   "success": True,   "prediction": "Critical Risk",   "prediction_class": 1,   "risk_score": 99,   "confidence": 99,   "risk_level": "Critical",   "risk_factors": [     {       "factor": "Customer Response Time",       "impact": 95     },     {       "factor": "High Gate Wait Time",       "impact": 15     },     {       "factor": "Previous Failed Deliveries",       "impact": 10     },     {       "factor": "Customer Unavailable",       "impact": 10     },     {       "factor": "Visitor Pass Pending",       "impact": 10     }   ],   "recommended_actions": [     {       "action": "Notify Security Gate",       "priority": "Medium",       "expected_improvement": 10     },     {       "action": "Monitor Delivery",       "priority": "Medium",       "expected_improvement": 10     },     {       "action": "Offer Reschedule",       "priority": "Medium",       "expected_improvement": 11     },     {       "action": "Request Visitor Approval",       "priority": "Medium",       "expected_improvement": 13     }   ],   "estimated_success_after_action": 45,   "estimated_time_saved_minutes": 0,   "estimated_cost_saved_rupees": 0,   "estimated_fuel_saved_liters": 0,   "model": "Venlix-XGBoost-v2",   "timestamp": "2026-08-07T06:56:47.411824Z" }
]

# Illustrative names so drafted SMS messages read naturally in demos.
_DEMO_PEOPLE = [
    ({"customer_id": "CUST-001", "name": "Priya Sharma"}, {"driver_id": "DRV-001", "name": "Ravi Kumar"}),
    ({"customer_id": "CUST-002", "name": "Arjun Mehta"}, {"driver_id": "DRV-002", "name": "Sana Qureshi"}),
    ({"customer_id": "CUST-003", "name": "Kavya Reddy"}, {"driver_id": "DRV-003", "name": "Imran Shaikh"}),
    ({"customer_id": "CUST-004", "name": "Rahul Nair"}, {"driver_id": "DRV-004", "name": "Deepa Iyer"}),
    ({"customer_id": "CUST-005", "name": "Ananya Rao"}, {"driver_id": "DRV-005", "name": "Vikram Singh"}),
]

DEMO_EXTRA_DATA: List[Dict[str, Any]] = [
    {
        "delivery_id": "DEL-DEMO-FRAUD",
        "prediction": "Critical Risk", "risk_score": 97, "risk_level": "Critical",
        "customer": {"customer_id": "CUST-900", "name": "Unknown Recipient"},
        "driver": {"driver_id": "DRV-009", "name": "Meera Joshi"},
        "risk_factors": [
            {"factor": "Suspicious Order Pattern", "impact": 92},
            {"factor": "Drop Distance Anomaly", "impact": 61},
            {"factor": "Customer Response Time", "impact": 12},
        ],
        "recommended_actions": [{"action": "Hold For Verification", "priority": "High", "expected_improvement": 20}],
        "model": "Venlix-XGBoost-v2",
    },
    {
        "delivery_id": "DEL-DEMO-TRAFFIC",
        "prediction": "Critical Risk", "risk_score": 91, "risk_level": "Critical",
        "customer": {"customer_id": "CUST-901", "name": "Neha Kapoor"},
        "driver": {"driver_id": "DRV-010", "name": "Suresh Babu", "status": "stuck in traffic"},
        "environment": {"weather": "Heavy Rain", "traffic": "Jam"},
        "risk_factors": [
            {"factor": "Heavy Traffic Congestion", "impact": 88},
            {"factor": "Adverse Weather", "impact": 54},
            {"factor": "Driver Reliability Score", "impact": 20},
        ],
        "recommended_actions": [{"action": "Reassign Driver", "priority": "High", "expected_improvement": 25}],
        "model": "Venlix-XGBoost-v2",
    },
]


def sample_deliveries(include_extras: bool = True) -> List[Dict[str, Any]]:
    """TEST_DATA enriched with demo customer/driver names (plus the extra demo cases)."""
    enriched = []
    for index, row in enumerate(copy.deepcopy(TEST_DATA)):
        customer, driver = _DEMO_PEOPLE[index % len(_DEMO_PEOPLE)]
        row.setdefault("delivery_id", f"DEL-TEST-{index + 1}")
        row.setdefault("customer", dict(customer))
        row.setdefault("driver", dict(driver))
        enriched.append(row)
    if include_extras:
        enriched.extend(copy.deepcopy(DEMO_EXTRA_DATA))
    return enriched
