from typing import Optional
from agent.tools_api import (
    convert_from_usd, convert_to_usd,
    calculate_total_value, calculate_estimated_revenue,
    format_duration_hours, calculate_percentage
)

def convert_currency_from_usd(amount_usd: float, target_currency: str, rate: Optional[float] = None) -> float:
    return convert_from_usd(amount_usd, rate)

def convert_currency_to_usd(amount: float, source_currency: str, rate: Optional[float] = None) -> float:
    return convert_to_usd(amount, rate)

API_TOOLS_REGISTRY = {
    "convert_currency_from_usd": {
        "fn": convert_currency_from_usd,
        "description": "Convert an amount in USD to a target currency.",
        "schema": {
            "type": "object",
            "properties": {
                "amount_usd": {"type": "number"},
                "target_currency": {"type": "string"},
                "rate": {"type": ["number", "null"], "description": "Conversion rate from RAG (1 USD = rate <currency>). Set to null if RAG unavailable."}
            },
            "required": ["amount_usd", "target_currency"]
        }
    },
    "convert_currency_to_usd": {
        "fn": convert_currency_to_usd,
        "description": "Convert an amount from a given currency into USD.",
        "schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "source_currency": {"type": "string"},
                "rate": {"type": ["number", "null"], "description": "Conversion rate from RAG (1 USD = rate <currency>). Set to null if RAG unavailable."}
            },
            "required": ["amount", "source_currency"]
        }
    },
    "calculate_total_value": {
        "fn": calculate_total_value,
        "description": "Calculate total value from quantity and unit price.",
        "schema": {
            "type": "object",
            "properties": {
                "quantity": {"type": "number"},
                "unit_price": {"type": "number"}
            },
            "required": ["quantity", "unit_price"]
        }
    },
    "calculate_estimated_revenue": {
        "fn": calculate_estimated_revenue,
        "description": "Calculate estimated revenue from count and average amount.",
        "schema": {
            "type": "object",
            "properties": {
                "count": {"type": "number"},
                "average_amount": {"type": "number"}
            },
            "required": ["count", "average_amount"]
        }
    },
    "format_duration_hours": {
        "fn": format_duration_hours,
        "description": "Convert minutes to hours and minutes format.",
        "schema": {
            "type": "object",
            "properties": {
                "minutes": {"type": "number"}
            },
            "required": ["minutes"]
        }
    },
    "calculate_percentage": {
        "fn": calculate_percentage,
        "description": "Calculate percentage: (part / total) * 100.",
        "schema": {
            "type": "object",
            "properties": {
                "part": {"type": "number"},
                "total": {"type": "number"}
            },
            "required": ["part", "total"]
        }
    },
}

