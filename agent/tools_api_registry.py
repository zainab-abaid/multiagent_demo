# agent/tools_api_registry.py

from agent.tools_api import convert_from_usd, convert_to_usd, get_weather

# Wrapper functions to match registry schema exactly
def convert_currency_from_usd(amount_usd: float, target_currency: str) -> float:
    """Wrapper for convert_from_usd to match registry schema."""
    return convert_from_usd(amount_usd, target_currency)

def convert_currency_to_usd(amount: float, source_currency: str) -> float:
    """Wrapper for convert_to_usd to match registry schema."""
    return convert_to_usd(amount, source_currency)

# later: import many more API functions

API_TOOLS_REGISTRY = {
    "convert_currency_from_usd": {
        "fn": convert_currency_from_usd,
        "description": "Convert an amount in USD to a target currency.",
        "schema": {
            "type": "object",
            "properties": {
                "amount_usd": {"type": "number"},
                "target_currency": {"type": "string"}
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
                "source_currency": {"type": "string"}
            },
            "required": ["amount", "source_currency"]
        }
    },
    "get_weather": {
        "fn": get_weather,
        "description": "Get weather info for a given city.",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    },
    # later: add 7–8 more tiny API tools here
}

