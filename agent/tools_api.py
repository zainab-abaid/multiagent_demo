"""API tools for external API calls."""

import warnings
from typing import Optional

def convert_to_usd(amount: float, rate: Optional[float] = None) -> float:
    """Convert amount from foreign currency to USD. Rate format: 1 USD = rate <currency>."""
    if rate is None:
        warnings.warn("No conversion rate provided. Returning amount unchanged.", UserWarning)
        return amount
    return round(amount / rate, 2)


def convert_from_usd(amount_usd: float, rate: Optional[float] = None) -> float:
    """Convert USD amount to foreign currency. Rate format: 1 USD = rate <currency>."""
    if rate is None:
        warnings.warn("No conversion rate provided. Returning amount unchanged.", UserWarning)
        return amount_usd
    return round(amount_usd * rate, 2)


def calculate_total_value(quantity: float, unit_price: float) -> float:
    """Calculate total value from quantity and unit price."""
    return round(quantity * unit_price, 2)


def calculate_estimated_revenue(count: float, average_amount: float) -> float:
    """Calculate estimated revenue from count and average amount."""
    return round(count * average_amount, 2)


def format_duration_hours(minutes: float) -> dict:
    """Convert minutes to hours and minutes format."""
    hours = int(minutes // 60)
    remaining_minutes = round(minutes % 60, 2)
    return {
        "hours": hours,
        "minutes": remaining_minutes,
        "formatted": f"{hours} hours and {remaining_minutes} minutes"
    }


def calculate_percentage(part: float, total: float) -> float:
    """Calculate percentage: (part / total) * 100."""
    if total == 0:
        return 0.0
    return round((part / total) * 100, 2)

