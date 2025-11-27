"""API tools placeholder for external API calls."""

from typing import Optional


# Simple currency conversion rates (placeholder)
# These represent "1 USD = X <currency>" rates (from RAG document)
CURRENCY_RATES = {
    "EUR": 1.18,  # 1 USD = 1.18 EUR
    "GBP": 1.37,  # 1 USD = 1.37 GBP
    "JPY": 110.0,  # 1 USD = 110 JPY
    "CAD": 1.25,  # 1 USD = 1.25 CAD
    "AUD": 1.35,  # 1 USD = 1.35 AUD
}


def convert_to_usd(amount: float, currency: str) -> float:
    """
    Simple deterministic conversion using fixed rates for a few currencies.
    
    This is a placeholder/demo tool. In production, this would call a real
    currency API.
    
    Parameters
    ----------
    amount : float
        Amount to convert
    currency : str
        Source currency code (EUR, GBP, JPY, CAD, AUD)
        
    Returns
    -------
    float
        Amount in USD
    """
    currency = currency.upper()
    if currency == "USD":
        return amount
    
    if currency not in CURRENCY_RATES:
        raise ValueError(f"Unsupported currency: {currency}. Supported: {list(CURRENCY_RATES.keys())}")
    
    # Convert to USD (divide by rate since CURRENCY_RATES stores "1 USD = X <currency>")
    # If 1 USD = 1.18 EUR, then 1 EUR = 1/1.18 USD
    return amount / CURRENCY_RATES[currency]


def convert_from_usd(amount_usd: float, target_currency: str) -> float:
    """
    Convert USD amount to target currency.
    
    Parameters
    ----------
    amount_usd : float
        Amount in USD to convert
    target_currency : str
        Target currency code (EUR, GBP, JPY, CAD, AUD)
        
    Returns
    -------
    float
        Amount in target currency
    """
    target_currency = target_currency.upper()
    if target_currency == "USD":
        return amount_usd
    
    if target_currency not in CURRENCY_RATES:
        raise ValueError(f"Unsupported currency: {target_currency}. Supported: {list(CURRENCY_RATES.keys())}")
    
    # Convert from USD: multiply by the rate
    # CURRENCY_RATES stores "1 USD = X <currency>" directly
    rate = CURRENCY_RATES[target_currency]
    return amount_usd * rate


def get_weather(city: str) -> dict:
    """
    Placeholder weather API tool.
    
    Returns a mock weather response. In production, this would call a real
    weather API.
    
    Parameters
    ----------
    city : str
        City name
        
    Returns
    -------
    dict
        Weather information
    """
    # Placeholder: return mock data
    return {
        "city": city,
        "temperature": 72,
        "condition": "sunny",
        "humidity": 65,
        "note": "This is placeholder data. Real implementation would call a weather API.",
    }

