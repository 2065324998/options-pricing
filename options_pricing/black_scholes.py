"""European option pricing using the Black-Scholes-Merton model.

Provides closed-form solutions for European call and put options on
stocks that may pay a continuous dividend yield. The model assumes
log-normal stock price dynamics with constant volatility.
"""

import math


def _cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x):
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(S, K, T, r, q, sigma):
    """Compute d1 in the Black-Scholes formula."""
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S, K, T, r, q, sigma):
    """Compute d2 in the Black-Scholes formula."""
    return _d1(S, K, T, r, q, sigma) - sigma * math.sqrt(T)


def black_scholes_price(S, K, T, r, q, sigma, option_type):
    """Calculate the Black-Scholes price for a European option.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free interest rate (annualized, continuous)
        q: Continuous dividend yield (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)

    if option_type == "call":
        price = S * math.exp(-q * T) * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * _cdf(-d2) - S * math.exp(-q * T) * _cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return price
