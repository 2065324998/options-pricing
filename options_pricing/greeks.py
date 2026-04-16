"""Option Greeks using Black-Scholes-Merton analytical formulas.

Greeks measure the sensitivity of an option's price to changes in
underlying parameters: stock price (delta, gamma), volatility (vega),
time (theta), and interest rates (rho).
"""

import math
from .black_scholes import _cdf, _pdf, _d1, _d2


def delta(S, K, T, r, q, sigma, option_type):
    """Option delta: sensitivity to stock price changes.

    Call delta is in [0, 1]; put delta is in [-1, 0].
    """
    d1 = _d1(S, K, T, r, q, sigma)
    if option_type == "call":
        return math.exp(-q * T) * _cdf(d1)
    else:
        return math.exp(-q * T) * (_cdf(d1) - 1.0)


def gamma(S, K, T, r, q, sigma):
    """Option gamma: sensitivity of delta to stock price changes.

    Gamma is the same for calls and puts.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    return math.exp(-q * T) * _pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S, K, T, r, q, sigma):
    """Option vega: sensitivity to volatility changes.

    Returns the price change per unit change in sigma.
    Vega is the same for calls and puts.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _pdf(d1) * math.sqrt(T)


def theta(S, K, T, r, q, sigma, option_type):
    """Option theta: rate of time decay (per year).

    Theta is typically negative for long option positions.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)

    term1 = -S * math.exp(-q * T) * _pdf(d1) * sigma / (2.0 * math.sqrt(T))

    if option_type == "call":
        term2 = q * S * math.exp(-q * T) * _cdf(d1)
        term3 = r * K * math.exp(-r * T) * _cdf(d2)
        return term1 + term2 - term3
    else:
        term2 = q * S * math.exp(-q * T) * _cdf(-d1)
        term3 = r * K * math.exp(-r * T) * _cdf(-d2)
        return term1 - term2 + term3


def rho(S, K, T, r, q, sigma, option_type):
    """Option rho: sensitivity to interest rate changes.

    Returns the price change per unit change in interest rate.
    """
    d2 = _d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return K * T * math.exp(-r * T) * _cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * _cdf(-d2)
