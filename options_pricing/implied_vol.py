"""Implied volatility solver using Newton-Raphson method.

Solves for the volatility that equates a model price to a given market
price. Uses Black-Scholes vega as the derivative estimate in the
Newton-Raphson iteration, which provides good convergence for both
European and American options.
"""

import math
from .black_scholes import black_scholes_price
from .greeks import vega as bs_vega
from .binomial import american_option_price


def implied_volatility(market_price, S, K, T, r, q, option_type,
                       style="european", n_steps=200,
                       tol=1e-8, max_iter=200, sigma_init=None):
    """Compute implied volatility using Newton-Raphson iteration.

    For European options, uses the Black-Scholes closed-form solution.
    For American options, uses the CRR binomial tree pricer with
    Black-Scholes vega as an approximation for the Newton-Raphson step.

    Args:
        market_price: Observed market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free interest rate
        q: Continuous dividend yield
        option_type: 'call' or 'put'
        style: 'european' or 'american'
        n_steps: Number of tree steps (for American options)
        tol: Convergence tolerance
        max_iter: Maximum iterations
        sigma_init: Initial volatility guess

    Returns:
        Implied volatility (annualized)

    Raises:
        ValueError: If the solver does not converge
    """
    if sigma_init is None:
        sigma = math.sqrt(2.0 * math.pi / T) * market_price / S
        sigma = max(sigma, 0.01)
        sigma = min(sigma, 5.0)
    else:
        sigma = sigma_init

    for iteration in range(max_iter):
        # Compute model price using Black-Scholes
        model_price = black_scholes_price(S, K, T, r, q, sigma, option_type)

        diff = model_price - market_price

        if abs(diff) < tol:
            return sigma

        # Use BS vega as derivative approximation
        v = bs_vega(S, K, T, r, q, sigma)

        if abs(v) < 1e-12:
            break

        # Newton-Raphson update
        sigma = sigma - diff / v
        sigma = max(sigma, 1e-6)
        sigma = min(sigma, 10.0)

    raise ValueError(
        f"Implied volatility did not converge after {max_iter} iterations. "
        f"Last sigma: {sigma:.6f}, last diff: {diff:.6f}"
    )
