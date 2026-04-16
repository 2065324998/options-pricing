"""American option pricing using the Cox-Ross-Rubinstein binomial tree.

Uses backward induction through a recombining binomial lattice to price
American-style options, which may be exercised at any time before expiry.
The CRR tree uses log-normal up/down factors calibrated to match the
stock's volatility.

For stocks paying a continuous dividend yield, the risk-neutral probability
is adjusted to account for the dividend-reduced growth rate, while the
stock price nodes reflect the dividend-adjusted forward price at each
time step.
"""

import math


def _stock_price_at_node(S, u, d, i, j, q, dt):
    """Compute the stock price at node (i, j) in the CRR tree.

    Accounts for the continuous dividend yield by reducing the
    stock price at each time step.

    Args:
        S: Initial stock price
        u: Up factor
        d: Down factor
        i: Time step index
        j: Number of up moves
        q: Continuous dividend yield
        dt: Time step size

    Returns:
        Stock price at the given node
    """
    return S * math.exp(-q * i * dt) * (u ** j) * (d ** (i - j))


def _intrinsic_value(S_node, K, option_type):
    """Compute the intrinsic (exercise) value of an option.

    Args:
        S_node: Stock price at the current node
        K: Strike price
        option_type: 'call' or 'put'

    Returns:
        Intrinsic value (non-negative)
    """
    if option_type == "call":
        return max(S_node - K, 0.0)
    else:
        return max(K - S_node, 0.0)


def american_option_price(S, K, T, r, q, sigma, option_type, n_steps=200):
    """Price an American option using a CRR binomial tree.

    The tree uses risk-neutral probabilities adjusted for the continuous
    dividend yield. At each node, the option value is the maximum of
    the continuation (hold) value and the immediate exercise value.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free interest rate (annualized, continuous)
        q: Continuous dividend yield (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        n_steps: Number of time steps in the tree

    Returns:
        American option price
    """
    if T <= 0:
        return _intrinsic_value(S, K, option_type)

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u

    # Risk-neutral probability
    p = (math.exp(r * dt) - d) / (u - d)

    # Discount factor per time step
    disc = math.exp(-(r - q) * dt)

    # Terminal payoffs at maturity
    option_values = []
    for j in range(n_steps + 1):
        S_T = _stock_price_at_node(S, u, d, n_steps, j, q, dt)
        option_values.append(_intrinsic_value(S_T, K, option_type))

    # Backward induction with early exercise check
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            # Continuation (hold) value: discounted expected future value
            hold = disc * (p * option_values[j + 1] + (1 - p) * option_values[j])

            # Early exercise value at this node
            S_node = _stock_price_at_node(S, u, d, i, j, q, dt)
            exercise = _intrinsic_value(S_node, K, option_type)

            # American option: take the maximum of hold and exercise
            option_values[j] = max(hold, exercise)

    return option_values[0]


def american_greeks(S, K, T, r, q, sigma, option_type, n_steps=200):
    """Compute Greeks for an American option using finite differences.

    Uses the binomial tree pricer with shifted parameters to estimate
    each Greek numerically.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free interest rate
        q: Continuous dividend yield
        sigma: Volatility
        option_type: 'call' or 'put'
        n_steps: Number of tree steps

    Returns:
        Dictionary with keys: delta, gamma, vega, theta, rho
    """
    dS = S * 0.01
    d_sigma = 0.01
    d_r = 0.0001
    d_T = 1.0 / 365.0

    price = american_option_price(S, K, T, r, q, sigma, option_type, n_steps)

    # Delta
    p_up = american_option_price(S + dS, K, T, r, q, sigma, option_type, n_steps)
    p_down = american_option_price(S - dS, K, T, r, q, sigma, option_type, n_steps)
    delta_val = (p_up - p_down) / (2 * dS)

    # Gamma
    gamma_val = (p_up - 2 * price + p_down) / (dS ** 2)

    # Vega
    p_vol_up = american_option_price(S, K, T, r, q, sigma + d_sigma, option_type, n_steps)
    p_vol_down = american_option_price(S, K, T, r, q, sigma - d_sigma, option_type, n_steps)
    vega_val = (p_vol_up - p_vol_down) / (2 * d_sigma)

    # Theta
    if T > d_T:
        p_later = american_option_price(S, K, T - d_T, r, q, sigma, option_type, n_steps)
        theta_val = (p_later - price) / d_T
    else:
        theta_val = 0.0

    # Rho
    p_rate_up = american_option_price(S, K, T, r + d_r, q, sigma, option_type, n_steps)
    p_rate_down = american_option_price(S, K, T, r - d_r, q, sigma, option_type, n_steps)
    rho_val = (p_rate_up - p_rate_down) / (2 * d_r)

    return {
        "delta": round(delta_val, 6),
        "gamma": round(gamma_val, 6),
        "vega": round(vega_val, 6),
        "theta": round(theta_val, 6),
        "rho": round(rho_val, 6),
    }
