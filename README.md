# Options Pricing Library

A Python library for pricing financial options using the Black-Scholes-Merton
model and Cox-Ross-Rubinstein binomial trees.

## Features

- European option pricing (Black-Scholes-Merton closed-form)
- Option Greeks (delta, gamma, theta, vega, rho)
- American option pricing (CRR binomial tree with early exercise)
- Implied volatility solver (Newton-Raphson)

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from options_pricing import black_scholes_price, american_option_price, implied_volatility

# European call
price = black_scholes_price(S=100, K=100, T=1.0, r=0.05, q=0.02,
                            sigma=0.20, option_type="call")

# American put on dividend-paying stock
price = american_option_price(S=100, K=100, T=1.0, r=0.05, q=0.03,
                              sigma=0.25, option_type="put")

# Implied volatility
iv = implied_volatility(market_price=10.5, S=100, K=100, T=1.0,
                        r=0.05, q=0.0, option_type="call")
```

## Parameters

- `S`: Current stock price
- `K`: Strike price
- `T`: Time to expiration (years)
- `r`: Risk-free interest rate (annualized, continuous compounding)
- `q`: Continuous dividend yield (annualized)
- `sigma`: Volatility (annualized)
- `option_type`: `"call"` or `"put"`
