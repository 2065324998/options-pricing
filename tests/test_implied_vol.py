"""Basic tests for implied volatility solver."""

import pytest
from options_pricing import implied_volatility, black_scholes_price


class TestImpliedVolBasic:
    def test_european_call_recovery(self):
        """Recover sigma from a European call price."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.25, "call")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, 0.0, "call")
        assert iv == pytest.approx(0.25, abs=0.001)

    def test_european_put_recovery(self):
        """Recover sigma from a European put price."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.25, "put")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, 0.0, "put")
        assert iv == pytest.approx(0.25, abs=0.001)

    def test_high_vol_recovery(self):
        """Recover high volatility."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.60, "call")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, 0.0, "call")
        assert iv == pytest.approx(0.60, abs=0.001)

    def test_low_vol_recovery(self):
        """Recover low volatility."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.10, "put")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, 0.0, "put")
        assert iv == pytest.approx(0.10, abs=0.001)
