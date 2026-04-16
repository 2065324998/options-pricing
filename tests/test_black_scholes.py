"""Tests for Black-Scholes European option pricing."""

import pytest
import math
from options_pricing import black_scholes_price


class TestEuropeanCall:
    def test_atm_call(self):
        """ATM call has positive value."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.20, "call")
        assert price > 0

    def test_deep_itm_call(self):
        """Deep ITM call approaches intrinsic value."""
        price = black_scholes_price(150, 100, 0.1, 0.05, 0.0, 0.20, "call")
        assert price > 49.0

    def test_deep_otm_call(self):
        """Deep OTM call approaches zero."""
        price = black_scholes_price(50, 100, 0.1, 0.05, 0.0, 0.20, "call")
        assert price < 0.01

    def test_known_call_price(self):
        """Verify known BS call price."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.20, "call")
        assert price == pytest.approx(10.4506, abs=0.01)

    def test_call_with_dividend(self):
        """Call price decreases with dividend yield."""
        p_no_div = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.20, "call")
        p_div = black_scholes_price(100, 100, 1.0, 0.05, 0.03, 0.20, "call")
        assert p_div < p_no_div

    def test_call_increases_with_vol(self):
        """Call price increases with volatility."""
        p_low = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.15, "call")
        p_high = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.30, "call")
        assert p_high > p_low


class TestEuropeanPut:
    def test_atm_put(self):
        """ATM put has positive value."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.20, "put")
        assert price > 0

    def test_deep_itm_put(self):
        """Deep ITM put approaches intrinsic value."""
        price = black_scholes_price(50, 100, 0.1, 0.05, 0.0, 0.20, "put")
        assert price > 49.0

    def test_known_put_price(self):
        """Verify known BS put price."""
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.20, "put")
        assert price == pytest.approx(5.5735, abs=0.01)

    def test_put_with_dividend(self):
        """Put price increases with dividend yield."""
        p_no_div = black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.20, "put")
        p_div = black_scholes_price(100, 100, 1.0, 0.05, 0.03, 0.20, "put")
        assert p_div > p_no_div


class TestPutCallParity:
    def test_parity_no_dividend(self):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.25
        call = black_scholes_price(S, K, T, r, q, sigma, "call")
        put = black_scholes_price(S, K, T, r, q, sigma, "put")
        parity = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert call - put == pytest.approx(parity, abs=0.001)

    def test_parity_with_dividend(self):
        """Put-call parity with dividends: C - P = S*exp(-qT) - K*exp(-rT)."""
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.03, 0.25
        call = black_scholes_price(S, K, T, r, q, sigma, "call")
        put = black_scholes_price(S, K, T, r, q, sigma, "put")
        parity = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert call - put == pytest.approx(parity, abs=0.001)


class TestBoundaryConditions:
    def test_invalid_option_type(self):
        """Invalid option type raises ValueError."""
        with pytest.raises(ValueError):
            black_scholes_price(100, 100, 1.0, 0.05, 0.0, 0.20, "straddle")

    def test_short_expiry(self):
        """Very short expiry converges to intrinsic."""
        call = black_scholes_price(110, 100, 0.001, 0.05, 0.0, 0.20, "call")
        assert call == pytest.approx(10.0, abs=0.5)
