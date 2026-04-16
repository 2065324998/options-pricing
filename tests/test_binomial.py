"""Basic tests for American option pricing via binomial tree."""

import pytest
from options_pricing import american_option_price


class TestAmericanBasic:
    def test_put_positive(self):
        p = american_option_price(100, 100, 1.0, 0.05, 0.0, 0.25, "put")
        assert p > 0

    def test_call_positive(self):
        p = american_option_price(100, 100, 1.0, 0.05, 0.0, 0.25, "call")
        assert p > 0

    def test_put_increases_with_strike(self):
        p1 = american_option_price(100, 90, 1.0, 0.05, 0.0, 0.25, "put")
        p2 = american_option_price(100, 110, 1.0, 0.05, 0.0, 0.25, "put")
        assert p2 > p1

    def test_call_decreases_with_strike(self):
        p1 = american_option_price(100, 90, 1.0, 0.05, 0.0, 0.25, "call")
        p2 = american_option_price(100, 110, 1.0, 0.05, 0.0, 0.25, "call")
        assert p1 > p2

    def test_put_increases_with_vol(self):
        p1 = american_option_price(100, 100, 1.0, 0.05, 0.0, 0.15, "put")
        p2 = american_option_price(100, 100, 1.0, 0.05, 0.0, 0.30, "put")
        assert p2 > p1

    def test_expired_option(self):
        p = american_option_price(100, 110, 0, 0.05, 0.0, 0.25, "put")
        assert p == pytest.approx(10.0, abs=0.01)
