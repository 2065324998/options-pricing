"""Tests for Black-Scholes option Greeks."""

import pytest
import math
from options_pricing.greeks import delta, gamma, vega, theta, rho
from options_pricing import black_scholes_price


S, K, T, R, Q, SIGMA = 100, 100, 1.0, 0.05, 0.0, 0.25


class TestDelta:
    def test_call_delta_positive(self):
        d = delta(S, K, T, R, Q, SIGMA, "call")
        assert 0 < d < 1

    def test_put_delta_negative(self):
        d = delta(S, K, T, R, Q, SIGMA, "put")
        assert -1 < d < 0

    def test_call_put_delta_relation(self):
        """Call delta - put delta = exp(-qT)."""
        dc = delta(S, K, T, R, Q, SIGMA, "call")
        dp = delta(S, K, T, R, Q, SIGMA, "put")
        assert dc - dp == pytest.approx(math.exp(-Q * T), abs=0.001)

    def test_itm_call_delta_near_one(self):
        d = delta(150, K, 0.1, R, Q, SIGMA, "call")
        assert d > 0.95

    def test_delta_with_dividend(self):
        dc = delta(S, K, T, R, 0.03, SIGMA, "call")
        dp = delta(S, K, T, R, 0.03, SIGMA, "put")
        assert dc - dp == pytest.approx(math.exp(-0.03 * T), abs=0.001)


class TestGamma:
    def test_gamma_positive(self):
        g = gamma(S, K, T, R, Q, SIGMA)
        assert g > 0

    def test_atm_gamma_highest(self):
        """ATM gamma > OTM gamma."""
        g_atm = gamma(100, 100, T, R, Q, SIGMA)
        g_otm = gamma(100, 130, T, R, Q, SIGMA)
        assert g_atm > g_otm

    def test_gamma_decreases_with_time(self):
        g_short = gamma(S, K, 0.1, R, Q, SIGMA)
        g_long = gamma(S, K, 2.0, R, Q, SIGMA)
        assert g_short > g_long


class TestVega:
    def test_vega_positive(self):
        v = vega(S, K, T, R, Q, SIGMA)
        assert v > 0

    def test_atm_vega_highest(self):
        v_atm = vega(100, 100, T, R, Q, SIGMA)
        v_otm = vega(100, 140, T, R, Q, SIGMA)
        assert v_atm > v_otm

    def test_vega_increases_with_time(self):
        v_short = vega(S, K, 0.25, R, Q, SIGMA)
        v_long = vega(S, K, 1.0, R, Q, SIGMA)
        assert v_long > v_short


class TestTheta:
    def test_call_theta_negative(self):
        """ATM call theta is typically negative."""
        t = theta(S, K, T, R, Q, SIGMA, "call")
        assert t < 0

    def test_put_theta_negative(self):
        """ATM put theta is typically negative."""
        t = theta(S, K, T, R, Q, SIGMA, "put")
        assert t < 0


class TestRho:
    def test_call_rho_positive(self):
        r_val = rho(S, K, T, R, Q, SIGMA, "call")
        assert r_val > 0

    def test_put_rho_negative(self):
        r_val = rho(S, K, T, R, Q, SIGMA, "put")
        assert r_val < 0
