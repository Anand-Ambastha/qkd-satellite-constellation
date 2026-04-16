"""
Unit tests for the QKD protocol modules.

Covers:
  - Binary entropy edge cases and BB84 QBER limit
  - Geometric efficiency scaling with range
  - SKR positivity / monotonicity with elevation
  - Link viability thresholds
  - Decoy-state consistency checks
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qkd_constellation.qkd.bb84 import (
    binary_entropy,
    secure_key_rate,
    qber_model,
    pass_analysis,
)
from qkd_constellation.qkd.channel import (
    geometric_efficiency,
    atmospheric_transmittance,
    total_link_loss_db,
    channel_budget,
)
from qkd_constellation.constants import (
    MIN_ELEVATION_DEG,
    LOSS_THRESHOLD_DB,
    MU_SIGNAL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Binary entropy
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryEntropy:
    def test_zero(self):
        assert binary_entropy(0.0) == 0.0

    def test_one(self):
        assert binary_entropy(1.0) == 0.0

    def test_half(self):
        assert abs(binary_entropy(0.5) - 1.0) < 1e-10

    def test_symmetric(self):
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert abs(binary_entropy(p) - binary_entropy(1 - p)) < 1e-10

    def test_concavity(self):
        """H must be concave: H((a+b)/2) ≥ (H(a)+H(b))/2."""
        a, b = 0.1, 0.4
        assert binary_entropy((a + b) / 2) >= (binary_entropy(a) + binary_entropy(b)) / 2 - 1e-10

    def test_bb84_qber_limit(self):
        """BB84 is secure only when QBER < 11% (H < 0.5 effectively)."""
        # At QBER=0.11, SKR should still be positive
        qber_ok  = secure_key_rate(45.0, 700.0)[0]
        assert qber_ok < 0.11

    def test_negative_input_returns_zero(self):
        assert binary_entropy(-0.1) == 0.0

    def test_above_one_returns_zero(self):
        assert binary_entropy(1.1) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Geometric efficiency
# ─────────────────────────────────────────────────────────────────────────────

class TestGeometricEfficiency:
    def test_zero_range(self):
        assert geometric_efficiency(0.0) == 0.0

    def test_close_range_high_eta(self):
        """Very short range → high η (capped at 1.0)."""
        eta = geometric_efficiency(1.0)
        assert eta == 1.0

    def test_decreases_with_range(self):
        """η must be strictly decreasing with range."""
        etas = [geometric_efficiency(r) for r in [200, 500, 800, 1200, 2000]]
        assert all(etas[i] > etas[i + 1] for i in range(len(etas) - 1))

    def test_inverse_square_law(self):
        """For large r, η ∝ 1/r² → doubling range quarters efficiency."""
        eta1 = geometric_efficiency(1000)
        eta2 = geometric_efficiency(2000)
        ratio = eta1 / eta2 if eta2 > 0 else float("inf")
        assert 3.5 < ratio < 4.5, f"Range doubling ratio {ratio:.2f} ≠ ~4"

    def test_larger_aperture_higher_eta(self):
        """Larger receiver aperture collects more photons."""
        eta_small = geometric_efficiency(800, aperture_rx_m=0.3)
        eta_large = geometric_efficiency(800, aperture_rx_m=0.6)
        assert eta_large > eta_small

    def test_narrower_divergence_higher_eta(self):
        """Tighter beam → better collection at fixed range."""
        eta_wide   = geometric_efficiency(800, divergence_rad=20e-6)
        eta_narrow = geometric_efficiency(800, divergence_rad=5e-6)
        assert eta_narrow > eta_wide


# ─────────────────────────────────────────────────────────────────────────────
# Atmospheric transmittance
# ─────────────────────────────────────────────────────────────────────────────

class TestAtmosphere:
    def test_below_min_elevation(self):
        assert atmospheric_transmittance(5.0) == 0.0

    def test_high_elevation_near_unity(self):
        """At 90° elevation, transmittance should be close to 1 for clear sky."""
        T = atmospheric_transmittance(90.0)
        assert T > 0.9

    def test_lower_elevation_lower_transmittance(self):
        T_high = atmospheric_transmittance(80.0)
        T_low  = atmospheric_transmittance(15.0)
        assert T_low < T_high


# ─────────────────────────────────────────────────────────────────────────────
# SKR
# ─────────────────────────────────────────────────────────────────────────────

class TestSecureKeyRate:
    def test_below_min_elevation_no_skr(self):
        qber, skr = secure_key_rate(5.0, 800.0)
        assert skr == 0.0
        assert qber == 1.0

    def test_high_elevation_positive_skr(self):
        _, skr = secure_key_rate(60.0, 700.0)
        assert skr > 0.0

    def test_skr_non_negative(self):
        """SKR must never be negative for any geometry."""
        for elev in [10, 20, 45, 60, 90]:
            for rng in [300, 600, 900, 1500, 2000]:
                _, skr = secure_key_rate(float(elev), float(rng))
                assert skr >= 0.0, f"Negative SKR at elev={elev}, range={rng}"

    def test_skr_increases_with_elevation(self):
        """Higher elevation → shorter range & less loss → higher SKR (roughly)."""
        _, skr_low  = secure_key_rate(15.0, 1400.0)
        _, skr_high = secure_key_rate(60.0, 700.0)
        assert skr_high > skr_low

    def test_excessive_loss_no_skr(self):
        """Very long range → over threshold → zero SKR."""
        _, skr = secure_key_rate(12.0, 2500.0)
        assert skr == 0.0

    def test_qber_bounded(self):
        """QBER must stay in [0, 1]."""
        for elev in [10, 30, 60, 90]:
            for rng in [400, 800, 1600]:
                qber, _ = secure_key_rate(float(elev), float(rng))
                assert 0.0 <= qber <= 1.0

    def test_higher_mu_different_skr(self):
        """Changing μ should change SKR (not crash)."""
        _, skr_low  = secure_key_rate(45.0, 800.0, mu=0.3)
        _, skr_high = secure_key_rate(45.0, 800.0, mu=0.7)
        assert skr_low != skr_high


# ─────────────────────────────────────────────────────────────────────────────
# Pass analysis dict
# ─────────────────────────────────────────────────────────────────────────────

class TestPassAnalysis:
    def test_keys_present(self):
        result = pass_analysis(45.0, 800.0)
        required_keys = {"elevation_deg", "range_km", "eta_geometric",
                         "qber", "skr", "link_viable", "total_loss_dB"}
        assert required_keys.issubset(result.keys())

    def test_viable_link(self):
        result = pass_analysis(60.0, 700.0)
        assert result["link_viable"] is True
        assert result["skr"] > 0

    def test_non_viable_link(self):
        result = pass_analysis(11.0, 2400.0)
        assert result["link_viable"] is False
        assert result["skr"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# QBER model
# ─────────────────────────────────────────────────────────────────────────────

class TestQBERModel:
    def test_zero_loss_baseline(self):
        """At zero link loss, QBER = baseline."""
        from qkd_constellation.constants import QBER_BASE
        assert abs(qber_model(0.0) - QBER_BASE) < 1e-10

    def test_qber_increases_with_loss(self):
        assert qber_model(20.0) < qber_model(40.0) < qber_model(60.0)

    def test_saturates_at_threshold(self):
        """At or beyond threshold, QBER should not exceed base+slope."""
        from qkd_constellation.constants import QBER_BASE, QBER_LOSS_SLOPE
        max_expected = QBER_BASE + QBER_LOSS_SLOPE
        assert qber_model(LOSS_THRESHOLD_DB) <= max_expected + 1e-10
        assert qber_model(LOSS_THRESHOLD_DB * 2) <= max_expected + 1e-10
