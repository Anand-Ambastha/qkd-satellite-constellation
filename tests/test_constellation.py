"""
Unit tests for the constellation generator module.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qkd_constellation.constellation.generator import (
    generate_constellation,
    constellation_summary,
    satellite_to_tle,
    Satellite,
)
from qkd_constellation.constants import R_EARTH_KM


class TestConstellationGenerator:
    def _default(self):
        return generate_constellation(num_planes=3, sats_per_plane=4)

    def test_total_satellite_count(self):
        sats = self._default()
        assert len(sats) == 12

    def test_satellite_names_unique(self):
        sats = self._default()
        names = [s.name for s in sats]
        assert len(names) == len(set(names))

    def test_altitude_correct(self):
        sats = generate_constellation(altitude_km=600)
        for s in sats:
            alt = s.elements.semi_major_axis_km - R_EARTH_KM
            assert abs(alt - 600.0) < 0.1

    def test_raan_spacing(self):
        """Planes must be evenly spaced in RAAN."""
        sats = generate_constellation(num_planes=3, sats_per_plane=1)
        raans = sorted(s.elements.raan_deg for s in sats)
        diffs = [raans[i+1] - raans[i] for i in range(len(raans)-1)]
        for d in diffs:
            assert abs(d - 120.0) < 0.01, f"RAAN spacing {d}° ≠ 120°"

    def test_inclination_preserved(self):
        sats = generate_constellation(inclination_deg=51.6)
        for s in sats:
            assert abs(s.elements.inclination_deg - 51.6) < 0.01

    def test_mean_anomaly_range(self):
        """Mean anomalies must be in [0, 360)."""
        sats = self._default()
        for s in sats:
            assert 0 <= s.elements.mean_anomaly_deg < 360.0

    def test_custom_size(self):
        sats = generate_constellation(num_planes=4, sats_per_plane=6)
        assert len(sats) == 24

    def test_period_positive(self):
        sats = self._default()
        for s in sats:
            assert s.elements.period_min > 0


class TestConstellationSummary:
    def test_summary_keys(self):
        sats = generate_constellation()
        s = constellation_summary(sats)
        assert "total_satellites" in s
        assert "period_min" in s
        assert "raan_dot_deg_per_day" in s

    def test_summary_values(self):
        sats = generate_constellation(num_planes=3, sats_per_plane=4, altitude_km=600)
        s = constellation_summary(sats)
        assert s["total_satellites"] == 12
        assert s["altitude_km"] == 600.0
        # At 600 km, orbital period is ~96-98 min
        assert 94 < s["period_min"] < 100


class TestTLEExport:
    def test_tle_format(self):
        sats = generate_constellation(num_planes=2, sats_per_plane=2)
        tle_str = satellite_to_tle(sats[0])
        lines = tle_str.strip().split("\n")
        assert len(lines) == 3
        assert lines[1].startswith("1 ")
        assert lines[2].startswith("2 ")
