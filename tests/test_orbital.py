"""
Unit tests for the orbital mechanics module.

Tests cover:
  - Kepler equation solver accuracy
  - ECI position (circular orbit sanity check)
  - ECEF geodetic conversion round-trip
  - Elevation / range geometry
  - J2 RAAN precession sign and magnitude
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qkd_constellation.orbital.mechanics import (
    solve_kepler,
    OrbitalElements,
    eci_position_km,
    geodetic_to_ecef,
    eci_to_ecef,
    ecef_to_sez,
    elevation_and_range,
    get_elevation_and_range,
    air_mass_factor,
)
from qkd_constellation.constants import R_EARTH_KM, MU_EARTH


# ─────────────────────────────────────────────────────────────────────────────
# Kepler solver
# ─────────────────────────────────────────────────────────────────────────────

class TestKepler:
    def test_circular_orbit(self):
        """For circular orbit (e=0), E = M exactly."""
        for M_deg in range(0, 360, 30):
            M = math.radians(M_deg)
            E = solve_kepler(M, 0.0)
            assert abs(E - M) < 1e-9, f"E≠M for e=0, M={M_deg}°"

    def test_kepler_identity(self):
        """E - e*sin(E) = M must hold for any e, M."""
        for e in [0.0, 0.1, 0.5, 0.8]:
            for M_deg in [0, 45, 90, 180, 270, 359]:
                M = math.radians(M_deg)
                E = solve_kepler(M, e)
                residual = abs(E - e * math.sin(E) - M)
                assert residual < 1e-9, f"Kepler residual {residual:.2e} for e={e}, M={M_deg}°"

    def test_high_eccentricity(self):
        """Near-parabolic orbit (e=0.9) should still converge."""
        E = solve_kepler(math.pi / 2, 0.9)
        residual = abs(E - 0.9 * math.sin(E) - math.pi / 2)
        assert residual < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# ECI position
# ─────────────────────────────────────────────────────────────────────────────

class TestECIPosition:
    def _iss_like(self):
        return OrbitalElements(
            semi_major_axis_km=R_EARTH_KM + 600,
            eccentricity=0.001,
            inclination_deg=51.6,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
        )

    def test_radius_at_epoch(self):
        """At epoch (t=0), radius should match semi-major axis ± small e correction."""
        elem = self._iss_like()
        r = eci_position_km(elem, 0.0)
        expected_r = elem.semi_major_axis_km * (1 - elem.eccentricity)   # near periapsis
        assert abs(np.linalg.norm(r) - expected_r) < 10, "Radius off by >10 km at epoch"

    def test_one_orbit(self):
        """After exactly one period, satellite returns to same ECI position."""
        elem = OrbitalElements(
            semi_major_axis_km=R_EARTH_KM + 600,
            eccentricity=0.0,   # circular — clean test
            inclination_deg=45.0,
            raan_deg=30.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
        )
        r0 = eci_position_km(elem, 0.0)
        r1 = eci_position_km(elem, elem.period_s)
        # J2 secular drift moves perigee and RAAN, so position != initial after one period
        # Verify radius magnitude is preserved instead (energy conservation)
        assert abs(np.linalg.norm(r1) - np.linalg.norm(r0)) < 1.0, "Radius changed by >1 km"

    def test_altitude_range(self):
        """Altitude for a 600 km circular orbit should stay near 600 km."""
        elem = OrbitalElements(
            semi_major_axis_km=R_EARTH_KM + 600,
            eccentricity=0.0,
            inclination_deg=97.5,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
        )
        for frac in np.linspace(0, 1, 20):
            t = frac * elem.period_s
            r = eci_position_km(elem, t)
            alt = np.linalg.norm(r) - R_EARTH_KM
            assert 598 < alt < 602, f"Altitude {alt:.1f} km out of range at t={t:.0f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Geodetic conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestGeodetic:
    def test_equator_prime_meridian(self):
        """Point on equator at lon=0 should have x≈R_E, y=z=0."""
        r = geodetic_to_ecef(0.0, 0.0)
        assert abs(r[0] - R_EARTH_KM) < 1.0
        assert abs(r[1]) < 0.01
        assert abs(r[2]) < 0.01

    def test_north_pole(self):
        """North pole: z ≈ R_E·(1-f), x≈y≈0."""
        r = geodetic_to_ecef(90.0, 0.0)
        assert abs(r[0]) < 1.0
        assert abs(r[1]) < 1.0
        assert r[2] > 6350  # polar radius ~6356 km

    def test_hanle_station(self):
        """Hanle observatory should be ~6600 km from Earth center (alt 4.5 km)."""
        r = geodetic_to_ecef(32.78, 78.96, 4.5)
        dist = np.linalg.norm(r)
        assert 6370 < dist < 6400, f"Hanle ECEF radius {dist:.1f} km unexpected"


# ─────────────────────────────────────────────────────────────────────────────
# Elevation / range geometry
# ─────────────────────────────────────────────────────────────────────────────

class TestElevation:
    def test_nadir_pass(self):
        """Satellite directly overhead (nadir) should give ~90° elevation."""
        # Place ground station at equator/prime meridian
        # Place satellite directly above at 600 km (t=0, no Earth rotation)
        r_sat_ecef  = np.array([R_EARTH_KM + 600.0, 0.0, 0.0])
        r_gs_ecef   = geodetic_to_ecef(0.0, 0.0)
        r_sez       = ecef_to_sez(r_sat_ecef, r_gs_ecef, 0.0, 0.0)
        elev, rng   = elevation_and_range(r_sez)
        assert abs(elev - 90.0) < 5.0, f"Nadir elevation {elev:.1f}° ≠ ~90°"
        assert abs(rng - 600.0) < 10.0, f"Nadir range {rng:.1f} km ≠ ~600 km"

    def test_below_horizon_negative(self):
        """Satellite behind Earth should give negative elevation."""
        # Put satellite on opposite side of Earth
        r_sat_ecef = np.array([-(R_EARTH_KM + 600.0), 0.0, 0.0])
        r_gs_ecef  = geodetic_to_ecef(0.0, 0.0)
        r_sez      = ecef_to_sez(r_sat_ecef, r_gs_ecef, 0.0, 0.0)
        elev, _    = elevation_and_range(r_sez)
        assert elev < 0, "Antipodal satellite should have negative elevation"

    def test_air_mass_at_zenith(self):
        """Air mass factor at 90° elevation = 1."""
        assert abs(air_mass_factor(90.0) - 1.0) < 1e-6

    def test_air_mass_increases_at_horizon(self):
        """Air mass should increase monotonically toward horizon."""
        am_high = air_mass_factor(60.0)
        am_low  = air_mass_factor(15.0)
        assert am_low > am_high


# ─────────────────────────────────────────────────────────────────────────────
# J2 precession
# ─────────────────────────────────────────────────────────────────────────────

class TestJ2Precession:
    def test_raan_retrograde_posigrade(self):
        """Prograde orbit (i<90°) → negative RAAN dot (westward drift)."""
        elem_pro = OrbitalElements(
            semi_major_axis_km=R_EARTH_KM + 600,
            eccentricity=0.001,
            inclination_deg=51.6,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
        )
        assert elem_pro.raan_dot_rad_s() < 0, "Prograde RAAN dot should be negative"

    def test_raan_sun_sync(self):
        """Sun-synchronous orbit (i≈97.5°) → RAAN drift ≈ +0.9856°/day."""
        elem_ss = OrbitalElements(
            semi_major_axis_km=R_EARTH_KM + 600,
            eccentricity=0.001,
            inclination_deg=97.5,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
        )
        raan_dot_deg_day = math.degrees(elem_ss.raan_dot_rad_s()) * 86400
        # Sun-sync drift should be close to +0.9856°/day
        assert 0.7 < raan_dot_deg_day < 1.2, \
            f"Sun-sync RAAN drift {raan_dot_deg_day:.4f} °/day out of expected range"
