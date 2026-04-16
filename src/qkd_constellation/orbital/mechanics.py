"""
Orbital mechanics for satellite constellation simulation.

Implements a Keplerian + J2 secular drift propagator.
No external orbital-dynamics library is required — everything is
derived from first principles (Bate, Mueller & White 1971; Montenbruck 2000).

Coordinate frames:
  ECI  — Earth-Centred Inertial (X toward vernal equinox, Z toward north pole)
  ECEF — Earth-Centred Earth-Fixed (co-rotates with Earth)
  SEZ  — South-East-Zenith topocentric frame at a ground station
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from qkd_constellation.constants import (
    R_EARTH_KM,
    MU_EARTH,
    J2,
    EARTH_ROT_RAD_PER_S,
    EARTH_FLATTENING,
)


# ──────────────────────────────────────────────────────────────────────────────
# Keplerian orbital elements container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OrbitalElements:
    """Classical Keplerian elements (WGS-84 / TEME-compatible)."""
    semi_major_axis_km: float    # a  [km]
    eccentricity: float          # e
    inclination_deg: float       # i  [°]
    raan_deg: float              # Ω  [°] — right ascension of ascending node
    arg_perigee_deg: float       # ω  [°] — argument of perigee
    mean_anomaly_deg: float      # M₀ [°] — mean anomaly at epoch
    epoch_s: float = 0.0         # epoch in seconds from simulation start

    # Derived at construction
    mean_motion_rad_s: float = 0.0

    def __post_init__(self):
        self.mean_motion_rad_s = math.sqrt(MU_EARTH / self.semi_major_axis_km**3)

    @property
    def period_s(self) -> float:
        return 2 * math.pi / self.mean_motion_rad_s

    @property
    def period_min(self) -> float:
        return self.period_s / 60.0

    # J2 secular RAAN precession rate [rad/s]
    def raan_dot_rad_s(self) -> float:
        n = self.mean_motion_rad_s
        p = self.semi_major_axis_km * (1 - self.eccentricity**2)
        cos_i = math.cos(math.radians(self.inclination_deg))
        return -1.5 * J2 * (R_EARTH_KM / p)**2 * n * cos_i

    # J2 secular argument-of-perigee drift [rad/s]
    def omega_dot_rad_s(self) -> float:
        n = self.mean_motion_rad_s
        p = self.semi_major_axis_km * (1 - self.eccentricity**2)
        sin_i = math.sin(math.radians(self.inclination_deg))
        return 0.75 * J2 * (R_EARTH_KM / p)**2 * n * (5 * sin_i**2 - 4) / sin_i if sin_i != 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Kepler equation solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_kepler(M_rad: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """
    Newton-Raphson solution of Kepler's equation:  E - e·sin(E) = M

    Returns eccentric anomaly E in radians.
    """
    E = M_rad if e < 0.8 else math.pi
    for _ in range(max_iter):
        dE = (M_rad - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


# ──────────────────────────────────────────────────────────────────────────────
# ECI position from orbital elements
# ──────────────────────────────────────────────────────────────────────────────

def eci_position_km(elements: OrbitalElements, t_s: float) -> np.ndarray:
    """
    Propagate orbital elements to ECI position vector [km] at elapsed time t_s [s].

    Applies J2 secular drift to RAAN and argument of perigee.
    """
    dt = t_s - elements.epoch_s

    # J2-corrected angles
    raan = math.radians(elements.raan_deg) + elements.raan_dot_rad_s() * dt
    omega = math.radians(elements.arg_perigee_deg) + elements.omega_dot_rad_s() * dt

    # Propagate mean anomaly
    M = math.radians(elements.mean_anomaly_deg) + elements.mean_motion_rad_s * dt
    M = M % (2 * math.pi)

    E = solve_kepler(M, elements.eccentricity)

    # True anomaly
    sqrt_term = math.sqrt((1 + elements.eccentricity) / (1 - elements.eccentricity))
    nu = 2.0 * math.atan2(sqrt_term * math.sin(E / 2), math.cos(E / 2))

    # Distance from centre
    r = elements.semi_major_axis_km * (1 - elements.eccentricity * math.cos(E))

    # Position in perifocal (PQW) frame
    p_pqw = r * np.array([math.cos(nu), math.sin(nu), 0.0])

    # Rotation matrices: ω, i, Ω
    i = math.radians(elements.inclination_deg)

    R3_raan = _Rz(-raan)
    R1_i    = _Rx(-i)
    R3_omega = _Rz(-omega)

    Q_pqw_to_eci = R3_raan @ R1_i @ R3_omega

    return Q_pqw_to_eci @ p_pqw


# ──────────────────────────────────────────────────────────────────────────────
# ECEF / topocentric conversions
# ──────────────────────────────────────────────────────────────────────────────

def eci_to_ecef(r_eci: np.ndarray, t_s: float, theta0_rad: float = 0.0) -> np.ndarray:
    """
    Rotate ECI → ECEF using Greenwich sidereal angle.

    theta0_rad: GMST at simulation epoch (default 0, relative).
    """
    theta = theta0_rad + EARTH_ROT_RAD_PER_S * t_s
    Rz = _Rz(theta)
    return Rz @ r_eci


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """
    Convert WGS-84 geodetic coordinates to ECEF [km].
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    f = EARTH_FLATTENING
    e2 = 2 * f - f**2
    N = R_EARTH_KM / math.sqrt(1 - e2 * math.sin(lat)**2)  # prime vertical radius

    x = (N + alt_km) * math.cos(lat) * math.cos(lon)
    y = (N + alt_km) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + alt_km) * math.sin(lat)
    return np.array([x, y, z])


def ecef_to_sez(r_sat_ecef: np.ndarray, r_gs_ecef: np.ndarray,
                lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Convert relative ECEF vector (satellite − ground station) to SEZ topocentric frame.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    dr = r_sat_ecef - r_gs_ecef

    # SEZ unit vectors in ECEF
    s = np.array([
        math.sin(lat) * math.cos(lon),
        math.sin(lat) * math.sin(lon),
        -math.cos(lat)
    ])
    e = np.array([-math.sin(lon), math.cos(lon), 0.0])
    z = np.array([
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat)
    ])

    return np.array([np.dot(dr, s), np.dot(dr, e), np.dot(dr, z)])


def elevation_and_range(r_sez: np.ndarray) -> Tuple[float, float]:
    """
    Compute elevation [°] and slant range [km] from SEZ vector.
    """
    rng = float(np.linalg.norm(r_sez))
    if rng < 1e-6:
        return 90.0, rng
    elev_rad = math.asin(r_sez[2] / rng)
    return math.degrees(elev_rad), rng


def get_elevation_and_range(
    elements: OrbitalElements,
    t_s: float,
    lat_deg: float,
    lon_deg: float,
    theta0_rad: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute topocentric elevation [°] and slant range [km] for a satellite
    as seen from a ground station at time t_s seconds from epoch.
    """
    r_eci = eci_position_km(elements, t_s)
    r_ecef_sat = eci_to_ecef(r_eci, t_s, theta0_rad)
    r_ecef_gs = geodetic_to_ecef(lat_deg, lon_deg)
    r_sez = ecef_to_sez(r_ecef_sat, r_ecef_gs, lat_deg, lon_deg)
    return elevation_and_range(r_sez)


# ──────────────────────────────────────────────────────────────────────────────
# Air-mass model
# ──────────────────────────────────────────────────────────────────────────────

def air_mass_factor(elevation_deg: float) -> float:
    """
    Chapman air-mass factor for a spherical atmosphere.
    Approximated as 1/sin(elev) for elevation > 5°.
    """
    if elevation_deg <= 0:
        return float("inf")
    return 1.0 / math.sin(math.radians(elevation_deg))


# ──────────────────────────────────────────────────────────────────────────────
# Rotation matrix helpers
# ──────────────────────────────────────────────────────────────────────────────

def _Rx(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _Rz(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
