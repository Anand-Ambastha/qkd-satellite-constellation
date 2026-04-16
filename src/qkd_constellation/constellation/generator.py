"""
Walker-Delta satellite constellation generator.

Produces orbital elements for a T/P/F Walker constellation (T satellites,
P orbital planes, F relative phasing).  Also exports synthetic TLE-format
strings for interoperability with other tools.

Reference: Walker, J.G. (1984) Satellite Constellations, JBIS 37, 559.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from qkd_constellation.constants import R_EARTH_KM, MU_EARTH, J2
from qkd_constellation.orbital.mechanics import OrbitalElements


# ──────────────────────────────────────────────────────────────────────────────
# Satellite record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Satellite:
    name: str
    plane: int
    seat: int
    elements: OrbitalElements

    def __repr__(self) -> str:
        return (f"Satellite({self.name!r}, plane={self.plane}, seat={self.seat}, "
                f"alt={self.elements.semi_major_axis_km - R_EARTH_KM:.0f} km, "
                f"i={self.elements.inclination_deg:.1f}°)")


# ──────────────────────────────────────────────────────────────────────────────
# Walker-Delta generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_constellation(
    num_planes: int = 3,
    sats_per_plane: int = 4,
    altitude_km: float = 600.0,
    inclination_deg: float = 97.5,
    eccentricity: float = 0.0002,
    arg_perigee_deg: float = 90.0,
    phasing_factor: int = 1,
    prefix: str = "ISRO-QKD",
) -> List[Satellite]:
    """
    Generate a Walker-Delta T/P/F constellation.

    Parameters
    ----------
    num_planes      : P — number of orbital planes
    sats_per_plane  : T/P — satellites per plane
    altitude_km     : circular orbit altitude above WGS-84 equatorial radius
    inclination_deg : orbital inclination (97.5° ≈ sun-synchronous at 600 km)
    eccentricity    : near-zero for Walker
    arg_perigee_deg : argument of perigee (90° → passes near poles at apogee)
    phasing_factor  : F — Walker phasing parameter (0 ≤ F < P)
    prefix          : satellite name prefix

    Returns
    -------
    List[Satellite]
    """
    sma_km = R_EARTH_KM + altitude_km
    n = math.sqrt(MU_EARTH / sma_km**3)              # mean motion [rad/s]
    total = num_planes * sats_per_plane

    raan_spacing_deg    = 360.0 / num_planes          # Δ Ω between planes
    ma_spacing_deg      = 360.0 / sats_per_plane      # Δ M within plane
    # Walker phasing: Δ M offset between adjacent planes
    phasing_deg         = phasing_factor * 360.0 / total

    satellites: List[Satellite] = []
    for p in range(num_planes):
        for s in range(sats_per_plane):
            sat_num = p * sats_per_plane + s + 1
            raan    = (p * raan_spacing_deg) % 360.0
            ma      = (s * ma_spacing_deg + p * phasing_deg) % 360.0

            elem = OrbitalElements(
                semi_major_axis_km=sma_km,
                eccentricity=eccentricity,
                inclination_deg=inclination_deg,
                raan_deg=raan,
                arg_perigee_deg=arg_perigee_deg,
                mean_anomaly_deg=ma,
                epoch_s=0.0,
            )
            satellites.append(
                Satellite(
                    name=f"{prefix}-{sat_num:02d}",
                    plane=p,
                    seat=s,
                    elements=elem,
                )
            )

    return satellites


# ──────────────────────────────────────────────────────────────────────────────
# TLE export (synthetic, for archival / external tool use)
# ──────────────────────────────────────────────────────────────────────────────

def _tle_checksum(line: str) -> int:
    total = 0
    for c in line[:-1]:
        if c.isdigit():
            total += int(c)
        elif c == "-":
            total += 1
    return total % 10


def satellite_to_tle(sat: Satellite, catalog_number: int = 99000) -> str:
    """Return a two-line TLE string for the satellite (synthetic epoch)."""
    e = sat.elements
    n_revs = e.mean_motion_rad_s * 86400 / (2 * math.pi)   # rev/day

    line1 = (
        f"1 {catalog_number + sat.plane * 100 + sat.seat:05d}U "
        f"24001A   24165.00000000  .00000023  00000-0  10000-3 0  9990"
    )
    line2 = (
        f"2 {catalog_number + sat.plane * 100 + sat.seat:05d} "
        f"{e.inclination_deg:8.4f} "
        f"{e.raan_deg:8.4f} "
        f"{int(e.eccentricity * 1e7):07d} "
        f"{e.arg_perigee_deg:8.4f} "
        f"{e.mean_anomaly_deg:8.4f} "
        f"{n_revs:11.8f}    10"
    )
    return f"{sat.name}\n{line1}\n{line2}"


def export_tle_file(satellites: List[Satellite], path: str = "outputs/data/synthetic_tle.txt") -> None:
    """Write all TLE strings to a file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for sat in satellites:
            fh.write(satellite_to_tle(sat) + "\n")
    print(f"[TLE] Exported {len(satellites)} TLEs → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Constellation metrics
# ──────────────────────────────────────────────────────────────────────────────

def constellation_summary(satellites: List[Satellite]) -> dict:
    """Return a summary dict of constellation parameters."""
    e0 = satellites[0].elements
    return {
        "total_satellites": len(satellites),
        "num_planes": max(s.plane for s in satellites) + 1,
        "sats_per_plane": max(s.seat for s in satellites) + 1,
        "altitude_km": round(e0.semi_major_axis_km - R_EARTH_KM, 1),
        "inclination_deg": e0.inclination_deg,
        "period_min": round(e0.period_min, 2),
        "mean_motion_rev_per_day": round(e0.mean_motion_rad_s * 86400 / (2 * math.pi), 6),
        "raan_dot_deg_per_day": round(
            math.degrees(e0.raan_dot_rad_s()) * 86400, 4
        ),
    }
