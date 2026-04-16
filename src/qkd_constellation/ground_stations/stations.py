"""
Ground station registry for the India QKD network.

Stations were selected based on:
  - Low atmospheric turbulence / high clear-sky fraction
  - Existing optical observatory infrastructure
  - Geographic spread to maximise satellite revisit diversity
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    altitude_m: float = 0.0
    description: str = ""

    def __repr__(self) -> str:
        return f"GroundStation({self.name!r}, {self.lat_deg:.2f}°N, {self.lon_deg:.2f}°E)"


# ──────────────────────────────────────────────────────────────────────────────
# Default India QKD network
# ──────────────────────────────────────────────────────────────────────────────

INDIA_GROUND_STATIONS: List[GroundStation] = [
    GroundStation(
        name="Hanle",
        lat_deg=32.78,
        lon_deg=78.96,
        altitude_m=4500,
        description="Indian Astronomical Observatory, Ladakh — high altitude, ~270 clear nights/yr",
    ),
    GroundStation(
        name="Dehradun",
        lat_deg=30.30,
        lon_deg=78.00,
        altitude_m=640,
        description="Uttarakhand capital — ISRO-SAC associated station",
    ),
    GroundStation(
        name="MtAbu",
        lat_deg=24.60,
        lon_deg=72.70,
        altitude_m=1220,
        description="Mount Abu Observatory, Rajasthan — PRL optical site",
    ),
    GroundStation(
        name="Shillong",
        lat_deg=25.60,
        lon_deg=91.80,
        altitude_m=1500,
        description="NEHU campus, Meghalaya — North-east link node",
    ),
    GroundStation(
        name="Kodaikanal",
        lat_deg=10.20,
        lon_deg=77.40,
        altitude_m=2343,
        description="Kodaikanal Solar Observatory, Tamil Nadu — southern anchor",
    ),
]


def get_station_dict() -> Dict[str, GroundStation]:
    """Return name → GroundStation mapping for default Indian network."""
    return {gs.name: gs for gs in INDIA_GROUND_STATIONS}


def from_config(entries) -> List[GroundStation]:
    """Build GroundStation list from config dataclass entries."""
    return [
        GroundStation(
            name=e.name,
            lat_deg=e.lat_deg,
            lon_deg=e.lon_deg,
        )
        for e in entries
    ]
