"""
Configuration loader for the QKD Constellation simulation.

Supports both programmatic defaults and YAML overrides.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class ConstellationConfig:
    num_planes: int = 3
    sats_per_plane: int = 4
    altitude_km: float = 600.0
    inclination_deg: float = 97.5
    eccentricity: float = 0.0002
    arg_perigee_deg: float = 90.0

    @property
    def total_satellites(self) -> int:
        return self.num_planes * self.sats_per_plane


@dataclass
class GroundStationEntry:
    name: str
    lat_deg: float
    lon_deg: float


@dataclass
class QKDConfig:
    mu_signal: float = 0.5
    mu_decoy: float = 0.1
    f_ec: float = 1.16
    qber_base: float = 0.02
    qber_loss_slope: float = 0.08
    aperture_rx_m: float = 0.50
    beam_divergence_rad: float = 10e-6
    base_loss_db: float = 5.0
    loss_threshold_db: float = 60.0


@dataclass
class SimulationConfig:
    duration_min: int = 1440
    step_min: int = 5
    min_elevation_deg: float = 10.0
    output_dir: str = "outputs"
    save_csv: bool = True
    save_plots: bool = True
    dpi: int = 300


@dataclass
class AppConfig:
    constellation: ConstellationConfig = field(default_factory=ConstellationConfig)
    qkd: QKDConfig = field(default_factory=QKDConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    ground_stations: List[GroundStationEntry] = field(default_factory=list)

    def __post_init__(self):
        if not self.ground_stations:
            self.ground_stations = _default_ground_stations()


def _default_ground_stations() -> List[GroundStationEntry]:
    return [
        GroundStationEntry("Hanle",      32.78, 78.96),
        GroundStationEntry("Dehradun",   30.30, 78.00),
        GroundStationEntry("MtAbu",      24.60, 72.70),
        GroundStationEntry("Shillong",   25.60, 91.80),
        GroundStationEntry("Kodaikanal", 10.20, 77.40),
    ]


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file or return defaults."""
    cfg = AppConfig()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    if not _HAS_YAML:
        print(f"[WARNING] PyYAML not available; using defaults (config ignored: {path})")
        return cfg

    with open(p) as fh:
        raw = yaml.safe_load(fh) or {}

    if "constellation" in raw:
        c = raw["constellation"]
        cfg.constellation = ConstellationConfig(**c)

    if "qkd" in raw:
        cfg.qkd = QKDConfig(**raw["qkd"])

    if "simulation" in raw:
        cfg.simulation = SimulationConfig(**raw["simulation"])

    if "ground_stations" in raw:
        cfg.ground_stations = [
            GroundStationEntry(**gs) for gs in raw["ground_stations"]
        ]

    return cfg
