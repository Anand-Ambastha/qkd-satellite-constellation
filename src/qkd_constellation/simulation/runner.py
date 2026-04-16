"""
Simulation runner for the QKD Satellite Constellation.

Orchestrates the 24-hour pass simulation over all satellite × ground-station
pairs, computes link metrics, and returns structured DataFrames ready for
analysis and visualisation.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qkd_constellation.config import AppConfig, load_config
from qkd_constellation.constellation.generator import (
    Satellite,
    generate_constellation,
    export_tle_file,
    constellation_summary,
)
from qkd_constellation.ground_stations.stations import (
    GroundStation,
    INDIA_GROUND_STATIONS,
    from_config,
)
from qkd_constellation.orbital.mechanics import get_elevation_and_range
from qkd_constellation.qkd.bb84 import secure_key_rate
from qkd_constellation.qkd.channel import channel_budget


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LinkRecord:
    satellite: str
    ground_station: str
    time_min: float
    elevation_deg: float
    range_km: float
    qber: float
    skr: float
    link_viable: bool


@dataclass
class StationSummary:
    ground_station: str
    total_skr: float
    secure_link_count: int
    total_time_steps: int
    avg_qber: float
    max_elevation_deg: float
    coverage_fraction: float    # fraction of time with elevation > threshold


@dataclass
class SimulationResult:
    config: AppConfig
    records: List[LinkRecord] = field(default_factory=list)
    station_summaries: Dict[str, StationSummary] = field(default_factory=dict)
    constellation_info: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.records])

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.station_summaries.values():
            rows.append(vars(s))
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(
    config: Optional[AppConfig] = None,
    verbose: bool = True,
) -> SimulationResult:
    """
    Execute the full 24-hour satellite QKD simulation.

    Parameters
    ----------
    config  : AppConfig instance (defaults used if None)
    verbose : print per-pass visibility messages

    Returns
    -------
    SimulationResult
    """
    if config is None:
        config = AppConfig()

    t0 = time.perf_counter()

    # ── Build constellation ──────────────────────────────────────────────────
    cfg_c = config.constellation
    cfg_q = config.qkd
    cfg_s = config.simulation

    satellites = generate_constellation(
        num_planes=cfg_c.num_planes,
        sats_per_plane=cfg_c.sats_per_plane,
        altitude_km=cfg_c.altitude_km,
        inclination_deg=cfg_c.inclination_deg,
        eccentricity=cfg_c.eccentricity,
        arg_perigee_deg=cfg_c.arg_perigee_deg,
    )

    ground_stations = (
        from_config(config.ground_stations)
        if config.ground_stations
        else INDIA_GROUND_STATIONS
    )

    # ── Time grid ────────────────────────────────────────────────────────────
    minutes = np.arange(0, cfg_s.duration_min, cfg_s.step_min)
    times_s = minutes * 60.0

    # ── Output directories ───────────────────────────────────────────────────
    out_dir   = Path(cfg_s.output_dir)
    data_dir  = out_dir / "data"
    fig_dir   = out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Export TLEs
    export_tle_file(satellites, str(data_dir / "synthetic_tle.txt"))

    result = SimulationResult(
        config=config,
        constellation_info=constellation_summary(satellites),
    )

    # ── Main simulation loop ─────────────────────────────────────────────────
    for gs in ground_stations:
        records_gs: List[LinkRecord] = []
        csv_path = data_dir / f"qkd_results_{gs.name}.csv"

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Satellite", "GroundStation", "Time_Minutes",
                             "Elevation_deg", "Range_km", "QBER", "SKR", "Viable"])

            for sat in satellites:
                for i, t_s in enumerate(times_s):
                    elev, rng = get_elevation_and_range(
                        sat.elements, t_s, gs.lat_deg, gs.lon_deg
                    )
                    qber, skr = secure_key_rate(
                        elev, rng,
                        mu=cfg_q.mu_signal,
                        f_ec=cfg_q.f_ec,
                        loss_threshold_db=cfg_q.loss_threshold_db,
                    )
                    viable = skr > 0

                    rec = LinkRecord(
                        satellite=sat.name,
                        ground_station=gs.name,
                        time_min=float(minutes[i]),
                        elevation_deg=round(elev, 4),
                        range_km=round(rng, 3),
                        qber=round(qber, 6),
                        skr=round(skr, 8),
                        link_viable=viable,
                    )
                    records_gs.append(rec)
                    result.records.append(rec)
                    writer.writerow([
                        sat.name, gs.name, minutes[i],
                        rec.elevation_deg, rec.range_km,
                        rec.qber, rec.skr, rec.link_viable,
                    ])

                    if verbose and elev > cfg_s.min_elevation_deg:
                        print(
                            f"  [{gs.name}] {sat.name} "
                            f"t={minutes[i]:4.0f} min  "
                            f"elev={elev:6.2f}°  "
                            f"range={rng:7.1f} km  "
                            f"QBER={qber:.4f}  SKR={skr:.4e}"
                        )

        # ── Station summary ──────────────────────────────────────────────────
        viable_records = [r for r in records_gs if r.link_viable]
        all_elevs      = [r.elevation_deg for r in records_gs]
        above_threshold = [r for r in records_gs if r.elevation_deg > cfg_s.min_elevation_deg]

        summary = StationSummary(
            ground_station=gs.name,
            total_skr=sum(r.skr for r in viable_records),
            secure_link_count=len(viable_records),
            total_time_steps=len(records_gs),
            avg_qber=(
                sum(r.qber for r in viable_records) / len(viable_records)
                if viable_records else float("nan")
            ),
            max_elevation_deg=max(all_elevs) if all_elevs else 0.0,
            coverage_fraction=len(above_threshold) / len(records_gs) if records_gs else 0.0,
        )
        result.station_summaries[gs.name] = summary

        if verbose:
            print(
                f"\n[SUMMARY] {gs.name}: "
                f"TotalSKR={summary.total_skr:.6f}  "
                f"SecureLinks={summary.secure_link_count}  "
                f"AvgQBER={summary.avg_qber:.4f}  "
                f"Coverage={summary.coverage_fraction*100:.1f}%\n"
            )

    result.elapsed_s = time.perf_counter() - t0
    if verbose:
        print(f"\n✅ Simulation complete in {result.elapsed_s:.2f} s")
    return result
