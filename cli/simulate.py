#!/usr/bin/env python3
"""
QKD Satellite Constellation — Command-Line Interface

Usage examples
--------------
# Run with defaults (3 planes × 4 sats, 600 km, India ground stations):
    python cli/simulate.py

# Custom configuration:
    python cli/simulate.py --planes 3 --sats-per-plane 3 --altitude 550 \
                           --inclination 98 --duration 480 --step 2

# Load from YAML config file:
    python cli/simulate.py --config config/config.yaml

# Skip plot generation (data-only run):
    python cli/simulate.py --no-plots

# Quiet mode (suppress per-pass messages):
    python cli/simulate.py --quiet
"""
from __future__ import annotations

import argparse
import sys
import os
import textwrap
from pathlib import Path

# Ensure package is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qkd_constellation.config import (
    AppConfig, ConstellationConfig, QKDConfig,
    SimulationConfig, load_config,
)
from qkd_constellation.simulation.runner import run_simulation
from qkd_constellation.visualization.plots import generate_all_plots
from qkd_constellation.constellation.generator import constellation_summary, generate_constellation


# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────

BANNER = textwrap.dedent("""
╔══════════════════════════════════════════════════════════════╗
║   QKD Satellite Constellation Simulator                      ║
║   Decoy-State BB84 Protocol · Walker-Delta Constellation     ║
║   India Ground Network · Aperture-Based η Channel Model      ║
╚══════════════════════════════════════════════════════════════╝
""")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qkd-sim",
        description="Simulate a satellite QKD constellation over Indian ground stations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file
    p.add_argument("--config", metavar="PATH",
                   help="YAML configuration file (overrides CLI flags)")

    # Constellation
    cg = p.add_argument_group("Constellation")
    cg.add_argument("--planes",          type=int,   default=3,    help="Number of orbital planes [3]")
    cg.add_argument("--sats-per-plane",  type=int,   default=4,    help="Satellites per plane [4]")
    cg.add_argument("--altitude",        type=float, default=600,  help="Orbital altitude km [600]")
    cg.add_argument("--inclination",     type=float, default=97.5, help="Inclination degrees [97.5]")
    cg.add_argument("--eccentricity",    type=float, default=2e-4, help="Eccentricity [0.0002]")

    # QKD
    qg = p.add_argument_group("QKD Protocol")
    qg.add_argument("--mu",              type=float, default=0.5,  help="Signal mean photon number [0.5]")
    qg.add_argument("--aperture",        type=float, default=0.5,  help="Rx aperture diameter m [0.5]")
    qg.add_argument("--divergence",      type=float, default=10e-6,help="Beam divergence rad [10e-6]")
    qg.add_argument("--loss-threshold",  type=float, default=60.0, help="Link loss threshold dB [60]")

    # Simulation
    sg = p.add_argument_group("Simulation")
    sg.add_argument("--duration",        type=int,   default=1440, help="Duration minutes [1440]")
    sg.add_argument("--step",            type=int,   default=5,    help="Time step minutes [5]")
    sg.add_argument("--output-dir",      default="outputs",        help="Output directory [outputs]")
    sg.add_argument("--dpi",             type=int,   default=300,  help="Plot resolution DPI [300]")
    sg.add_argument("--no-plots",        action="store_true",      help="Skip plot generation")
    sg.add_argument("--quiet",           action="store_true",      help="Suppress verbose output")
    sg.add_argument("--export-csv",      action="store_true", default=True,
                                                                    help="Export per-station CSV files")

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Build config from args
# ──────────────────────────────────────────────────────────────────────────────

def config_from_args(args) -> AppConfig:
    if args.config:
        cfg = load_config(args.config)
        print(f"[CONFIG] Loaded from {args.config}")
    else:
        cfg = AppConfig(
            constellation=ConstellationConfig(
                num_planes=args.planes,
                sats_per_plane=args.sats_per_plane,
                altitude_km=args.altitude,
                inclination_deg=args.inclination,
                eccentricity=args.eccentricity,
            ),
            qkd=QKDConfig(
                mu_signal=args.mu,
                aperture_rx_m=args.aperture,
                beam_divergence_rad=args.divergence,
                loss_threshold_db=args.loss_threshold,
            ),
            simulation=SimulationConfig(
                duration_min=args.duration,
                step_min=args.step,
                output_dir=args.output_dir,
                save_csv=args.export_csv,
                save_plots=not args.no_plots,
                dpi=args.dpi,
            ),
        )
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = config_from_args(args)

    # Print constellation summary before simulation
    sats = generate_constellation(
        num_planes=cfg.constellation.num_planes,
        sats_per_plane=cfg.constellation.sats_per_plane,
        altitude_km=cfg.constellation.altitude_km,
        inclination_deg=cfg.constellation.inclination_deg,
    )
    summary = constellation_summary(sats)
    print("─" * 60)
    print("Constellation Parameters:")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print("─" * 60)
    print(f"Simulation: {cfg.simulation.duration_min} min  "
          f"(step={cfg.simulation.step_min} min, "
          f"{cfg.simulation.duration_min // cfg.simulation.step_min} time steps)")
    print(f"Ground stations: {[gs.name for gs in cfg.ground_stations]}")
    print("─" * 60)

    # Run simulation
    result = run_simulation(cfg, verbose=not args.quiet)

    # Summary table
    print("\n" + "═" * 70)
    print(f"{'Station':<14} {'TotalSKR':>12} {'SecureLinks':>12} "
          f"{'AvgQBER%':>10} {'Coverage%':>10} {'MaxElev°':>10}")
    print("─" * 70)
    for gs_name, s in result.station_summaries.items():
        print(
            f"{gs_name:<14} {s.total_skr:>12.6f} {s.secure_link_count:>12} "
            f"{s.avg_qber*100:>10.3f} {s.coverage_fraction*100:>10.2f} "
            f"{s.max_elevation_deg:>10.2f}"
        )
    print("═" * 70)
    print(f"\nTotal simulation time: {result.elapsed_s:.2f} s")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating publication-quality plots …")
        fig_dir = Path(cfg.simulation.output_dir) / "figures"
        saved   = generate_all_plots(result, str(fig_dir), cfg.simulation.dpi)
        print(f"✅  {len(saved)} figures saved to {fig_dir}/")

    print("\nDone. Output written to:", cfg.simulation.output_dir)


if __name__ == "__main__":
    main()
