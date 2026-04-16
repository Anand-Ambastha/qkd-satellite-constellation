#!/usr/bin/env python3
"""
Generate all publication-quality figures from a completed simulation.

Usage:
    python scripts/generate_plots.py               # run + plot (default config)
    python scripts/generate_plots.py --dpi 600     # high-res for paper submission
    python scripts/generate_plots.py --csv-dir outputs/data  # from existing CSVs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qkd_constellation.config import AppConfig, SimulationConfig
from qkd_constellation.simulation.runner import run_simulation
from qkd_constellation.visualization.plots import generate_all_plots


def main():
    p = argparse.ArgumentParser(description="Generate QKD simulation plots")
    p.add_argument("--dpi",     type=int, default=300,   help="Plot DPI [300]")
    p.add_argument("--out",     default="outputs/figures", help="Figure output dir")
    p.add_argument("--duration", type=int, default=1440, help="Simulation duration min")
    p.add_argument("--step",    type=int, default=5,     help="Time step min")
    args = p.parse_args()

    print("🔭  Running simulation …")
    cfg = AppConfig(
        simulation=SimulationConfig(
            duration_min=args.duration,
            step_min=args.step,
            output_dir="outputs",
            save_plots=False,
        )
    )
    result = run_simulation(cfg, verbose=False)
    print(f"   Done — {len(result.records):,} link records")

    print(f"\n📊  Generating figures (DPI={args.dpi}) …")
    saved = generate_all_plots(result, args.out, dpi=args.dpi)

    print(f"\n✅  {len(saved)} figures saved to: {args.out}/")
    for p_str in saved:
        print(f"   {p_str}")


if __name__ == "__main__":
    main()
