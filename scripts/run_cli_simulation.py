#!/usr/bin/env python3
"""
Interactive CLI Simulation — animated pass-by-pass output with summary tables.

This script provides a terminal-friendly, coloured simulation experience
suited for demos and paper reproductions.

Usage:
    python scripts/run_cli_simulation.py
    python scripts/run_cli_simulation.py --planes 3 --sats 4 --alt 600 --quiet
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from qkd_constellation.config import (
    AppConfig, ConstellationConfig, QKDConfig, SimulationConfig,
)
from qkd_constellation.constellation.generator import (
    generate_constellation, constellation_summary,
)
from qkd_constellation.ground_stations.stations import INDIA_GROUND_STATIONS
from qkd_constellation.orbital.mechanics import get_elevation_and_range
from qkd_constellation.qkd.bb84 import secure_key_rate, pass_analysis
from qkd_constellation.qkd.channel import channel_budget
from qkd_constellation.constants import MIN_ELEVATION_DEG


# ── ANSI colour helpers ────────────────────────────────────────────────────────
def _c(code: int, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c(32, t)
YELLOW = lambda t: _c(33, t)
CYAN   = lambda t: _c(36, t)
RED    = lambda t: _c(31, t)
BOLD   = lambda t: _c(1,  t)
DIM    = lambda t: _c(2,  t)


BANNER = f"""
{BOLD('╔══════════════════════════════════════════════════════════════╗')}
{BOLD('║')}  {CYAN('QKD Satellite Constellation — Interactive CLI Simulation')}  {BOLD('║')}
{BOLD('║')}  Decoy-State BB84  ·  Walker-Δ  ·  India Ground Network      {BOLD('║')}
{BOLD('╚══════════════════════════════════════════════════════════════╝')}
"""


def spinner_frames():
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    while True:
        yield frames[i % len(frames)]
        i += 1


def print_link_budget_table(elev: float, rng: float, mu: float = 0.5):
    budget = channel_budget(elev, rng)
    _, skr = secure_key_rate(elev, rng, mu=mu)
    print(f"\n  {'─'*52}")
    print(f"  {'Parameter':<28} {'Value':>22}")
    print(f"  {'─'*52}")
    for k, v in budget.items():
        fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
        print(f"  {k:<28} {fmt:>22}")
    print(f"  {'Secure Key Rate':<28} {skr:>22.8f}")
    print(f"  {'─'*52}\n")


def run_interactive_simulation(cfg: AppConfig, quiet: bool = False):
    print(BANNER)

    sats = generate_constellation(
        num_planes=cfg.constellation.num_planes,
        sats_per_plane=cfg.constellation.sats_per_plane,
        altitude_km=cfg.constellation.altitude_km,
        inclination_deg=cfg.constellation.inclination_deg,
    )
    gs_list = INDIA_GROUND_STATIONS
    info = constellation_summary(sats)

    # Constellation overview
    print(BOLD("Constellation Configuration"))
    print("  " + "─" * 45)
    for k, v in info.items():
        print(f"  {k:<34} {CYAN(str(v))}")
    print()

    cfg_s = cfg.simulation
    minutes  = np.arange(0, cfg_s.duration_min, cfg_s.step_min)
    times_s  = minutes * 60.0
    n_steps  = len(minutes)

    print(BOLD(f"Simulation Window: {cfg_s.duration_min} min  |  "
               f"Step: {cfg_s.step_min} min  |  "
               f"Total time points: {n_steps}"))
    print(BOLD(f"Satellites: {len(sats)}  |  Ground Stations: {len(gs_list)}"))
    print(BOLD(f"Total link computations: {len(sats) * len(gs_list) * n_steps:,}"))
    print()

    station_stats = {gs.name: {"total_skr": 0.0, "secure_links": 0,
                                "max_elev": -90.0, "above_threshold": 0}
                     for gs in gs_list}

    total_ops  = len(sats) * len(gs_list) * n_steps
    done       = 0
    t_start    = time.perf_counter()
    spin_gen   = spinner_frames()

    for gs in gs_list:
        print(BOLD(f"\n{'═'*60}"))
        print(BOLD(f"  Ground Station: {CYAN(gs.name)}  "
                   f"({gs.lat_deg:.2f}°N, {gs.lon_deg:.2f}°E)"))
        print(BOLD(f"{'═'*60}"))

        gs_stat  = station_stats[gs.name]
        all_elevs = []

        for sat in sats:
            visible_windows = []
            in_window = False
            window_start = None

            for i, t_s in enumerate(times_s):
                elev, rng = get_elevation_and_range(
                    sat.elements, t_s, gs.lat_deg, gs.lon_deg
                )
                qber, skr = secure_key_rate(
                    elev, rng,
                    mu=cfg.qkd.mu_signal,
                    loss_threshold_db=cfg.qkd.loss_threshold_db,
                )
                all_elevs.append(elev)
                gs_stat["max_elev"] = max(gs_stat["max_elev"], elev)

                if elev > MIN_ELEVATION_DEG:
                    gs_stat["above_threshold"] += 1

                if skr > 0:
                    gs_stat["total_skr"]    += skr
                    gs_stat["secure_links"] += 1

                if not quiet and elev > MIN_ELEVATION_DEG:
                    status = GREEN("●") if skr > 0 else YELLOW("○")
                    print(
                        f"  {status} {sat.name:<16}  "
                        f"t={minutes[i]:5.0f} min  "
                        f"elev={elev:6.2f}°  "
                        f"range={rng:7.1f} km  "
                        f"QBER={YELLOW(f'{qber*100:.2f}%')}  "
                        f"SKR={GREEN(f'{skr:.3e}') if skr > 0 else DIM('0')}"
                    )

                    # Track visibility windows
                    if not in_window:
                        window_start = minutes[i]
                        in_window = True
                else:
                    if in_window:
                        visible_windows.append((window_start, minutes[i - 1]))
                        in_window = False

                done += 1

                # Progress bar
                if done % max(1, total_ops // 100) == 0 or done == total_ops:
                    pct  = done / total_ops * 100
                    bar  = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
                    elapsed = time.perf_counter() - t_start
                    rate    = done / elapsed if elapsed > 0 else 0
                    eta_s   = (total_ops - done) / rate if rate > 0 else 0
                    print(
                        f"\r  {next(spin_gen)} [{bar}] {pct:5.1f}%  "
                        f"{done:,}/{total_ops:,}  "
                        f"ETA {eta_s:.0f}s  ",
                        end="", flush=True
                    )

            if in_window:
                visible_windows.append((window_start, minutes[-1]))

            if not quiet and visible_windows:
                win_str = ", ".join(f"[{a:.0f}–{b:.0f}]" for a, b in visible_windows)
                print(f"\n    {DIM(f'↳ {sat.name} visible windows (min): {win_str}')}")

        print()  # newline after progress bar

        # Station summary
        total_steps = len(sats) * n_steps
        coverage_pct = gs_stat["above_threshold"] / total_steps * 100
        print(f"\n  {BOLD('Station Summary:')}")
        print(f"    {'Total Cumulative SKR':<30} {CYAN(f'{gs_stat["total_skr"]:.6f}')}")
        print(f"    {'Secure Link Time Steps':<30} {CYAN(str(gs_stat['secure_links']))}")
        print(f"    {'Max Elevation':<30} {CYAN(f'{gs_stat["max_elev"]:.2f}°')}")
        print(f"    {'Coverage (> 10° fraction)':<30} {CYAN(f'{coverage_pct:.2f}%')}")

    # ── Final global summary ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"\n\n{'═'*70}")
    print(BOLD("  FINAL SUMMARY — 24h QKD Link Performance"))
    print(f"{'═'*70}")
    print(f"  {'Station':<14} {'TotalSKR':>12} {'SecureSteps':>13} {'Coverage%':>11} {'MaxElev°':>10}")
    print(f"  {'─'*62}")
    for gs_name, s in station_stats.items():
        cov = s["above_threshold"] / (len(sats) * n_steps) * 100
        print(
            f"  {gs_name:<14} "
            f"{CYAN(f'{s["total_skr"]:>12.6f}')} "
            f"{s['secure_links']:>13} "
            f"{cov:>11.2f} "
            f"{s['max_elev']:>10.2f}"
        )
    print(f"{'═'*70}")
    print(f"\n  {GREEN('✅')} Simulation completed in {BOLD(f'{elapsed:.2f} s')}")

    # ── Example link budget ────────────────────────────────────────────────
    print(f"\n{BOLD('Example Link Budget (elevation=45°, range=800 km):')}")
    print_link_budget_table(45.0, 800.0, mu=cfg.qkd.mu_signal)


def main():
    p = argparse.ArgumentParser(description="Interactive QKD CLI Simulation")
    p.add_argument("--planes",  type=int,   default=3,    help="Orbital planes [3]")
    p.add_argument("--sats",    type=int,   default=4,    help="Sats per plane [4]")
    p.add_argument("--alt",     type=float, default=600,  help="Altitude km [600]")
    p.add_argument("--inc",     type=float, default=97.5, help="Inclination deg [97.5]")
    p.add_argument("--dur",     type=int,   default=1440, help="Duration min [1440]")
    p.add_argument("--step",    type=int,   default=5,    help="Time step min [5]")
    p.add_argument("--mu",      type=float, default=0.5,  help="Signal photon number [0.5]")
    p.add_argument("--quiet",   action="store_true",      help="Suppress per-pass output")
    args = p.parse_args()

    cfg = AppConfig(
        constellation=ConstellationConfig(
            num_planes=args.planes,
            sats_per_plane=args.sats,
            altitude_km=args.alt,
            inclination_deg=args.inc,
        ),
        qkd=QKDConfig(mu_signal=args.mu),
        simulation=SimulationConfig(
            duration_min=args.dur,
            step_min=args.step,
            save_plots=False,
        ),
    )
    run_interactive_simulation(cfg, quiet=args.quiet)


if __name__ == "__main__":
    main()
