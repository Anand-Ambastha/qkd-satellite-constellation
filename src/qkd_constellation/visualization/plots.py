"""
Visualisation module for the QKD Satellite Constellation simulation.

All plots are publication-ready (300 dpi, LaTeX-style labels, tight layout).
Functions accept a SimulationResult and write PNG files to the output directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for server/CLI use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from qkd_constellation.simulation.runner import SimulationResult
from qkd_constellation.qkd.channel import geometric_efficiency
from qkd_constellation.constants import MIN_ELEVATION_DEG, LOSS_THRESHOLD_DB

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "figure.dpi":       100,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "lines.linewidth":  1.4,
})

PALETTE = plt.cm.tab10.colors   # consistent satellite colouring


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Elevation vs Time  (per ground station)
# ─────────────────────────────────────────────────────────────────────────────

def plot_elevation_vs_time(
    result: SimulationResult,
    output_dir: str = "outputs/figures",
    dpi: int = 300,
) -> List[str]:
    """
    Satellite elevation vs simulation time for each ground station.
    Satellite passes above QKD threshold are highlighted.
    """
    df = result.to_dataframe()
    stations = df["ground_station"].unique()
    saved = []

    for gs_name in stations:
        df_gs = df[df["ground_station"] == gs_name]
        sats  = df_gs["satellite"].unique()

        fig, ax = plt.subplots(figsize=(12, 5))
        for idx, sat in enumerate(sats):
            df_sat = df_gs[df_gs["satellite"] == sat].sort_values("time_min")
            t   = df_sat["time_min"].values
            elev = df_sat["elevation_deg"].values
            max_e = elev.max()
            col  = PALETTE[idx % len(PALETTE)]
            ax.plot(t, elev, color=col, label=f"{sat} (max {max_e:.1f}°)", alpha=0.85)

            # Shade viable windows
            viable = df_sat["link_viable"].values
            _shade_viable(ax, t, elev, viable, col)

        ax.axhline(MIN_ELEVATION_DEG, color="crimson", ls="--", lw=1.6,
                   label=f"Min QKD elevation ({MIN_ELEVATION_DEG}°)")
        ax.set_title(f"Satellite Elevation vs Time — {gs_name}")
        ax.set_xlabel("Time (minutes from simulation epoch)")
        ax.set_ylabel("Elevation (degrees)")
        ax.set_ylim(-10, 90)
        ax.legend(loc="upper right", ncol=2, fontsize=7)
        fig.tight_layout()

        path = Path(output_dir) / f"elevation_{gs_name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
        print(f"[PLOT] {path}")

    return saved


def _shade_viable(ax, t, elev, viable, col):
    """Fill the area under the curve during viable QKD windows."""
    in_window = False
    t0_w = None
    for i, v in enumerate(viable):
        if v and not in_window:
            t0_w, in_window = t[i], True
        elif not v and in_window:
            ax.axvspan(t0_w, t[i - 1], alpha=0.12, color=col)
            in_window = False
    if in_window:
        ax.axvspan(t0_w, t[-1], alpha=0.12, color=col)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SKR vs Time  (per ground station, all satellites stacked)
# ─────────────────────────────────────────────────────────────────────────────

def plot_skr_vs_time(
    result: SimulationResult,
    output_dir: str = "outputs/figures",
    dpi: int = 300,
) -> List[str]:
    df = result.to_dataframe()
    stations = df["ground_station"].unique()
    saved = []

    for gs_name in stations:
        df_gs  = df[df["ground_station"] == gs_name]
        sats   = df_gs["satellite"].unique()
        times  = df_gs["time_min"].unique()
        times.sort()

        # Aggregate best SKR at each time step
        best_skr = df_gs.groupby("time_min")["skr"].max().reindex(times, fill_value=0)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top: per-satellite SKR
        for idx, sat in enumerate(sats):
            df_sat = df_gs[df_gs["satellite"] == sat].sort_values("time_min")
            axes[0].plot(df_sat["time_min"], df_sat["skr"],
                         color=PALETTE[idx % len(PALETTE)], label=sat, alpha=0.8)
        axes[0].set_ylabel("Secure Key Rate (normalised)")
        axes[0].set_title(f"Per-Satellite SKR — {gs_name}")
        axes[0].legend(loc="upper right", ncol=3, fontsize=7)

        # Bottom: cumulative best SKR
        cum_skr = np.cumsum(best_skr.values)
        axes[1].fill_between(times, cum_skr, alpha=0.45, color="steelblue")
        axes[1].plot(times, cum_skr, color="steelblue", lw=1.5)
        axes[1].set_ylabel("Cumulative SKR")
        axes[1].set_xlabel("Time (min)")
        axes[1].set_title(f"Cumulative Best-Link SKR — {gs_name}")

        fig.tight_layout()
        path = Path(output_dir) / f"skr_{gs_name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
        print(f"[PLOT] {path}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 3.  QBER vs Slant Range  (all stations combined scatter)
# ─────────────────────────────────────────────────────────────────────────────

def plot_qber_vs_range(
    result: SimulationResult,
    output_dir: str = "outputs/figures",
    dpi: int = 300,
) -> str:
    df = result.to_dataframe()
    viable = df[df["link_viable"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    stations = viable["ground_station"].unique()
    markers  = ["o", "s", "^", "D", "v"]

    for idx, gs_name in enumerate(stations):
        dg = viable[viable["ground_station"] == gs_name]
        ax.scatter(dg["range_km"], dg["qber"] * 100,
                   label=gs_name, marker=markers[idx % len(markers)],
                   alpha=0.5, s=14, edgecolors="none")

    # Theoretical curve
    r_range = np.linspace(200, 2000, 300)
    from qkd_constellation.qkd.bb84 import qber_model
    from qkd_constellation.qkd.channel import total_link_loss_db
    theo_qber = []
    for r in r_range:
        loss, _ = total_link_loss_db(30.0, r)   # representative 30° elevation
        theo_qber.append(qber_model(loss) * 100)
    ax.plot(r_range, theo_qber, "k--", lw=1.5, label="Theory (30° elev.)")

    ax.axhline(11.0, color="crimson", ls=":", lw=1.4,
               label="BB84 QBER limit (11 %)")
    ax.set_xlabel("Slant Range (km)")
    ax.set_ylabel("QBER (%)")
    ax.set_title("QBER vs Slant Range — All Viable Links")
    ax.legend()
    fig.tight_layout()

    path = Path(output_dir) / "qber_vs_range.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Channel Efficiency η vs Range
# ─────────────────────────────────────────────────────────────────────────────

def plot_eta_vs_range(
    output_dir: str = "outputs/figures",
    dpi: int = 300,
) -> str:
    ranges = np.linspace(100, 2500, 500)
    eta    = np.array([geometric_efficiency(r) for r in ranges])
    eta_db = -10 * np.log10(eta + 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ranges, eta, color="darkorange", lw=2)
    axes[0].set_xlabel("Slant Range (km)")
    axes[0].set_ylabel("Geometric Efficiency η")
    axes[0].set_title("Beam-Collection Efficiency vs Range")
    axes[0].set_yscale("log")
    axes[0].annotate(
        "D = 0.50 m, θ = 10 μrad",
        xy=(0.05, 0.08), xycoords="axes fraction", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )

    axes[1].plot(ranges, eta_db, color="steelblue", lw=2)
    axes[1].axhline(LOSS_THRESHOLD_DB, color="crimson", ls="--",
                    label=f"Threshold ({LOSS_THRESHOLD_DB} dB)")
    axes[1].set_xlabel("Slant Range (km)")
    axes[1].set_ylabel("Geometric Loss (dB)")
    axes[1].set_title("Link Loss vs Range")
    axes[1].legend()

    fig.suptitle("Free-Space Optical Channel — Aperture-Based η Model", y=1.01)
    fig.tight_layout()

    path = Path(output_dir) / "eta_vs_range.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Station Summary Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_station_summary(
    result: SimulationResult,
    output_dir: str = "outputs/figures",
    dpi: int = 300,
) -> str:
    summaries = result.station_summaries
    names     = list(summaries.keys())
    total_skr = [summaries[n].total_skr for n in names]
    coverage  = [summaries[n].coverage_fraction * 100 for n in names]
    avg_qber  = [summaries[n].avg_qber * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    for ax, vals, title, ylabel in zip(
        axes,
        [total_skr, coverage, avg_qber],
        ["Cumulative SKR (24 h)", "Coverage (% time > 10°)", "Average QBER (%)"],
        ["Normalised SKR", "Coverage (%)", "QBER (%)"],
    ):
        bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Ground Station")
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{v:.3f}" if v < 1 else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8)

    fig.suptitle("Ground Station Performance Summary — 24h Simulation", fontsize=14)
    fig.tight_layout()

    path = Path(output_dir) / "station_summary.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SKR Heatmap: Satellite × Ground Station
# ─────────────────────────────────────────────────────────────────────────────

def plot_skr_heatmap(
    result: SimulationResult,
    output_dir: str = "outputs/figures",
    dpi: int = 300,
) -> str:
    df    = result.to_dataframe()
    pivot = df.pivot_table(values="skr", index="satellite",
                           columns="ground_station", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Cumulative SKR (normalised)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Satellite × Ground Station Cumulative SKR Heatmap (24 h)")
    ax.set_xlabel("Ground Station")
    ax.set_ylabel("Satellite")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=6.5, color="black" if val < pivot.values.max() * 0.6 else "white")

    fig.tight_layout()
    path = Path(output_dir) / "skr_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Ground Track of Constellation
# ─────────────────────────────────────────────────────────────────────────────

def plot_ground_track(
    result: SimulationResult,
    output_dir: str = "outputs/figures",
    dpi: int = 300,
    max_time_min: int = 200,
) -> str:
    """
    Ground tracks of all satellites for the first `max_time_min` minutes,
    with Indian ground stations marked.
    """
    from qkd_constellation.constellation.generator import generate_constellation
    from qkd_constellation.orbital.mechanics import eci_to_ecef, eci_position_km
    from qkd_constellation.ground_stations.stations import INDIA_GROUND_STATIONS
    import math

    cfg  = result.config.constellation
    sats = generate_constellation(
        num_planes=cfg.num_planes,
        sats_per_plane=cfg.sats_per_plane,
        altitude_km=cfg.altitude_km,
        inclination_deg=cfg.inclination_deg,
        eccentricity=cfg.eccentricity,
        arg_perigee_deg=cfg.arg_perigee_deg,
    )

    times_s = np.arange(0, max_time_min * 60, 60)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Draw simple plate-carrée world outline
    _draw_coastlines(ax)

    for idx, sat in enumerate(sats):
        lons, lats = [], []
        for t_s in times_s:
            r_eci  = eci_position_km(sat.elements, t_s)
            r_ecef = eci_to_ecef(r_eci, t_s)
            lat_d, lon_d = _ecef_to_latlon(r_ecef)
            lats.append(lat_d)
            lons.append(lon_d)

        # Split track at antimeridian crossing to avoid horizontal lines
        lons_arr = np.array(lons)
        lats_arr = np.array(lats)
        _plot_track_with_splits(ax, lons_arr, lats_arr,
                                color=PALETTE[idx % len(PALETTE)],
                                label=sat.name, alpha=0.7)

    # Ground stations
    for gs in INDIA_GROUND_STATIONS:
        ax.plot(gs.lon_deg, gs.lat_deg, "r*", markersize=12, zorder=5)
        ax.annotate(gs.name, (gs.lon_deg, gs.lat_deg),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="crimson", fontweight="bold")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Satellite Ground Tracks — First {max_time_min} minutes")
    ax.legend(loc="lower left", ncol=3, fontsize=7)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)

    fig.tight_layout()
    path = Path(output_dir) / "ground_tracks.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return str(path)


def _ecef_to_latlon(r_ecef: np.ndarray):
    import math
    x, y, z = r_ecef
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    return lat, lon


def _plot_track_with_splits(ax, lons, lats, **kwargs):
    """Plot ground track, splitting at ±180° antimeridian wrap."""
    segs_lon, segs_lat = [[]], [[]]
    for i in range(len(lons)):
        if i > 0 and abs(lons[i] - lons[i - 1]) > 180:
            segs_lon.append([])
            segs_lat.append([])
        segs_lon[-1].append(lons[i])
        segs_lat[-1].append(lats[i])
    for sl, sa in zip(segs_lon, segs_lat):
        ax.plot(sl, sa, **kwargs)
        kwargs.pop("label", None)   # only label first segment


def _draw_coastlines(ax):
    """Draw a simple graticule and continent outlines using matplotlib patches."""
    # Graticule
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color="lightgray", lw=0.4, zorder=0)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color="lightgray", lw=0.4, zorder=0)
    ax.set_facecolor("#e8f4f8")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Link Budget Waterfall (single representative pass)
# ─────────────────────────────────────────────────────────────────────────────

def plot_link_budget_waterfall(
    output_dir: str = "outputs/figures",
    elevation_deg: float = 45.0,
    range_km: float = 800.0,
    dpi: int = 300,
) -> str:
    from qkd_constellation.qkd.channel import channel_budget

    budget = channel_budget(elevation_deg, range_km)

    labels  = ["Geometric Loss", "Atmospheric Loss", "Fixed System Loss"]
    values  = [budget["geo_loss_dB"], budget["atm_loss_dB"], budget["base_loss_dB"]]
    running = [0.0]
    for v in values:
        running.append(running[-1] + v)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e05c5c", "#f0a050", "#5090d0"]
    for i, (label, val) in enumerate(zip(labels, values)):
        ax.bar(label, val, bottom=running[i], color=colors[i],
               edgecolor="white", lw=0.8, label=f"{label}: {val:.1f} dB")

    ax.axhline(LOSS_THRESHOLD_DB, color="crimson", ls="--",
               label=f"Link threshold ({LOSS_THRESHOLD_DB} dB)")
    ax.set_ylabel("Cumulative Loss (dB)")
    ax.set_title(
        f"Link Budget Waterfall\n"
        f"(elevation = {elevation_deg}°, range = {range_km:.0f} km, "
        f"total = {budget['total_loss_dB']:.1f} dB)"
    )
    ax.legend()
    fig.tight_layout()

    path = Path(output_dir) / "link_budget_waterfall.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SKR vs Elevation Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_skr_vs_elevation(
    output_dir: str = "outputs/figures",
    dpi: int = 300,
) -> str:
    from qkd_constellation.qkd.bb84 import secure_key_rate
    from qkd_constellation.constants import R_EARTH_KM

    alt_km = 600.0
    elevs  = np.linspace(10, 90, 200)

    fig, ax = plt.subplots(figsize=(9, 5))

    for mu_val, ls in [(0.3, "--"), (0.5, "-"), (0.7, "-.")]:
        skrs = []
        for e in elevs:
            e_rad = np.radians(e)
            # Approximate slant range from elevation
            R_s = R_EARTH_KM
            h   = alt_km
            sin_e = np.sin(e_rad)
            rng = np.sqrt((R_s + h)**2 - R_s**2 * np.cos(e_rad)**2) - R_s * sin_e
            _, skr = secure_key_rate(e, rng, mu=mu_val)
            skrs.append(skr)
        ax.plot(elevs, skrs, ls=ls, lw=1.8, label=f"μ = {mu_val}")

    ax.set_xlabel("Elevation Angle (degrees)")
    ax.set_ylabel("Secure Key Rate (normalised)")
    ax.set_title("SKR vs Elevation — Decoy-State BB84 (altitude 600 km)")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()

    path = Path(output_dir) / "skr_vs_elevation.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Master: generate all plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(result: SimulationResult, output_dir: str = "outputs/figures", dpi: int = 300) -> List[str]:
    """Generate and save every plot. Returns list of saved file paths."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved = []
    saved += plot_elevation_vs_time(result, output_dir, dpi)
    saved += plot_skr_vs_time(result, output_dir, dpi)
    saved.append(plot_qber_vs_range(result, output_dir, dpi))
    saved.append(plot_eta_vs_range(output_dir, dpi))
    saved.append(plot_station_summary(result, output_dir, dpi))
    saved.append(plot_skr_heatmap(result, output_dir, dpi))
    saved.append(plot_ground_track(result, output_dir, dpi))
    saved.append(plot_link_budget_waterfall(output_dir, dpi=dpi))
    saved.append(plot_skr_vs_elevation(output_dir, dpi))

    print(f"\n📊 {len(saved)} plots saved to {output_dir}/")
    return saved
