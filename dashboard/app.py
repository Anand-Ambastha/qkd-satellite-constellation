"""
QKD Satellite Constellation — Interactive Streamlit Dashboard

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow import from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import io
import time

from qkd_constellation.config import (
    AppConfig, ConstellationConfig, QKDConfig, SimulationConfig,
)
from qkd_constellation.simulation.runner import run_simulation
from qkd_constellation.visualization.plots import (
    plot_elevation_vs_time,
    plot_skr_vs_time,
    plot_qber_vs_range,
    plot_eta_vs_range,
    plot_station_summary,
    plot_skr_heatmap,
    plot_ground_track,
    plot_link_budget_waterfall,
    plot_skr_vs_elevation,
)
from qkd_constellation.qkd.bb84 import secure_key_rate, pass_analysis, binary_entropy
from qkd_constellation.qkd.channel import geometric_efficiency, channel_budget
from qkd_constellation.constellation.generator import (
    generate_constellation, constellation_summary,
)
from qkd_constellation.ground_stations.stations import INDIA_GROUND_STATIONS
from qkd_constellation.constants import R_EARTH_KM


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QKD Satellite Constellation Simulator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: 700; color: #1a3a5c;}
    .metric-box {background:#f0f6ff; border-radius:8px; padding:12px 18px; margin:4px;}
    .sidebar .sidebar-content {background: #1a1a2e;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Parameters
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/ISRO_Logo.svg/200px-ISRO_Logo.svg.png", width=80)
    st.title("🛰️ QKD Sim Parameters")

    st.header("🌐 Constellation")
    num_planes       = st.slider("Orbital Planes",        1, 6,  3)
    sats_per_plane   = st.slider("Satellites per Plane",  1, 8,  4)
    altitude_km      = st.slider("Altitude (km)",       300, 1200, 600, step=50)
    inclination_deg  = st.slider("Inclination (°)",      50.0, 110.0, 97.5, step=0.5)

    st.header("🔬 QKD Protocol")
    mu_signal        = st.slider("Signal Photon Number μ", 0.1, 1.0, 0.5, step=0.05)
    aperture_rx_m    = st.slider("Receiver Aperture (m)",  0.1, 1.0, 0.5, step=0.05)
    divergence_urad  = st.slider("Beam Divergence (μrad)", 1.0, 50.0, 10.0, step=1.0)
    loss_threshold   = st.slider("Loss Threshold (dB)",   30.0, 80.0, 60.0, step=5.0)

    st.header("⏱️ Simulation")
    duration_min     = st.select_slider("Duration (min)", options=[60, 120, 360, 720, 1440], value=1440)
    step_min         = st.select_slider("Time Step (min)", options=[1, 2, 5, 10, 15], value=5)

    run_btn = st.button("🚀  Run Simulation", type="primary", use_container_width=True)

    st.markdown("---")
    st.header("🔍 Single Link Explorer")
    explorer_elev = st.slider("Elevation (°)",  10, 90, 45)
    explorer_rng  = st.slider("Slant Range (km)", 200, 2000, 800, step=50)
    st.markdown("---")
    st.caption("Research-grade simulation | Decoy-State BB84 | Walker-Δ Constellation")


# ──────────────────────────────────────────────────────────────────────────────
# Title
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🛰️ QKD Satellite Constellation Simulator</p>', unsafe_allow_html=True)
st.markdown("**Decoy-State BB84 Protocol · Walker-Delta Constellation · India Ground Network · Aperture-Based η Model**")
st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────────────────────────────────────

tab_sim, tab_channel, tab_explorer, tab_theory = st.tabs([
    "📡 Full Simulation",
    "📶 Channel Model",
    "🔭 Link Explorer",
    "📘 Theory & Parameters",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Full Simulation
# ══════════════════════════════════════════════════════════════════════════════

with tab_sim:

    # Constellation info (always shown)
    sats = generate_constellation(num_planes, sats_per_plane, altitude_km, inclination_deg)
    info = constellation_summary(sats)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Satellites",    info["total_satellites"])
    col2.metric("Altitude (km)",       info["altitude_km"])
    col3.metric("Inclination (°)",     info["inclination_deg"])
    col4.metric("Period (min)",        info["period_min"])
    col5.metric("RAAN Drift (°/day)",  info["raan_dot_deg_per_day"])

    st.markdown("---")

    if run_btn:
        cfg = AppConfig(
            constellation=ConstellationConfig(
                num_planes=num_planes,
                sats_per_plane=sats_per_plane,
                altitude_km=altitude_km,
                inclination_deg=inclination_deg,
            ),
            qkd=QKDConfig(
                mu_signal=mu_signal,
                aperture_rx_m=aperture_rx_m,
                beam_divergence_rad=divergence_urad * 1e-6,
                loss_threshold_db=loss_threshold,
            ),
            simulation=SimulationConfig(
                duration_min=duration_min,
                step_min=step_min,
                output_dir="outputs",
                save_csv=True,
                save_plots=False,   # plots generated separately below
                dpi=150,
            ),
        )

        with st.spinner(f"Simulating {num_planes * sats_per_plane} satellites × {len(INDIA_GROUND_STATIONS)} stations × {duration_min // step_min} time steps …"):
            result = run_simulation(cfg, verbose=False)

        st.session_state["result"] = result
        st.success(f"✅ Simulation complete in {result.elapsed_s:.2f} s  |  "
                   f"{len(result.records):,} link records computed")

    if "result" in st.session_state:
        result = st.session_state["result"]
        df = result.to_dataframe()

        # ── Summary metrics ──────────────────────────────────────────────────
        st.subheader("Ground Station Performance")
        sum_df = result.summary_dataframe()
        st.dataframe(
            sum_df.style.format({
                "total_skr": "{:.6f}",
                "avg_qber": "{:.4f}",
                "coverage_fraction": "{:.3f}",
                "max_elevation_deg": "{:.2f}",
            }),
            use_container_width=True
        )

        # ── Plots ─────────────────────────────────────────────────────────
        fig_dir = "outputs/figures"
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

        st.subheader("📊 Visualisations")

        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.markdown("**Station Summary**")
            saved = plot_station_summary(result, fig_dir, dpi=150)
            st.image(saved, use_container_width=True)

        with plot_col2:
            st.markdown("**SKR Heatmap (Satellite × Station)**")
            saved = plot_skr_heatmap(result, fig_dir, dpi=150)
            st.image(saved, use_container_width=True)

        st.markdown("**Ground Tracks**")
        saved = plot_ground_track(result, fig_dir, dpi=120)
        st.image(saved, use_container_width=True)

        # Per-station elevation plots
        st.subheader("Elevation Profiles")
        stations = df["ground_station"].unique()
        gs_tabs  = st.tabs(list(stations))

        for tab, gs_name in zip(gs_tabs, stations):
            with tab:
                col_a, col_b = st.columns(2)
                elev_paths = plot_elevation_vs_time(result, fig_dir, dpi=120)
                skr_paths  = plot_skr_vs_time(result, fig_dir, dpi=120)

                for p in elev_paths:
                    if gs_name in p:
                        col_a.image(p, caption=f"Elevation — {gs_name}", use_container_width=True)
                for p in skr_paths:
                    if gs_name in p:
                        col_b.image(p, caption=f"SKR — {gs_name}", use_container_width=True)

        # ── Data export ──────────────────────────────────────────────────────
        st.subheader("📥 Data Export")
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("⬇️  Download Full Results CSV", csv_bytes,
                           "qkd_simulation_results.csv", "text/csv")

    else:
        st.info("👈  Configure parameters in the sidebar and press **Run Simulation** to start.")

        # Show static channel plots as preview
        col1, col2 = st.columns(2)
        with col1:
            p = plot_eta_vs_range("outputs/figures", dpi=100)
            st.image(p, caption="Channel Efficiency Preview", use_container_width=True)
        with col2:
            p = plot_skr_vs_elevation("outputs/figures", dpi=100)
            st.image(p, caption="SKR vs Elevation Preview", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Channel Model
# ══════════════════════════════════════════════════════════════════════════════

with tab_channel:
    st.subheader("Free-Space Optical Channel Analysis")
    st.markdown(
        "Explore the geometric beam-collection efficiency and link loss "
        "as a function of slant range and orbital altitude."
    )

    c1, c2 = st.columns(2)
    with c1:
        ch_altitude = st.slider("Orbital Altitude (km)", 300, 1200, 600, step=50, key="ch_alt")
        ch_aperture = st.slider("Receiver Aperture (m)", 0.1, 1.5, 0.5, step=0.05, key="ch_ap")
    with c2:
        ch_div  = st.slider("Beam Divergence (μrad)", 1.0, 50.0, 10.0, step=1.0, key="ch_div")
        ch_elev = st.slider("Elevation Angle (°)",     10,  90, 45, key="ch_el")

    ranges  = np.linspace(100, 2500, 300)
    etas    = [geometric_efficiency(r, ch_aperture, ch_div * 1e-6) for r in ranges]
    eta_db  = [-10 * np.log10(e + 1e-30) for e in etas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(ranges, etas, color="darkorange", lw=2)
    axes[0].set_xlabel("Slant Range (km)")
    axes[0].set_ylabel("η (geometric)")
    axes[0].set_title("Collection Efficiency vs Range")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(ranges, eta_db, color="steelblue", lw=2)
    axes[1].axhline(loss_threshold, color="crimson", ls="--",
                    label=f"Threshold ({loss_threshold:.0f} dB)")
    axes[1].set_xlabel("Slant Range (km)")
    axes[1].set_ylabel("Geometric Loss (dB)")
    axes[1].set_title("Link Loss vs Range")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Budget table for chosen elevation + range
    st.markdown("**Link Budget at Selected Geometry**")
    budget = channel_budget(ch_elev, 800.0, ch_aperture, ch_div * 1e-6)
    budget_df = pd.DataFrame([budget])
    st.dataframe(budget_df.style.format("{:.4f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Single Link Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab_explorer:
    st.subheader("🔭 Single Link Pass Explorer")
    st.markdown("Analyse any satellite–ground geometry in detail.")

    analysis = pass_analysis(
        explorer_elev, explorer_rng,
        mu=mu_signal,
        f_ec=1.16,
        loss_threshold_db=loss_threshold,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QBER",              f"{analysis['qber']*100:.3f} %")
    c2.metric("Secure Key Rate",   f"{analysis['skr']:.3e}")
    c3.metric("Geometric η",       f"{analysis['eta_geometric']:.4e}")
    c4.metric("Total Loss",        f"{analysis['total_loss_dB']:.1f} dB")

    viable = "✅ Link viable" if analysis["link_viable"] else "❌ Link not viable"
    st.markdown(f"**Status:** {viable}")

    # Full budget table
    st.markdown("**Complete Link Budget**")
    rows = {
        "Parameter": list(analysis.keys()),
        "Value":     list(analysis.values()),
    }
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # SKR sweep over elevation
    st.markdown("**SKR vs Elevation (at given range)**")
    elevs_sw = np.linspace(10, 90, 100)
    skrs_sw  = [secure_key_rate(e, explorer_rng, mu=mu_signal,
                                loss_threshold_db=loss_threshold)[1]
                for e in elevs_sw]

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(elevs_sw, skrs_sw, color="steelblue", lw=2)
    ax2.axvline(explorer_elev, color="crimson", ls="--", label=f"Selected: {explorer_elev}°")
    ax2.set_xlabel("Elevation (°)")
    ax2.set_ylabel("SKR (normalised)")
    ax2.set_yscale("log")
    ax2.set_title(f"SKR vs Elevation  (range = {explorer_rng} km, μ = {mu_signal})")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Theory & Parameters
# ══════════════════════════════════════════════════════════════════════════════

with tab_theory:
    st.subheader("📘 Theoretical Background")

    st.markdown("""
### Protocol: Decoy-State BB84

The simulator implements the **weak coherent pulse (WCP) decoy-state BB84**
protocol as described by Lo, Ma & Chen (PRL 2005):

**Secure Key Rate (asymptotic, simplified):**

$$R \\approx q_\\mu \\left[1 - f_{EC} \\cdot H(e_\\mu)\\right]$$

where
- $q_\\mu = \\eta \\cdot \\mu \\cdot e^{-\\mu}$ — detected single-photon gain
- $\\eta$ — channel transmittance (geometric + atmospheric)
- $\\mu$ — signal mean photon number (default 0.5)
- $f_{EC} = 1.16$ — error-correction efficiency
- $H(x) = -x \\log_2 x - (1-x) \\log_2(1-x)$ — binary entropy
- $e_\\mu$ — quantum bit error rate (QBER)

---

### Channel Model: Aperture-Based η

Geometric collection efficiency:

$$\\eta_{\\text{geo}} = \\left(\\frac{D_{rx}}{\\theta_{\\text{div}} \\cdot r}\\right)^2$$

where $D_{rx} = 0.5$ m (receiver aperture), $\\theta_{\\text{div}} = 10\\,\\mu\\text{rad}$
(beam divergence), and $r$ is the slant range.

---

### Orbital Model: Walker-Delta Constellation

A $T/P/F$ Walker constellation with $T$ satellites distributed evenly across
$P$ orbital planes, with phasing factor $F$. J₂ secular drift is applied to
RAAN and argument of perigee.

**RAAN precession (J₂):**

$$\\dot{\\Omega} = -\\frac{3}{2} J_2 \\left(\\frac{R_\\oplus}{p}\\right)^2 n \\cos i$$

---

### Ground Network

| Station     | Lat (°N) | Lon (°E) | Alt (m) | Notes |
|-------------|----------|----------|---------|-------|
| Hanle       | 32.78    | 78.96    | 4500    | Indian Astronomical Observatory |
| Dehradun    | 30.30    | 78.00    | 640     | ISRO-SAC associated site |
| Mt Abu      | 24.60    | 72.70    | 1220    | PRL optical telescope |
| Shillong    | 25.60    | 91.80    | 1500    | NE India node |
| Kodaikanal  | 10.20    | 77.40    | 2343    | Southern anchor |

---

### References
1. Lo, H.-K., Ma, X. & Chen, K. *Decoy state quantum key distribution.* PRL 94, 230504 (2005)
2. Liao, S.-K. *et al.* *Satellite-to-ground quantum key distribution.* Nature 549, 43–47 (2017)
3. Bourgoin, J.-P. *et al.* *A comprehensive design and performance analysis of LEO satellite QKD.* New J. Phys. 15, 023006 (2013)
4. Bennett, C.H. & Brassard, G. *Quantum cryptography: Public key distribution and coin tossing.* Proc. IEEE ISIT (1984)
5. Walker, J.G. *Satellite constellations.* J. Brit. Interplanet. Soc. 37, 559–571 (1984)
""")

    st.markdown("---")
    st.subheader("Current Simulation Parameters")
    params_df = pd.DataFrame({
        "Parameter": [
            "Signal photon number μ", "Error-correction efficiency f_EC",
            "QBER baseline", "QBER loss slope",
            "Receiver aperture", "Beam divergence",
            "Loss threshold", "Min elevation",
        ],
        "Value": [
            mu_signal, 1.16, 0.02, 0.08,
            f"{aperture_rx_m} m", f"{divergence_urad} μrad",
            f"{loss_threshold} dB", f"{10.0}°",
        ],
    })
    st.dataframe(params_df, use_container_width=True)
