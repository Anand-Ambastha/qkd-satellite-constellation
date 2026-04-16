"""
Free-space optical channel model for satellite-to-ground QKD.

Models geometric beam spreading, pointing loss, and atmospheric extinction
as a function of satellite elevation and slant range.

Physical basis:
  - Geometric loss: η_geo = (D_rx / (θ_div · r))²
    where D_rx is the receiver aperture, θ_div is the full-angle beam divergence,
    r is the slant range.
  - Atmospheric loss: Angström turbidity model with visibility-based extinction
  - Fixed system losses: detector efficiency, coupling, optical elements

References:
  Liao et al., Nature 549, 43–47 (2017)
  Bourgoin et al., New J. Phys. 15, 023006 (2013)
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from qkd_constellation.constants import (
    APERTURE_RX_M,
    BEAM_DIVERGENCE_RAD,
    BASE_LOSS_DB,
    LOSS_THRESHOLD_DB,
    VISIBILITY_KM,
    K_AOD,
    LAMBDA_M,
    LAMBDA_NM,
    MIN_ELEVATION_DEG,
)
from qkd_constellation.orbital.mechanics import air_mass_factor


# ──────────────────────────────────────────────────────────────────────────────
# Geometric channel efficiency
# ──────────────────────────────────────────────────────────────────────────────

def geometric_efficiency(
    range_km: float,
    aperture_rx_m: float = APERTURE_RX_M,
    divergence_rad: float = BEAM_DIVERGENCE_RAD,
) -> float:
    """
    Fraction of transmitted photons captured by the receiver aperture
    due to diffraction-limited beam spreading.

    η_geo = (D_rx / (θ_div · r))²   [dimensionless, clamped to [0, 1]]
    """
    if range_km <= 0:
        return 0.0
    r_m = range_km * 1e3
    eta = (aperture_rx_m / (divergence_rad * r_m)) ** 2
    return float(min(1.0, eta))


# ──────────────────────────────────────────────────────────────────────────────
# Atmospheric extinction (Angström–Beer law)
# ──────────────────────────────────────────────────────────────────────────────

def atmospheric_transmittance(
    elevation_deg: float,
    visibility_km: float = VISIBILITY_KM,
    k_aod: float = K_AOD,
    wavelength_nm: float = LAMBDA_NM,
) -> float:
    """
    Zenith atmospheric optical depth (AOD) via the Ångström turbidity formula,
    scaled by air-mass for the actual elevation angle.

    T_atm = exp(-τ · X(elevation))

    where:
      τ = k_AOD / V_km  (rough aerosol + Rayleigh)
      X = air-mass factor = 1/sin(elevation)
    """
    if elevation_deg < MIN_ELEVATION_DEG:
        return 0.0
    tau_zenith = k_aod / visibility_km
    X = air_mass_factor(elevation_deg)
    return float(np.exp(-tau_zenith * X))


# ──────────────────────────────────────────────────────────────────────────────
# Combined link-budget loss
# ──────────────────────────────────────────────────────────────────────────────

def total_link_loss_db(
    elevation_deg: float,
    range_km: float,
    aperture_rx_m: float = APERTURE_RX_M,
    divergence_rad: float = BEAM_DIVERGENCE_RAD,
    base_loss_db: float = BASE_LOSS_DB,
) -> Tuple[float, float]:
    """
    Compute total downlink loss [dB] and geometric channel efficiency η.

    Returns
    -------
    (total_loss_dB, eta_geometric)
    """
    eta_geo = geometric_efficiency(range_km, aperture_rx_m, divergence_rad)
    geo_loss_db = -10.0 * math.log10(eta_geo + 1e-30)
    return base_loss_db + geo_loss_db, eta_geo


# ──────────────────────────────────────────────────────────────────────────────
# Public summary helper
# ──────────────────────────────────────────────────────────────────────────────

def channel_budget(
    elevation_deg: float,
    range_km: float,
    aperture_rx_m: float = APERTURE_RX_M,
    divergence_rad: float = BEAM_DIVERGENCE_RAD,
    base_loss_db: float = BASE_LOSS_DB,
) -> dict:
    """Return a full link budget dictionary for diagnostics."""
    eta_geo = geometric_efficiency(range_km, aperture_rx_m, divergence_rad)
    T_atm   = atmospheric_transmittance(elevation_deg)
    geo_loss_db = -10.0 * math.log10(eta_geo + 1e-30)
    atm_loss_db = -10.0 * math.log10(T_atm + 1e-30) if T_atm > 0 else 999.0
    total_db    = base_loss_db + geo_loss_db + atm_loss_db

    return {
        "elevation_deg": elevation_deg,
        "range_km": range_km,
        "eta_geometric": eta_geo,
        "T_atmospheric": T_atm,
        "geo_loss_dB": round(geo_loss_db, 3),
        "atm_loss_dB": round(atm_loss_db, 3),
        "base_loss_dB": base_loss_db,
        "total_loss_dB": round(total_db, 3),
    }
