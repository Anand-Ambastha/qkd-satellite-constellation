"""
Decoy-State BB84 Quantum Key Distribution Protocol.

Implements the asymptotic secure key rate formula for the weak-coherent-pulse
decoy-state method under collective attacks.

Protocol overview:
  - Signal pulses: mean photon number μ (Poissonian)
  - Decoy pulses:  mean photon number ν < μ
  - Key rate lower bound (GLLP + Lo-Chau-Ardehali):
      R ≥ Q_1(1 − H(e_1)) − Q_μ · f_EC · H(e_μ)
  - Simplified aperture-efficiency version (matches original notebook):
      R = q_μ · (1 − f_EC · H(e_μ))
    where q_μ = η · μ · exp(−μ)  (single-photon-dominant term)

References:
  BB84:    Bennett & Brassard, Proc. IEEE ISIT (1984)
  Decoy:   Lo, Ma & Chen, PRL 94, 230504 (2005)
           Hwang, PRL 91, 057901 (2003)
  GLLP:    Gottesman, Lo, Lütkenhaus & Preskill, QIC 4, 325 (2004)
  Liao et al., Nature 549, 43–47 (2017)  — satellite demonstration
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from qkd_constellation.constants import (
    MU_SIGNAL,
    F_EC,
    QBER_BASE,
    QBER_LOSS_SLOPE,
    LOSS_THRESHOLD_DB,
    MIN_ELEVATION_DEG,
)
from qkd_constellation.qkd.channel import total_link_loss_db


# ──────────────────────────────────────────────────────────────────────────────
# Information-theoretic helpers
# ──────────────────────────────────────────────────────────────────────────────

def binary_entropy(p: float) -> float:
    """
    Binary Shannon entropy H(p) = -p·log₂(p) - (1-p)·log₂(1-p).
    Returns 0 for p ∈ {0, 1} and 1 for p = 0.5.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


# ──────────────────────────────────────────────────────────────────────────────
# QBER model
# ──────────────────────────────────────────────────────────────────────────────

def qber_model(
    total_loss_db: float,
    loss_threshold_db: float = LOSS_THRESHOLD_DB,
    qber_base: float = QBER_BASE,
    qber_slope: float = QBER_LOSS_SLOPE,
) -> float:
    """
    Empirical QBER as a function of link loss.

    e_μ = e_base + slope · (loss / threshold)

    The baseline (2 %) represents alignment + dark-count errors.
    The slope term captures detector noise growth at long range.
    """
    normalised = min(total_loss_db / loss_threshold_db, 1.0)
    return qber_base + qber_slope * normalised


# ──────────────────────────────────────────────────────────────────────────────
# Decoy-state BB84 SKR
# ──────────────────────────────────────────────────────────────────────────────

def secure_key_rate(
    elevation_deg: float,
    range_km: float,
    mu: float = MU_SIGNAL,
    f_ec: float = F_EC,
    loss_threshold_db: float = LOSS_THRESHOLD_DB,
) -> Tuple[float, float]:
    """
    Compute QBER and normalised secure key rate (SKR) for a satellite pass.

    Parameters
    ----------
    elevation_deg     : satellite elevation above horizon [°]
    range_km          : slant range from ground station [km]
    mu                : mean signal photon number
    f_ec              : error-correction efficiency (Shannon limit = 1.0)
    loss_threshold_db : link budget cut-off [dB]

    Returns
    -------
    (qber, skr)  — both dimensionless / normalised
      qber = 1.0 and skr = 0.0 when the link is not usable
    """
    if elevation_deg < MIN_ELEVATION_DEG:
        return 1.0, 0.0

    total_loss_db, eta = total_link_loss_db(elevation_deg, range_km)

    if total_loss_db > loss_threshold_db:
        return 0.5, 0.0

    # Detected single-photon–dominant gain
    q_mu = eta * mu * math.exp(-mu)

    # QBER
    e_mu = qber_model(total_loss_db, loss_threshold_db)

    if e_mu >= 0.5:
        return e_mu, 0.0

    # Decoy-state secure key fraction
    skr = q_mu * (1.0 - f_ec * binary_entropy(e_mu))
    return e_mu, max(0.0, skr)


# ──────────────────────────────────────────────────────────────────────────────
# Detailed per-pass analysis
# ──────────────────────────────────────────────────────────────────────────────

def pass_analysis(
    elevation_deg: float,
    range_km: float,
    mu: float = MU_SIGNAL,
    f_ec: float = F_EC,
    loss_threshold_db: float = LOSS_THRESHOLD_DB,
) -> dict:
    """
    Full diagnostic breakdown for a single satellite-ground geometry.

    Useful for link-budget tables in research publications.
    """
    from qkd_constellation.qkd.channel import channel_budget

    budget = channel_budget(elevation_deg, range_km)
    qber, skr = secure_key_rate(elevation_deg, range_km, mu, f_ec, loss_threshold_db)
    total_loss = budget["total_loss_dB"]
    q_mu = budget["eta_geometric"] * mu * math.exp(-mu)

    return {
        **budget,
        "mu": mu,
        "q_mu": round(q_mu, 8),
        "qber": round(qber, 6),
        "H_qber": round(binary_entropy(qber), 6),
        "skr": round(skr, 8),
        "link_viable": skr > 0,
    }
