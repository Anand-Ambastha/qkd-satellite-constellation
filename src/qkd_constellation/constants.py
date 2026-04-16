"""
Physical, orbital, and QKD protocol constants.

All values are SI-consistent unless noted. References:
  - Wertz & Larson, "Space Mission Engineering" (2011)
  - Lo et al., Phys. Rev. Lett. 94, 230504 (2005) — decoy-state BB84
  - Liao et al., Nature 549, 43–47 (2017) — satellite QKD
"""

# ──────────────────────────────────────────────
# Earth & Orbital Constants
# ──────────────────────────────────────────────
R_EARTH_KM: float = 6378.137          # WGS-84 equatorial radius [km]
MU_EARTH: float = 398600.4418         # gravitational parameter [km³/s²]
J2: float = 1.08263e-3                # Earth oblateness coefficient
EARTH_ROT_RAD_PER_S: float = 7.2921150e-5  # Earth sidereal rotation rate [rad/s]
EARTH_FLATTENING: float = 1.0 / 298.257223563  # WGS-84 flattening

# ──────────────────────────────────────────────
# Photon / Optical Channel
# ──────────────────────────────────────────────
LAMBDA_NM: float = 850.0              # QKD photon wavelength [nm]
LAMBDA_M: float = LAMBDA_NM * 1e-9   # [m]

APERTURE_TX_M: float = 0.15          # Transmitter aperture diameter [m]  (satellite)
APERTURE_RX_M: float = 0.50          # Receiver aperture diameter [m]  (ground)
BEAM_DIVERGENCE_RAD: float = 10e-6   # Full-angle beam divergence [rad]
BASE_LOSS_DB: float = 5.0            # Fixed optical + detector loss [dB]
LOSS_THRESHOLD_DB: float = 60.0      # Max acceptable link loss [dB]

# Atmospheric / visibility
VISIBILITY_KM: float = 15.0          # Climatological visibility [km]
K_AOD: float = 1.0                   # Aerosol optical depth scaling
MIN_ELEVATION_DEG: float = 10.0      # Min usable elevation angle [°]

# ──────────────────────────────────────────────
# Decoy-State BB84 Protocol Parameters
# ──────────────────────────────────────────────
MU_SIGNAL: float = 0.5               # Mean photon number for signal pulses
MU_DECOY: float = 0.1                # Mean photon number for decoy pulses
F_EC: float = 1.16                   # Error-correction efficiency factor
QBER_BASE: float = 0.02              # Baseline QBER (alignment + detector noise)
QBER_LOSS_SLOPE: float = 0.08        # QBER increase per unit normalised loss

# ──────────────────────────────────────────────
# Simulation Defaults
# ──────────────────────────────────────────────
SIM_DURATION_MIN: int = 1440         # 24-hour simulation window [min]
SIM_STEP_MIN: int = 5                # Temporal resolution [min]
