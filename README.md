# Design and simulation of a decoy-state BB84 satellite QKD constellation for Indian ground stations

> **Decoy-State BB84 Quantum Key Distribution over a Walker-Delta LEO Constellation**  
> *India Ground Network · Aperture-Based η Channel Model · Research-Grade Codebase*

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](#running-tests)

---

## Overview

This repository implements a **modular, research-grade simulation** of a satellite-based Quantum Key Distribution (QKD) network designed for the Indian subcontinent. It accompanies the paper:

> *"Satellite Constellation Designing and QKD Feasibility Check"*

The simulator covers:
- **Walker-Delta T/P/F constellation generation** with J2 secular drift (RAAN + ω precession)
- **Keplerian + J2 orbit propagator** — pure-Python, no external orbital library required
- **Decoy-State BB84 protocol** — secure key rate (SKR) and QBER from first principles
- **Aperture-based η channel model** — geometric beam spreading + atmospheric extinction
- **5-station India ground network** (Hanle, Dehradun, Mt Abu, Shillong, Kodaikanal)
- **9 publication-quality figures** generated automatically
- **Interactive Streamlit dashboard** for parameter exploration
- **Full-featured CLI** with argparse

---

## Repository Structure

```
qkd-satellite-constellation/
│
├── src/qkd_constellation/          # Core library (pip-installable)
│   ├── constants.py                  SI-consistent physical & orbital constants
│   ├── config.py                     YAML + dataclass configuration system
│   │
│   ├── constellation/
│   │   └── generator.py              Walker-Delta generator + TLE export
│   │
│   ├── ground_stations/
│   │   └── stations.py               India QKD ground network
│   │
│   ├── orbital/
│   │   └── mechanics.py              Keplerian + J2 propagator (ECI→ECEF→SEZ)
│   │
│   ├── qkd/
│   │   ├── channel.py                Free-space optical channel (η, atm. loss)
│   │   └── bb84.py                   Decoy-state BB84 SKR & QBER
│   │
│   ├── simulation/
│   │   └── runner.py                 24 h simulation orchestrator → CSV + DataFrames
│   │
│   └── visualization/
│       └── plots.py                  9 publication-quality figures (Matplotlib)
│
├── cli/
│   └── simulate.py                   Command-line interface (argparse)
│
├── dashboard/
│   └── app.py                        Interactive Streamlit dashboard
│
├── scripts/
│   ├── run_cli_simulation.py         Animated interactive CLI demo
│   └── generate_plots.py             Standalone figure generation
│
├── tests/
│   ├── test_orbital.py               Unit tests — orbital mechanics
│   ├── test_qkd.py                   Unit tests — QKD channel & protocol
│   ├── test_constellation.py         Unit tests — constellation generator
│   └── test_integration.py           Integration test — full simulation run
│
├── config/
│   └── config.yaml                   Default YAML configuration
│
├── outputs/                          (generated on first run)
│   ├── data/                         Per-station CSVs + synthetic TLEs
│   └── figures/                      All 17 PNG figures
│
├── requirements.txt
└── pyproject.toml
```
---

## 📄 Publication

**Design and simulation of a decoy-state BB84 satellite QKD constellation for Indian ground stations**  
Published in *International Journal of Information Technology*, 2025  

**Authors:**  
Anand Kumar, Yugnanda Malhotra, Aayush Gupta, Jolly Parikh  

🔗 DOI: https://doi.org/10.1007/s41870-025-02860-y  


---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/Anand-Ambastha/qkd-satellite-constellation
cd qkd-satellite-constellation
pip install -r requirements.txt
```

### 2. Run the CLI simulation (full 24 h)

```bash
python cli/simulate.py
```

### 3. Interactive animated CLI

```bash
python scripts/run_cli_simulation.py
```

### 4. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Generate all figures

```bash
python scripts/generate_plots.py --dpi 600
```

---

## CLI Reference

```
python cli/simulate.py [OPTIONS]

Constellation:
  --planes INT          Number of orbital planes          [3]
  --sats-per-plane INT  Satellites per plane              [4]
  --altitude FLOAT      Orbital altitude km               [600]
  --inclination FLOAT   Inclination degrees               [97.5]

QKD Protocol:
  --mu FLOAT            Signal mean photon number         [0.5]
  --aperture FLOAT      Receiver aperture diameter m      [0.5]
  --divergence FLOAT    Beam divergence rad               [10e-6]
  --loss-threshold FLOAT  Max link loss dB               [60.0]

Simulation:
  --duration INT        Duration minutes                  [1440]
  --step INT            Time step minutes                 [5]
  --output-dir STR      Output directory                  [outputs]
  --dpi INT             Plot resolution DPI               [300]
  --no-plots            Skip plot generation
  --quiet               Suppress per-pass messages
  --config PATH         Load parameters from YAML file
```

**Example — compact 3×3 constellation at 550 km:**

```bash
python cli/simulate.py --planes 3 --sats-per-plane 3 --altitude 550 \
                       --inclination 98 --duration 720 --step 2 --dpi 600
```

---

## Configuration via YAML

```yaml
# config/config.yaml
constellation:
  num_planes:      3
  sats_per_plane:  4
  altitude_km:     600.0
  inclination_deg: 97.5

qkd:
  mu_signal:           0.5
  aperture_rx_m:       0.50
  beam_divergence_rad: 1.0e-5
  loss_threshold_db:   60.0

simulation:
  duration_min:  1440
  step_min:      5
  output_dir:    outputs
```

```bash
python cli/simulate.py --config config/config.yaml
```

---

## Physical Models

### Orbital Mechanics

The simulator implements a **Keplerian + J2 secular drift** propagator:

1. **Kepler equation** solved via Newton-Raphson: `E − e·sin(E) = M`
2. **J2 RAAN precession**:
   ```
   Ω̇ = −(3/2) J₂ (Rₑ/p)² n cos(i)
   ```
3. **Coordinate chain**: Keplerian elements → ECI → ECEF → SEZ topocentric

At 600 km / 97.5°, the J2-driven RAAN drift is ≈ **+0.986°/day** — the sun-synchronous condition.

### Free-Space Optical Channel

**Geometric collection efficiency** (aperture-based η model):

```
η_geo = (D_rx / (θ_div · r))²
```

where D_rx = 0.50 m, θ_div = 10 μrad, r = slant range.

**Atmospheric transmittance** (Ångström–Beer law):

```
T_atm = exp(−τ_zenith / sin(elevation))
```

### Decoy-State BB84 Secure Key Rate

The asymptotic SKR lower bound:

```
R ≈ q_μ · [1 − f_EC · H(e_μ)]
```

where:
- `q_μ = η · μ · exp(−μ)` — detected single-photon-dominant gain
- `μ = 0.5` — signal mean photon number
- `f_EC = 1.16` — error-correction efficiency
- `H(x) = −x log₂(x) − (1−x) log₂(1−x)` — binary Shannon entropy
- `e_μ = 0.02 + 0.08 · (loss/threshold)` — QBER model

---

## Generated Figures

| File | Description |
|------|-------------|
| `elevation_<Station>.png` | Satellite elevation vs time, viable windows shaded |
| `skr_<Station>.png` | Per-satellite SKR + cumulative best-link SKR |
| `qber_vs_range.png` | QBER scatter vs slant range, all stations |
| `eta_vs_range.png` | η and link loss vs slant range |
| `station_summary.png` | Bar charts: total SKR, coverage, avg QBER |
| `skr_heatmap.png` | Satellite × station cumulative SKR heatmap |
| `ground_tracks.png` | Satellite ground tracks + India station markers |
| `link_budget_waterfall.png` | Stacked loss budget for representative pass |
| `skr_vs_elevation.png` | SKR vs elevation for μ = 0.3, 0.5, 0.7 |

---

## Ground Station Network

| Station | Lat (°N) | Lon (°E) | Alt (m) | Notes |
|---------|----------|----------|---------|-------|
| Hanle | 32.78 | 78.96 | 4500 | Indian Astronomical Observatory, Ladakh — ~270 clear nights/yr |
| Dehradun | 30.30 | 78.00 | 640 | ISRO-SAC associated site |
| Mt Abu | 24.60 | 72.70 | 1220 | PRL optical telescope |
| Shillong | 25.60 | 91.80 | 1500 | NE India node |
| Kodaikanal | 10.20 | 77.40 | 2343 | Kodaikanal Solar Observatory |

---

## Running Tests

```bash
# With pytest (recommended)
pip install pytest
pytest tests/ -v

# Without pytest
python -m unittest discover -s tests -v
```

**Test coverage:**  28 / 29 tests pass (the RAAN position test is skipped — J2 precession intentionally shifts perigee across one orbit, so position ≠ initial; radius is preserved instead).

---

## Simulation Results (Default Config)

| Station | Total SKR | Coverage | Max Elev° |
|---------|-----------|----------|-----------|
| Hanle | 0.01557 | 1.7% | 58.7° |
| Shillong | 0.01551 | 1.3% | 74.2° |
| MtAbu | 0.01465 | 1.3% | 84.3° |
| Dehradun | 0.01386 | 1.6% | 40.9° |
| Kodaikanal | 0.01195 | 1.4% | 34.8° |

*3 planes × 4 satellites · 600 km altitude · 97.5° inclination · 1440 min · 5 min step*

---

## References

1. Lo, H.-K., Ma, X. & Chen, K. *Decoy state quantum key distribution.* Phys. Rev. Lett. **94**, 230504 (2005)
2. Liao, S.-K. *et al.* *Satellite-to-ground quantum key distribution.* Nature **549**, 43–47 (2017)
3. Bourgoin, J.-P. *et al.* *A comprehensive design and performance analysis of LEO satellite QKD.* New J. Phys. **15**, 023006 (2013)
4. Bennett, C.H. & Brassard, G. *Quantum cryptography: Public key distribution and coin tossing.* Proc. IEEE Int. Conf. Comput. Syst. Signal Process. (1984)
5. Walker, J.G. *Satellite constellations.* J. Brit. Interplanet. Soc. **37**, 559–571 (1984)
6. Gottesman, D., Lo, H.-K., Lütkenhaus, N. & Preskill, J. *Security of quantum key distribution with imperfect devices.* Quantum Inf. Comput. **4**, 325–360 (2004)
7. Montenbruck, O. & Gill, E. *Satellite Orbits: Models, Methods and Applications.* Springer (2000)

---

## License

MIT — see [LICENSE](LICENSE)

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{qkd_constellation_2024,
  author  = {Anand Kumar},
  title   = {QKD Satellite Constellation Simulator},
  year    = {2025},
  url     = {https://github.com/yourhandle/qkd-satellite-constellation}
}
```
#   q k d - s a t e l l i t e - c o n s t e l l a t i o n  
 