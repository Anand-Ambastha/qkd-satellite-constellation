"""
Integration test — runs a short simulation and validates the output structure.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qkd_constellation.config import (
    AppConfig, ConstellationConfig, QKDConfig, SimulationConfig,
)
from qkd_constellation.simulation.runner import run_simulation


@pytest.fixture(scope="module")
def small_result(tmp_path_factory):
    """Run a 60-minute, 3-satellite simulation for fast integration testing."""
    tmp = tmp_path_factory.mktemp("outputs")
    cfg = AppConfig(
        constellation=ConstellationConfig(num_planes=2, sats_per_plane=2),
        simulation=SimulationConfig(
            duration_min=60,
            step_min=5,
            output_dir=str(tmp),
            save_csv=True,
            save_plots=False,
        ),
    )
    return run_simulation(cfg, verbose=False)


class TestSimulationResult:
    def test_records_non_empty(self, small_result):
        assert len(small_result.records) > 0

    def test_record_count(self, small_result):
        """60 min / 5 min * 4 sats * 5 stations = 240 records."""
        expected = (60 // 5) * 4 * 5
        assert len(small_result.records) == expected

    def test_station_summaries_populated(self, small_result):
        assert len(small_result.station_summaries) == 5

    def test_skr_non_negative(self, small_result):
        for r in small_result.records:
            assert r.skr >= 0.0

    def test_qber_bounded(self, small_result):
        for r in small_result.records:
            assert 0.0 <= r.qber <= 1.0

    def test_dataframe_columns(self, small_result):
        df = small_result.to_dataframe()
        required = {"satellite", "ground_station", "time_min",
                    "elevation_deg", "range_km", "qber", "skr", "link_viable"}
        assert required.issubset(df.columns)

    def test_csv_files_created(self, small_result, tmp_path_factory):
        """CSVs should be written for each ground station."""
        out_dir = Path(small_result.config.simulation.output_dir)
        csv_files = list((out_dir / "data").glob("qkd_results_*.csv"))
        assert len(csv_files) == 5

    def test_tle_file_created(self, small_result):
        out_dir = Path(small_result.config.simulation.output_dir)
        tle_file = out_dir / "data" / "synthetic_tle.txt"
        assert tle_file.exists()
        assert tle_file.stat().st_size > 0

    def test_elapsed_positive(self, small_result):
        assert small_result.elapsed_s > 0

    def test_constellation_info(self, small_result):
        assert "total_satellites" in small_result.constellation_info
        assert small_result.constellation_info["total_satellites"] == 4
