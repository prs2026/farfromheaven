"""Offline checks for Open-Meteo conversion and vibration-input preparation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from Vibes.openmeteo_profile import fetch_profile, plot_profile, profile_from_response, save_profile
from Vibes.prepare_vibration_trajectory import prepare_vibration_trajectory
from Vibes import preliminary_vibration
from Vibes.vibration_workflow import run_workflow
from Vibes.vibration_rsf import save_rsf, load_rsf


def example_response():
    return {
        "elevation": 1000.0,
        "hourly": {
            "time": ["2026-09-26T14:00"],
            "surface_pressure": [900.0],
            "temperature_2m": [20.0],
            "wind_speed_10m": [5.0],
            "wind_direction_10m": [270.0],
            "geopotential_height_850hPa": [1500.0],
            "temperature_850hPa": [15.0],
            "wind_speed_850hPa": [10.0],
            "wind_direction_850hPa": [270.0],
            "geopotential_height_700hPa": [3000.0],
            "temperature_700hPa": [5.0],
            "wind_speed_700hPa": [15.0],
            "wind_direction_700hPa": [270.0],
        },
    }


class VibrationWorkflowTests(unittest.TestCase):
    def test_historical_profile_and_properties(self):
        with patch("Vibes.openmeteo_profile._get_json", return_value=example_response()) as get:
            profile, metadata = fetch_profile(40.88, -119.08, "2026-09-26T14:00",
                                              "America/Los_Angeles", "historical")
        self.assertIn("historical-forecast-api.open-meteo.com", get.call_args.args[0])
        self.assertEqual(metadata["mode"], "historical")
        self.assertEqual(len(profile), 3)
        self.assertAlmostEqual(profile.loc[0, "pressure_pa"], 90000)
        self.assertAlmostEqual(profile.loc[0, "wind_u_mps"], 5)
        self.assertAlmostEqual(profile.loc[0, "air_density_kgm3"],
                               90000 / (287.05 * 293.15))
        with tempfile.TemporaryDirectory() as temp:
            plot = plot_profile(profile, Path(temp) / "weather.png")
            self.assertTrue(plot.exists())

    def test_wrong_returned_hour_rejected(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        with self.assertRaisesRegex(ValueError, "requested local hour"):
            profile_from_response(example_response(),
                                  datetime(2026, 9, 26, 15, tzinfo=ZoneInfo("America/Los_Angeles")))

    def test_launch_minute_rounds_to_nearest_hour(self):
        response = example_response()
        response["hourly"]["time"] = ["2026-09-26T15:00"]
        with patch("Vibes.openmeteo_profile._get_json", return_value=response) as get:
            _, metadata = fetch_profile(40.88, -119.08, "2026-09-26T14:43",
                                        "America/Los_Angeles", "historical")
        self.assertIn("start_hour=2026-09-26T15%3A00", get.call_args.args[0])
        self.assertEqual(metadata["input_local_time"], "2026-09-26T14:43")
        self.assertIn("15:00", metadata["local_time"])

    def test_prepare_and_run_pressure_screen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("Vibes.openmeteo_profile._get_json", return_value=example_response()):
                weather, metadata = fetch_profile(40.88, -119.08, "2026-09-26T14:00",
                                                  "America/Los_Angeles", "historical")
            weather_path, _ = save_profile(weather, metadata, root / "weather.csv")
            times = np.arange(0, 5.01, 0.1)
            flight = pd.DataFrame({
                "time": times, "x": 0, "y": 0, "z": times * 20,
                "vx": 0, "vy": 0, "vz": 20,
                "e0": 1, "e1": 0, "e2": 0, "e3": 0,
            })
            flight_path = root / "flight.csv"
            flight.to_csv(flight_path, index=False)
            motor = pd.DataFrame({"time_s": [0, 1, 2], "thrust_N": [100, 100, 0]})
            motor_path = root / "motor.csv"
            motor.to_csv(motor_path, index=False)
            prepared, notes = prepare_vibration_trajectory(
                flight_path, weather_path, motor_curve_path=motor_path,
                exhaust_velocity_mps=2000,
            )
            self.assertAlmostEqual(prepared.loc[0, "velocity_mps"], 20)
            self.assertGreater(prepared.loc[0, "mach"], 0)
            self.assertEqual(prepared.loc[0, "thrust_N"], 100)
            self.assertEqual(prepared.loc[30, "thrust_N"], 0)
            self.assertIn("wind NOT subtracted", notes["speed_basis"])
            values = {
                "trajectory": str(flight_path), "hr": "", "lr": "",
                "end_time": "", "provenance": "simulated",
                "latitude": "40.88", "longitude": "-119.08",
                "local_time": "2026-09-26T14:00", "timezone": "America/Los_Angeles",
                "mode": "historical", "elevation": "",
                "motor_curve": str(motor_path), "exhaust_velocity": "2000",
                "ignition_time": "0", "length": "3", "station": "1",
                "motor_mass": "5", "avionics_mass": "1", "damping": "0.03",
                "transfer": "", "motor_psd": "", "ref_thrust": "",
                "ref_ve": "", "ref_mass": "", "cq": "0.02",
                "frequency_max": "2000", "output_dir": str(root),
            }
            rsf_path = save_rsf(values, root / "setup.RSF",
                                options={"window_s": 0.5, "frequency_min_hz": 25})
            values, options, warnings = load_rsf(rsf_path)
            self.assertEqual(warnings, [])
            with patch("Vibes.vibration_workflow.fetch_profile", return_value=(weather, metadata)), \
                 patch("Vibes.vibration_workflow.preliminary_vibration.run",
                       wraps=preliminary_vibration.run) as run:
                run_dir = run_workflow("complete", values, lambda message: None, options=options)
            manifest = json.loads((run_dir / "run_inputs.json").read_text(encoding="utf-8"))
            actual_args = {
                key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(run.call_args.args[0]).items()
            }
            self.assertEqual(manifest["preliminary_vibration_inputs"], actual_args)
            self.assertEqual(actual_args["window_s"], 0.5)
            self.assertEqual(actual_args["frequency_min_hz"], 25)
            self.assertEqual(manifest["all_gui_field_values"], values)
            self.assertEqual(manifest["resolved_files_and_weather"]["weather_metadata"], metadata)
            self.assertTrue((run_dir / "run_inputs.md").exists())
            output = run_dir / "vibration"
            self.assertTrue((output / "summary.md").exists())
            self.assertTrue((output / "powered_ascent_envelope.csv").exists())

    def test_supplied_blue_raven_ascent_can_be_prepared(self):
        flight_path = Path(__file__).parent / "blue_raven_output" / "flight_trajectory.csv"
        if not flight_path.exists():
            self.skipTest("Converted example trajectory not present")
        with tempfile.TemporaryDirectory() as temp:
            weather_path = Path(temp) / "synthetic_weather_for_test.csv"
            pd.DataFrame({
                "altitude_m_asl": [0, 10000, 25000],
                "pressure_pa": [101325, 26000, 2500],
                "temperature_k": [288, 230, 220],
                "air_density_kgm3": [1.225, 0.394, 0.04],
                "speed_of_sound_mps": [340, 304, 297],
                "wind_u_mps": [0, 0, 0],
                "wind_v_mps": [0, 0, 0],
            }).to_csv(weather_path, index=False)
            prepared, _ = prepare_vibration_trajectory(
                flight_path, weather_path, end_time_s=23.3,
            )
            self.assertGreater(len(prepared), 1000)
            self.assertTrue(np.isfinite(prepared.to_numpy()).all())
            self.assertTrue((prepared["altitude_m"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
